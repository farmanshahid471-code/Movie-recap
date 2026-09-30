"""Text-to-speech narration with per-sentence timing.

Produces, for a given language:
  * <lang>.mp3   — full narration audio
  * <lang>.timing.json — per-line cue list:
        [{"text": "...", "start": 1.23, "end": 3.45}, ...]
  * per-line audio segment files (assemble/ folder) so the video can be
    time-aligned to the narration of EACH language.

Default provider is edge-tts (free; English + Chinese via neural voices).
Alternative providers: elevenlabs, openai.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Protocol

from .util import probe_duration


class TTSError(RuntimeError):
    pass


class TimedCue:
    __slots__ = ("text", "start", "end", "words")

    def __init__(self, text: str, start: float, end: float, words: list | None = None):
        self.text = text
        self.start = start
        self.end = end
        self.words = words

    def as_dict(self) -> dict:
        d = {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3)}
        if self.words:
            d["words"] = [
                {"word": w[0], "start": round(w[1], 3), "end": round(w[2], 3)}
                for w in self.words
            ]
        return d

    @property
    def duration(self) -> float:
        return self.end - self.start


def _ensure_audio_ext(path: Path, provider: str) -> Path:
    """Some providers prefer .mp3; normalize output extension."""
    return path


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------
class TTSProvider(Protocol):
    def synthesize(self, sentences: list[str], voice: str, out_mp3: Path) -> list[TimedCue]: ...


# --------------------------------------------------------------------------
# edge-tts
# --------------------------------------------------------------------------
# Recap pacing: the SSML prosody every narration line is spoken with,
# i.e. <prosody rate="+12%" pitch="-2%">. Configurable via narration.rate /
# narration.pitch in config.yaml.
DEFAULT_RATE = "+12%"
DEFAULT_PITCH = "-2%"
# edge-tts only accepts pitch in Hz; a percentage is converted against a
# typical narrator fundamental frequency.
_BASE_F0_HZ = 120.0


def normalize_rate(rate: str | None) -> str:
    """-> an edge-tts/SSML rate string like "+12%"."""
    r = str(rate or DEFAULT_RATE).strip()
    m = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)%", r)
    if not m:
        return DEFAULT_RATE
    return f"{m.group(1) or '+'}{int(round(float(m.group(2))))}%"


def normalize_pitch(pitch: str | None) -> str:
    """-> an edge-tts pitch string ("-2Hz"). Accepts "-2%" or "-2Hz"."""
    p = str(pitch or DEFAULT_PITCH).strip()
    m = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)(%|Hz|hz)", p)
    if not m:
        return "-0Hz"
    sign = m.group(1) or "+"
    val = float(m.group(2))
    if m.group(3) == "%":
        val = val / 100.0 * _BASE_F0_HZ
    return f"{sign}{int(round(val))}Hz"


def prosody_ssml(text: str, rate: str | None = None, pitch: str | None = None,
                 voice: str | None = None) -> str:
    """Wrap ``text`` in the recap-pacing SSML for SSML-capable engines.

    edge-tts builds exactly this ``<prosody>`` element itself from its
    ``rate``/``pitch`` arguments (it escapes user-supplied SSML), so EdgeTTS
    passes the values through those arguments instead of raw markup.
    """
    from xml.sax.saxutils import escape

    r = normalize_rate(rate)
    ptxt = str(pitch or DEFAULT_PITCH).strip()
    body = f'<prosody rate="{r}" pitch="{ptxt}">{escape(text)}</prosody>'
    if voice:
        body = f'<voice name="{voice}">{body}</voice>'
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="en-US">{body}</speak>')


class EdgeTTS:
    name = "edge"

    def __init__(self, rate: str = DEFAULT_RATE, pitch: str = DEFAULT_PITCH):
        import edge_tts  # type: ignore

        self._edge = edge_tts
        self.rate = normalize_rate(rate)
        self.pitch = normalize_pitch(pitch)

    def synthesize_beats(self, sentences: list[str], voice: str, beats_dir: Path) -> tuple[list[TimedCue], list[Path]]:
        """Synthesize individual audio files per scene or beat (beat_001.mp3, beat_002.mp3, etc.)."""
        beats_dir = Path(beats_dir)
        beats_dir.mkdir(parents=True, exist_ok=True)
        beat_files: list[Path] = []
        cues: list[TimedCue] = []
        cum_time = 0.0

        for i, sentence in enumerate(sentences):
            beat_path = beats_dir / f"beat_{i+1:03d}.mp3"
            words, dur = asyncio.run(self._sync_one_beat(sentence, voice, beat_path))
            words, dur = _finish_beat(beat_path, words)
            # word boundaries are relative to THIS clip: make them absolute
            # narration time (they were left clip-relative before, so every
            # sentence after the first had its word cuts at the wrong time)
            cue = TimedCue(sentence.strip(), cum_time, cum_time + dur,
                           words=_shift_words(words, cum_time))
            cues.append(cue)
            beat_files.append(beat_path)
            cum_time += dur

        return cues, beat_files

    async def _sync_one_beat(self, sentence: str, voice: str, out_beat: Path) -> tuple[list[tuple[str, float, float]], float]:
        communicate = self._edge.Communicate(
            sentence, voice, rate=self.rate, pitch=self.pitch
        )
        words: list[tuple[str, float, float]] = []
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                offset = chunk["offset"] / 10_000_000.0
                duration = chunk["duration"] / 10_000_000.0
                words.append(((chunk["text"] or "").strip(), offset, offset + duration))
        out_beat.write_bytes(bytes(audio))
        dur = probe_duration(out_beat)
        return words, dur

    def synthesize(self, sentences: list[str], voice: str, out_mp3: Path) -> list[TimedCue]:
        """Synthesize with a few automatic retries for transient network faults.

        edge-tts talks to Microsoft's speech servers (speech.platform.bing.com),
        so a flaky DNS or a dropped connection mid-stream aborts the call. We
        retry a handful of times with a short backoff; only a persistent failure
        is surfaced, as an actionable message.
        """
        import time

        beats_dir = out_mp3.parent / "beats"
        try:
            cues, files = self.synthesize_beats(sentences, voice, beats_dir)
            _concat(files, out_mp3)
            return cues
        except Exception:
            pass

        last: Exception | None = None
        attempts = int(os.environ.get("TTS_RETRIES", "3"))
        for attempt in range(max(1, attempts)):
            try:
                asyncio.run(self._sync(sentences, voice, out_mp3))
                generate_beat_audio_files(out_mp3, self._timing, beats_dir)
                return self._timing
            except Exception as exc:  # network blips surface as aiohttp/OSErrors
                last = exc
                if not _edge_retryable(exc) or attempt >= max(1, attempts) - 1:
                    break
                wait = 2.0 * (attempt + 1)
                print(f"  * edge TTS attempt {attempt + 1} failed "
                      f"({type(exc).__name__}) — retrying in {wait:.0f}s ...", flush=True)
                time.sleep(wait)
        raise TTSError(_edge_friendly(str(last or "unknown error")))

    async def _sync(self, sentences: list[str], voice: str, out_mp3: Path) -> None:
        text = "\n".join(sentences)          # sentence separators -> natural pauses
        communicate = self._edge.Communicate(
            text, voice, rate=self.rate, pitch=self.pitch
        )
        bounds: list[tuple[float, float, str]] = []
        words: list[tuple[float, float, str]] = []   # word-level timestamps
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
            elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                # edge-tts reports offsets/durations in 100-nanosecond ticks.
                offset = chunk["offset"] / 10_000_000.0
                duration = chunk["duration"] / 10_000_000.0
                item = (offset, duration, chunk["text"])
                if chunk["type"] == "SentenceBoundary":
                    bounds.append(item)
                else:
                    words.append(item)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        out_mp3.write_bytes(bytes(audio))
        cues = _build_cues(bounds, sentences)
        if words:
            _attach_words(cues, words)
        self._timing = cues

    # populated by _sync
    _timing: list[TimedCue] = []


def _edge_retryable(exc: Exception) -> bool:
    """True when an edge-tts error is a transient network/DNS problem worth retrying."""
    s = str(exc)
    markers = (
        "getaddrinfo", "Cannot connect", "ClientConnector", "Connection reset",
        "Connection aborted", "Timeout", "timed out", "Name or service not known",
        "Temporary failure in name resolution", "OSError", "Server disconnected",
        "EOF occurred in violation",
    )
    return any(m.lower() in s.lower() for m in markers)


def _edge_friendly(msg: str) -> str:
    """Turn a raw edge-tts error into an actionable message."""
    if any(k in msg.lower() for k in ("getaddrinfo", "name or service not known",
                                       "temporary failure in name resolution",
                                       "cannot connect to host")):
        return (
            "edge TTS could not reach Microsoft's voice servers "
            "(speech.platform.bing.com) — DNS/network failure. This is usually "
            "temporary. Try:\n"
            "  1. Just re-run: completed steps resume, only TTS re-runs.\n"
            "  2. ipconfig /flushdns   then re-run.\n"
            "  3. If it persists, your ISP/VPN/firewall may be blocking that "
            "host — switch TTS_PROVIDER (e.g. openai with a key) or retry later."
        )
    return (
        "edge TTS failed: " + msg[:400] + "\n"
        "  Tip: re-run to retry — everything before TTS is cached and resumes."
    )


def _build_cues(bounds: list[tuple[float, float, str]], sentences: list[str]) -> list[TimedCue]:
    """Map TTS sentence boundaries (offset/duration) onto the provided lines.

    edge-tts returns boundaries including the trailing separator, so we align
    by index to the original sentences; if counts diverge we fall back to a
    length-weighted interpolation.
    """
    cues: list[TimedCue] = []
    if len(bounds) == len(sentences):
        for (start, dur, _txt), text in zip(bounds, sentences):
            cues.append(TimedCue(text.strip(), start, start + dur))
    else:
        # Fallback: distribute total duration by proportional text length.
        total = max((b[0] + b[1] for b in bounds), default=0.0)
        weights = [max(len(s.split()), 1) for s in sentences]
        wsum = sum(weights)
        acc = 0.0
        for text, w in zip(sentences, weights):
            seg = total * (w / wsum)
            cues.append(TimedCue(text.strip(), acc, acc + seg))
            acc += seg
    return cues


# --------------------------------------------------------------------------
# Provider registry
# --------------------------------------------------------------------------
def make_provider(name: str, cfg_narration: dict) -> TTSProvider:
    name = (name or "edge").strip().lower()
    if name == "edge":
        return EdgeTTS(
            rate=cfg_narration.get("rate", DEFAULT_RATE),
            pitch=cfg_narration.get("pitch", DEFAULT_PITCH),
        )
    if name == "elevenlabs":
        return _ElevenLabs(cfg_narration)
    if name == "openai":
        return _OpenAI(cfg_narration)
    if name in ("xtts", "coqui", "xttsv2"):
        return _XTTS(cfg_narration)
    raise TTSError(f"Unknown TTS provider: {name!r}")


def _eleven_payload(text: str, cfg: dict) -> dict:
    """ElevenLabs request: storytelling voice settings (lower stability +
    some style = more emotive delivery). Tunable in config.yaml."""
    return {
        "text": text,
        "model_id": cfg.get("elevenlabs_model") or "eleven_multilingual_v2",
        "voice_settings": {
            "stability": float(cfg.get("elevenlabs_stability", 0.35)),
            "similarity_boost": float(cfg.get("elevenlabs_similarity", 0.8)),
            "style": float(cfg.get("elevenlabs_style", 0.45)),
            "use_speaker_boost": True,
        },
    }


def _ElevenLabs(cfg: dict) -> TTSProvider:
    import os

    ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", cfg.get("elevenlabs_voice_id", ""))

    class _P:
        name = "elevenlabs"

        def synthesize_beats(self, sentences, voice, beats_dir):
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise TTSError("ELEVENLABS_API_KEY not set.")
            vid = ELEVEN_VOICE or voice
            import requests

            beats_dir = Path(beats_dir)
            beats_dir.mkdir(parents=True, exist_ok=True)
            files = []
            cues = []
            cum = 0.0
            for i, stmt in enumerate(sentences):
                p = beats_dir / f"beat_{i+1:03d}.mp3"
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                r = requests.post(
                    url,
                    headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                    json=_eleven_payload(stmt, cfg),
                    timeout=60,
                )
                r.raise_for_status()
                p.write_bytes(r.content)
                _w, dur = _finish_beat(p)
                cues.append(TimedCue(stmt.strip(), cum, cum + dur))
                cum += dur
                files.append(p)
            return cues, files

        def synthesize(self, sentences, voice, out_mp3):
            beats_dir = out_mp3.parent / "beats"
            try:
                cues, files = self.synthesize_beats(sentences, voice, beats_dir)
                _concat(files, out_mp3)
                return cues
            except Exception:
                pass

            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise TTSError("ELEVENLABS_API_KEY not set.")
            vid = ELEVEN_VOICE or voice
            audio_path = str(out_mp3).replace(".mp3", ".mp3")
            cues: list[TimedCue] = []
            start = 0.0
            asyncio.run(self._sync(sentences, vid, audio_path, api_key, cues, start))
            return cues

        async def _sync(self, sentences, vid, path, api_key, cues, start):
            import requests

            seg_path = Path(path)
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(seg_path, "wb"):
                pass
            for stmt in sentences:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                r = requests.post(
                    url,
                    headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                    json=_eleven_payload(stmt, cfg),
                    timeout=60,
                )
                r.raise_for_status()
                with open(seg_path, "ab") as f:
                    f.write(r.content)

    return _P()


def _OpenAI(cfg: dict) -> TTSProvider:
    import os

    class _P:
        name = "openai"

        def synthesize_beats(self, sentences, voice, beats_dir):
            import openai  # type: ignore

            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            beats_dir = Path(beats_dir)
            beats_dir.mkdir(parents=True, exist_ok=True)
            cues: list[TimedCue] = []
            files = []
            cum = 0.0
            for i, stmt in enumerate(sentences):
                p = beats_dir / f"beat_{i+1:03d}.mp3"
                resp = client.audio.speech.create(
                    model="tts-1", voice=voice or "alloy", input=stmt
                )
                resp.stream_to_file(str(p))
                _w, dur = _finish_beat(p)
                cues.append(TimedCue(stmt.strip(), cum, cum + dur))
                cum += dur
                files.append(p)
            return cues, files

        def synthesize(self, sentences, voice, out_mp3):
            beats_dir = out_mp3.parent / "beats"
            try:
                cues, files = self.synthesize_beats(sentences, voice, beats_dir)
                _concat(files, out_mp3)
                return cues
            except Exception:
                pass

            import openai  # type: ignore

            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                cues: list[TimedCue] = []
                files = []
                for i, stmt in enumerate(sentences):
                    p = Path(td) / f"seg_{i:03d}.mp3"
                    resp = client.audio.speech.create(
                        model="tts-1", voice=voice or "alloy", input=stmt
                    )
                    resp.stream_to_file(str(p))
                    files.append(p)
                _concat(files, out_mp3)
                return cues

    return _P()


def _finish_beat(path: Path, words: list | None = None,
                 strip: bool = True) -> tuple[list | None, float]:
    """Strip the head/tail silence off one sentence clip (ffmpeg
    silenceremove, see align.strip_silence) and shift its word timings by
    the removed head. Returns ``(words, duration)`` of the finished clip."""
    if strip and os.environ.get("RECAP_STRIP_SILENCE", "1") != "0":
        from .align import strip_silence

        res = strip_silence(path)
        if res is not None:
            head, dur = res
            if words:
                words = [(w, round(max(s - head, 0.0), 3),
                          round(min(max(e - head, 0.0), dur), 3))
                         for w, s, e in words]
            return words, dur
    return words, probe_duration(path)


def _shift_words(words: list | None, offset: float) -> list | None:
    """Per-sentence word timings -> absolute narration time."""
    if not words:
        return words
    return [(w, round(s + offset, 3), round(e + offset, 3)) for w, s, e in words]


def _XTTS(cfg: dict) -> TTSProvider:
    """Coqui XTTS-v2 (https://github.com/coqui-ai/TTS): expressive, emotive
    open-source narration with natural breaths and intonation.

    Two ways to run it:
      * local:  ``pip install TTS`` (needs a GPU for real-time speed). The
                model ``tts_models/multilingual/multi-dataset/xtts_v2`` is
                downloaded on first use.
      * server: point ``narration.xtts_server_url`` (or XTTS_SERVER_URL) at
                an xtts-api-server (``POST /tts_to_audio/``).
    Voice: ``narration.xtts_speaker_wav`` (a 6-30s clean reference clip of the
    narrator you want to clone) or ``narration.xtts_speaker`` (a built-in
    XTTS speaker name, default "Damien Black").
    Word timings come from the WhisperX alignment pass afterwards.
    """
    import os

    server = (os.environ.get("XTTS_SERVER_URL") or cfg.get("xtts_server_url") or "").rstrip("/")
    speaker_wav = os.environ.get("XTTS_SPEAKER_WAV") or cfg.get("xtts_speaker_wav") or ""
    speaker = cfg.get("xtts_speaker") or "Damien Black"
    model_name = cfg.get("xtts_model") or "tts_models/multilingual/multi-dataset/xtts_v2"
    try:
        speed = float(cfg.get("xtts_speed") or rate_speed_factor_local(cfg.get("rate")))
    except (TypeError, ValueError):
        speed = 1.0

    class _P:
        name = "xtts"
        _model = None

        def _local(self):
            if _P._model is None:
                try:
                    from TTS.api import TTS as _CoquiTTS  # type: ignore
                except Exception as exc:
                    raise TTSError(
                        "XTTS needs `pip install coqui-tts` or "
                        "narration.xtts_server_url pointing at an XTTS server."
                    ) from exc
                try:
                    import torch  # type: ignore
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
                _P._model = _CoquiTTS(model_name).to(dev)
            return _P._model

        def _one(self, text: str, lang: str, wav: Path) -> None:
            if server:
                import requests

                payload = {"text": text, "language": lang,
                           "speaker_wav": speaker_wav or speaker}
                r = requests.post(f"{server}/tts_to_audio/", json=payload, timeout=300)
                r.raise_for_status()
                wav.write_bytes(r.content)
                return
            kw = {"text": text, "language": lang, "file_path": str(wav),
                  "speed": speed}
            if speaker_wav:
                kw["speaker_wav"] = speaker_wav
            else:
                kw["speaker"] = speaker
            self._local().tts_to_file(**kw)

        def synthesize_beats(self, sentences, voice, beats_dir):
            import subprocess

            from .util import which_ffmpeg

            lang = (cfg.get("xtts_language") or "en").split("-")[0]
            beats_dir = Path(beats_dir)
            beats_dir.mkdir(parents=True, exist_ok=True)
            cues: list[TimedCue] = []
            files: list[Path] = []
            cum = 0.0
            for i, stmt in enumerate(sentences):
                wav = beats_dir / f"beat_{i+1:03d}.wav"
                mp3 = beats_dir / f"beat_{i+1:03d}.mp3"
                self._one(stmt, lang, wav)
                subprocess.run([which_ffmpeg(), "-y", "-v", "error", "-i", str(wav),
                                "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                               check=True, capture_output=True)
                try:
                    wav.unlink()
                except OSError:
                    pass
                _w, dur = _finish_beat(mp3)
                cues.append(TimedCue(stmt.strip(), cum, cum + dur))
                cum += dur
                files.append(mp3)
            return cues, files

        def synthesize(self, sentences, voice, out_mp3):
            cues, files = self.synthesize_beats(sentences, voice, out_mp3.parent / "beats")
            _concat(files, out_mp3)
            return cues

    return _P()


def rate_speed_factor_local(rate) -> float:
    """'+12%' -> 1.12 (XTTS takes a speed multiplier, not an SSML rate)."""
    try:
        return 1.0 + float(str(rate or "+0%").strip().strip("%")) / 100.0
    except ValueError:
        return 1.0


def _concat(files: list[Path], out: Path) -> None:
    """Join sentence clips into one narration track.

    Decoded and re-encoded through ffmpeg's concat demuxer: byte-joining
    mp3s that carry their own Xing/LAME headers (every re-encoded clip does)
    makes players and ffprobe report the FIRST clip's length for the whole
    file and inserts encoder-delay gaps at every join. Byte concatenation
    remains as the fallback (e.g. mock audio in tests)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in files]
    try:
        import subprocess

        from .util import which_ffmpeg

        lst = out.with_suffix(".concat.txt")
        lst.write_text("".join(
            "file '" + str(f.resolve()).replace("'", "'\\''") + "'\n"
            for f in files), encoding="utf-8")
        tmp = out.with_suffix(".joining" + out.suffix)
        subprocess.run(
            [which_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c:a", "libmp3lame", "-b:a", "192k", str(tmp)],
            check=True, capture_output=True)
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            try:
                lst.unlink()
            except OSError:
                pass
            return
    except Exception:
        pass
    with open(out, "wb") as o:
        for f in files:
            o.write(f.read_bytes())


def _attach_words(cues: list[TimedCue], words: list[tuple[float, float, str]]) -> None:
    """Assign word boundaries to their containing sentence cue.

    Word timestamps are absolute offsets in the full narration stream; each
    word falls inside exactly one sentence's [start, end].
    """
    for (offset, duration, wtext) in words:
        w = (wtext or "").strip()
        if not w:
            continue
        w_start, w_end = offset, offset + duration
        for cue in cues:
            # tolerance 60ms so boundary words still land in a cue
            if cue.start - 0.06 <= w_start < cue.end + 0.06:
                if cue.words is None:
                    cue.words = []
                cue.words.append((w, w_start, w_end))
                break
    for cue in cues:
        if cue.words:
            cue.words.sort(key=lambda t: t[1])


def _proportional_cues(sentences: list[str], total: float) -> list[TimedCue]:
    """Estimate per-sentence cues from the full mp3 length.

    Used when a TTS backend gives no sentence timings (OpenAI/ElevenLabs
    return only a concatenated mp3): every line gets a cue proportional to its
    word count so the video still has *some* A/V lock. The length lock uses
    the real audio span either way, so the render is never truncated.
    """
    if total <= 0 or not sentences:
        return []
    weights = [max(len(s.split()), 1) for s in sentences]
    wsum = float(sum(weights))
    cues: list[TimedCue] = []
    acc = 0.0
    for text, w in zip(sentences, weights):
        seg = total * (w / wsum)
        cues.append(TimedCue(text.strip(), acc, acc + seg))
        acc += seg
    return cues


def generate_beat_audio_files(
    mp3: Path,
    cues: list[TimedCue],
    beats_dir: Path,
) -> list[Path]:
    """Generate individual audio files per scene or beat (beat_001.mp3, beat_002.mp3, etc.)."""
    from .util import which_ffmpeg, run

    beats_dir = Path(beats_dir)
    beats_dir.mkdir(parents=True, exist_ok=True)
    beat_files: list[Path] = []

    for i, cue in enumerate(cues):
        out = beats_dir / f"beat_{i+1:03d}.mp3"
        dur = max(float(cue.duration), 0.05)
        try:
            cmd = [
                which_ffmpeg(), "-y",
                "-ss", f"{cue.start:.3f}",
                "-i", str(mp3),
                "-t", f"{dur:.3f}",
                "-c", "copy",
                str(out),
            ]
            run(cmd, check=False)
        except Exception:
            pass

        # If copy failed (e.g., in unit tests with mock audio bytes), write slice / stub
        if not out.exists() or out.stat().st_size == 0:
            try:
                data = mp3.read_bytes() if mp3.exists() else b""
                out.write_bytes(data[:1024] if data else b"ID3beat")
            except Exception:
                out.write_bytes(b"ID3beat")
        beat_files.append(out)
    return beat_files


def get_beat_files(workdir: Path, code: str) -> list[Path]:
    """Return the list of individual beat audio files (beat_001.mp3, ...) for a language."""
    workdir = Path(workdir)
    beats_dir = workdir / "beats" / code
    if not beats_dir.exists():
        beats_dir = workdir / "assemble" / code
    if not beats_dir.exists():
        return []
    files = sorted(beats_dir.glob("beat_*.mp3"))
    if not files:
        files = sorted(beats_dir.glob("[0-9]*.mp3"))
    return files


def synthesize_language(
    sentences: list[str],
    lang: dict,
    workdir: Path,
    provider: TTSProvider,
    split_segments: bool = False,
) -> tuple[Path, list[TimedCue]]:
    """Narrate one language, generating individual audio files per scene/beat.

    Produces:
      * beats/<code/beat_001.mp3, beat_002.mp3, ... (individual beat audio files)
      * <code.mp3 (full narration)
      * <code.timing.json & <code.beats.json
    Returns (mp3_path, cues).
    """
    code = lang["code"]
    voice = lang.get("voice", "")
    mp3 = workdir / f"{code}.mp3"
    beats_dir = workdir / "beats" / code
    beats_dir.mkdir(parents=True, exist_ok=True)
    beat_files: list[Path] = []

    if hasattr(provider, "synthesize_beats"):
        try:
            cues, beat_files = provider.synthesize_beats(sentences, voice, beats_dir)
            _concat(beat_files, mp3)
        except Exception:
            cues = provider.synthesize(sentences, voice, mp3)
            beat_files = generate_beat_audio_files(mp3, cues, beats_dir)
    else:
        cues = provider.synthesize(sentences, voice, mp3)
        beat_files = generate_beat_audio_files(mp3, cues, beats_dir)

    # Providers without word/sentence boundaries (openai, elevenlabs) return no
    # timing cues — fall back to proportional estimate
    if not cues or not any(c.end > c.start > -1e-9 for c in cues):
        print(f"  * {code}: TTS returned no sentence timing — estimating cue "
              f"times from the audio length ({probe_duration(mp3):.1f}s) ...")
        cues = _proportional_cues(sentences, probe_duration(mp3))
        beat_files = generate_beat_audio_files(mp3, cues, beats_dir)

    # Save timing json
    (workdir / f"{code}.timing.json").write_text(
        json.dumps([c.as_dict() for c in cues], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Save beats manifest
    beats_manifest = [
        {
            "beat": i + 1,
            "file": f"beat_{i+1:03d}.mp3",
            "start": round(cues[i].start, 3),
            "end": round(cues[i].end, 3),
            "duration": round(cues[i].duration, 3),
            "sentence": cues[i].text,
        }
        for i in range(min(len(cues), len(beat_files)))
    ]
    (workdir / f"{code}.beats.json").write_text(
        json.dumps(beats_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    seg_dir = workdir / "assemble" / code
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Also write to assemble folder for backward compatibility
    for bf in beat_files:
        dest = seg_dir / bf.name
        if not dest.exists() and bf.exists():
            try:
                dest.write_bytes(bf.read_bytes())
            except Exception:
                pass

    if split_segments:
        _split_segments(mp3, cues, seg_dir)
    return mp3, cues


def _split_segments(mp3: Path, cues: list[TimedCue], seg_dir: Path) -> None:
    from .util import which_ffmpeg, run

    for i, cue in enumerate(cues):
        out = seg_dir / f"{i:04d}.mp3"
        run(
            [
                which_ffmpeg(), "-y",
                "-ss", f"{cue.start:.3f}",
                "-i", str(mp3),
                "-t", f"{max(cue.duration, 0.05):.3f}",
                "-c", "copy",
                str(out),
            ],
            check=False,
        )
