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
from pathlib import Path
from typing import Protocol

from .util import fmt_ts_ass


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
class EdgeTTS:
    name = "edge"

    def __init__(self, rate: str = "+0%"):
        import edge_tts  # type: ignore

        self._edge = edge_tts
        self.rate = rate

    def synthesize(self, sentences: list[str], voice: str, out_mp3: Path) -> list[TimedCue]:
        asyncio.run(self._sync(sentences, voice, out_mp3))
        return self._timing

    async def _sync(self, sentences: list[str], voice: str, out_mp3: Path) -> None:
        text = "\n".join(sentences)          # sentence separators -> natural pauses
        communicate = self._edge.Communicate(text, voice, rate=self.rate)
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
        return EdgeTTS(rate=cfg_narration.get("rate", "+0%"))
    if name == "elevenlabs":
        return _ElevenLabs(cfg_narration)
    if name == "openai":
        return _OpenAI(cfg_narration)
    raise TTSError(f"Unknown TTS provider: {name!r}")


def _ElevenLabs(cfg: dict) -> TTSProvider:
    import os

    ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", cfg.get("elevenlabs_voice_id", ""))

    class _P:
        name = "elevenlabs"

        def synthesize(self, sentences, voice, out_mp3):
            import requests  # type: ignore

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
            # ElevenLabs is POST-only per utterance; synth sequentially.
            import requests

            for stmt in sentences:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                r = requests.post(
                    url,
                    headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                    json={"text": stmt, "model_id": "eleven_multilingual_v2"},
                    timeout=60,
                )
                r.raise_for_status()
                seg_path = Path(path)
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                with open(seg_path, "ab") as f:
                    f.write(r.content)

    return _P()


def _OpenAI(cfg: dict) -> TTSProvider:
    import os

    class _P:
        name = "openai"

        def synthesize(self, sentences, voice, out_mp3):
            import openai  # type: ignore

            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                cues: list[TimedCue] = []
                start = 0.0
                files = []
                for i, stmt in enumerate(sentences):
                    p = Path(td) / f"seg_{i:03d}.mp3"
                    resp = client.audio.speech.create(
                        model="tts-1", voice=voice or "alloy", input=stmt
                    )
                    resp.stream_to_file(str(p))
                    files.append(p)
                # Concatenate to full mp3
                _concat(files, out_mp3)
                return cues

    return _P()


def _concat(files: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
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


def synthesize_language(
    sentences: list[str],
    lang: dict,
    workdir: Path,
    provider: TTSProvider,
    split_segments: bool = False,
) -> tuple[Path, list[TimedCue]]:
    """Narrate one language. Returns (mp3_path, cues)."""
    code = lang["code"]
    voice = lang.get("voice", "")
    mp3 = workdir / f"{code}.mp3"
    cues = provider.synthesize(sentences, voice, mp3)
    seg_dir = workdir / "assemble" / code
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Save timing json
    (workdir / f"{code}.timing.json").write_text(
        json.dumps([c.as_dict() for c in cues], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Optional: split the narration into per-sentence audio clips
    # (e.g. if you want to replace individual takes or re-time scenes).
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
