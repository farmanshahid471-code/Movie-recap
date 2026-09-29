"""Step C+ — lock the visuals to the SPOKEN narration with WhisperX forced alignment.

Why this module exists
----------------------
The timeline locks each narration sentence to a film window, and the beat
boundaries come from the TTS cue times. But cue times are *estimates* unless
the TTS backend reports word boundaries (only edge-tts does; OpenAI and
ElevenLabs return a bare mp3, and the old fallback guessed cue times
proportionally by word count — drifting seconds away from the voice).

The fix: run the GENERATED narration audio through WhisperX forced alignment.
WhisperX utilizes Voice Activity Detection (VAD) and phoneme-level forced
alignment to guarantee highly accurate word timestamps:

* every sentence cue start becomes the true start of its first spoken word,
* every cue carries word-level timestamps with phoneme-level accuracy,
* outputs a structured JSON with precise start/end times for every word,
  eliminating drift entirely.

Degrades gracefully: if whisperx is unavailable, falls back to faster-whisper,
and if transcription fails or word-to-sentence match is poor, keeps the
provider's own cues — the pipeline never breaks because of this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .tts import TimedCue

# Word a narrator naturally has before a clause word where a cut feels good.
# Whisper word starts run a few tens of milliseconds late (onset detection),
# so switching the picture this much earlier makes the cut land *with* the
# voice instead of a hair after it.
DEFAULT_AUDIO_PRE_ROLL = 0.1

# Forced-alignment model. The whisperx default (WAV2VEC2_ASR_BASE_960H for
# English) struggles with synthetic TTS voices; the large LV60K model keeps
# the word locks tight. Override with RECAP_ALIGN_MODEL ("" = whisperx
# default). Only used for English; other languages use whisperx's defaults.
DEFAULT_ALIGN_MODEL = "WAV2VEC2_ASR_LARGE_LV60K_960H"


def _align_model_name(lang: str) -> str | None:
    import os
    name = os.environ.get("RECAP_ALIGN_MODEL", DEFAULT_ALIGN_MODEL).strip()
    if not name or not (lang or "en").lower().startswith("en"):
        return None
    return name


def _load_align_model(whisperx, lang: str, dev: str):
    """whisperx.load_align_model with the large wav2vec2 model, falling back
    to whisperx's default model if it cannot be loaded."""
    name = _align_model_name(lang)
    if name:
        try:
            return whisperx.load_align_model(
                language_code=lang, device=dev, model_name=name)
        except Exception as exc:
            print(f"  ! whisperx align model {name} unavailable ({exc}); "
                  "using the whisperx default", flush=True)
    return whisperx.load_align_model(language_code=lang, device=dev)


_PUNCT_STRIP = re.compile(r"[^\w'\u4e00-\u9fff\u0600-\u06ff\u00c0-\u024f]+")
# CJK punctuation that marks a clause boundary even without spaces.
_CJK_BOUNDARY = "，。！？；：、）】」』"


def _norm(token: str) -> str:
    """Normalize one spoken token for matching (case/punctuation insensitive)."""
    t = (token or "").strip().lower()
    t = _PUNCT_STRIP.sub("", t)
    return t


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def narration_words_whisperx(
    mp3: Path,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> list[tuple[str, float, float]] | None:
    """Run WhisperX with VAD and phoneme-level forced alignment.

    Utilizes Voice Activity Detection (VAD) and phoneme-level alignment
    to guarantee highly accurate word timestamps, eliminating drift.
    """
    try:
        import whisperx  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None

    try:
        import os
        dev = "cuda" if (device == "cuda" or (device in ("auto", None) and torch.cuda.is_available())) else "cpu"
        # On Windows CPU, PyTorch/torchcodec has missing C++ DLLs (libtorchcodec_core.dll)
        # and pyannote crashes Python with code 255. Fallback safely to faster-whisper.
        if dev == "cpu" and os.name == "nt":
            return None
        compute_type = "float16" if dev == "cuda" else "int8"

        # 1. Load audio
        audio = whisperx.load_audio(str(mp3))

        # 2. Transcribe with VAD
        try:
            model = whisperx.load_model(
                model_size,
                device=dev,
                compute_type=compute_type,
                language=language,
            )
        except Exception:
            model = whisperx.load_model(
                model_size,
                device=dev,
                compute_type="float32",
                language=language,
            )
        result = model.transcribe(audio, batch_size=16, language=language)

        # 3. Phoneme-level forced alignment
        align_lang = result.get("language") or language or "en"
        try:
            model_a, metadata = _load_align_model(whisperx, align_lang, dev)
            aligned = whisperx.align(
                result.get("segments", []),
                model_a,
                metadata,
                audio,
                dev,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", [])
        except Exception as align_err:
            print(f"  ! whisperx forced alignment model fallback: {align_err}", flush=True)
            segments = result.get("segments", [])

        words: list[tuple[str, float, float]] = []
        for seg in segments:
            for w in seg.get("words", []):
                wt = (w.get("word") or "").strip()
                if wt and "start" in w and "end" in w:
                    words.append((wt, float(w["start"]), float(w["end"])))
        return words or None
    except Exception as e:
        print(f"  ! whisperx alignment failed ({type(e).__name__}: {e}), trying fallback...", flush=True)
        return None


def narration_words_whisper(
    mp3: Path,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> list[tuple[str, float, float]] | None:
    """Fallback: transcribe with faster-whisper when whisperx is not available."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None
    try:
        model = WhisperModel(model_size, device=device or "auto", compute_type="int8")
        segments, _info = model.transcribe(
            str(mp3),
            language=language,
            beam_size=1,
            word_timestamps=True,
        )
        words: list[tuple[str, float, float]] = []
        for seg in segments:
            for w in (getattr(seg, "words", None) or []):
                wt = (getattr(w, "word", "") or "").strip()
                if wt:
                    words.append((wt, float(w.start), float(w.end)))
        return words or None
    except Exception:
        return None


def narration_words(
    mp3: Path,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> list[tuple[str, float, float]] | None:
    """Extract spoken word timestamps from narration audio using WhisperX forced alignment.

    Swaps out base Whisper for WhisperX (VAD + phoneme alignment).
    Degrades gracefully to faster-whisper if WhisperX is unavailable.
    """
    words = narration_words_whisperx(
        mp3, model_size=model_size, device=device, language=language
    )
    if words:
        return words
    return narration_words_whisper(
        mp3, model_size=model_size, device=device, language=language
    )


def align_with_whisperx(
    audio_path: Path | str,
    language: str | None = None,
    device: str = "auto",
    model_size: str = "small",
) -> dict | None:
    """Run WhisperX end-to-end forced alignment on an audio file.

    Returns the aligned dictionary structure with segments and word timestamps.
    """
    try:
        import whisperx  # type: ignore
        import torch  # type: ignore

        import os
        dev = "cuda" if (device == "cuda" or (device in ("auto", None) and torch.cuda.is_available())) else "cpu"
        if dev == "cpu" and os.name == "nt":
            return None
        compute_type = "float16" if dev == "cuda" else "int8"
        audio = whisperx.load_audio(str(audio_path))
        try:
            model = whisperx.load_model(model_size, device=dev, compute_type=compute_type, language=language)
        except Exception:
            model = whisperx.load_model(model_size, device=dev, compute_type="float32", language=language)
        result = model.transcribe(audio, batch_size=16, language=language)
        align_lang = result.get("language") or language or "en"
        model_a, metadata = _load_align_model(whisperx, align_lang, dev)
        aligned = whisperx.align(result.get("segments", []), model_a, metadata, audio, dev, return_char_alignments=False)
        return aligned
    except Exception as e:
        print(f"  ! align_with_whisperx error: {e}", flush=True)
        return None


def map_words_to_sentences(
    words: list[tuple[str, float, float]],
    sentences: list[str],
) -> list[list[tuple[str, float, float]]] | None:
    """Assign whisper-measured words to the script sentences they belong to.

    Whisper-on-TTS is near-perfect but not byte-perfect: numbers change form
    ("5" vs "five"), punctuation appears and disappears, tokens merge. A
    bounded-lookahead greedy matcher walks both lists, tolerating insertions
    (whisper said something extra) and deletions (a script word went
    untranscribed) on either side.

    Returns one word-list per sentence (empty lists allowed), or ``None`` when
    the match is too poor to trust (below half the sentences matched) — the
    caller then keeps the provider's cues untouched.
    """
    if not words or not sentences:
        return None

    # Flatten the expected tokens, remembering which sentence each belongs to.
    expected: list[tuple[str, int]] = []
    for si, sent in enumerate(sentences):
        for tok in re.findall(r"[\w'\u4e00-\u9fff\u0600-\u06ff]+", sent or ""):
            n = _norm(tok)
            if n:
                expected.append((n, si))

    groups: list[list[tuple[str, float, float]]] = [[] for _ in sentences]
    LOOKAHEAD = 8
    wi = 0  # index into whisper words
    matched_sentences = 0
    last_sent = -1
    clean = 0        # tokens that matched exactly (directly or in lookahead)
    substituted = 0  # tokens accepted as blind substitutions

    for ei, (tok, si) in enumerate(expected):
        if wi >= len(words):
            break
        # Direct match?
        if _norm(words[wi][0]) == tok:
            groups[si].append(words[wi])
            if si != last_sent:
                matched_sentences += 1
                last_sent = si
            clean += 1
            wi += 1
            continue
        # Insertion: whisper produced extra tokens — look ahead for ours.
        found = None
        for k in range(wi + 1, min(wi + 1 + LOOKAHEAD, len(words))):
            if _norm(words[k][0]) == tok:
                found = k
                break
        if found is not None:
            groups[si].append(words[found])
            if si != last_sent:
                matched_sentences += 1
                last_sent = si
            clean += 1
            wi = found + 1
            continue
        # Deletion or substitution: does the NEXT expected token match the
        # current whisper word? Then this script token was skipped.
        if ei + 1 < len(expected) and _norm(words[wi][0]) == expected[ei + 1][0]:
            continue
        # Last resort: accept the mismatch as a substitution.
        groups[si].append(words[wi])
        if si != last_sent:
            matched_sentences += 1
            last_sent = si
        substituted += 1
        wi += 1

    # Quality gate: at least half the sentences located AND the matches must
    # be dominated by real agreement, not blind substitution.
    if matched_sentences < max(1, len(sentences) // 2):
        return None
    if clean == 0 or substituted > clean:
        return None
    return groups


def refine_cues(
    cues: list[TimedCue],
    sentences: list[str],
    words: list[tuple[str, float, float]],
    *,
    audio_pre_roll: float = DEFAULT_AUDIO_PRE_ROLL,
) -> list[TimedCue]:
    """Rewrite cue start/ends from the measured words; fill word timings.

    * A sentence whose words were located starts exactly when its first word
      is spoken (minus a small pre-roll so the picture/subtitle switch lands
      *with* the voice, not after it) and ends at its last word's end.
    * Sentences without a mapping keep the provider's times.
    * Monotonicity is enforced: a refined start can never move before the
      previous sentence's end.
    """
    groups = map_words_to_sentences(words, sentences)
    if groups is None:
        return cues

    out: list[TimedCue] = []
    prev_end = 0.0
    for cue, grp in zip(cues, groups):
        if grp:
            start = max(0.0, grp[0][1] - audio_pre_roll)
            end = grp[-1][2]
            if end <= start:
                end = start + cue.duration
            # keep the timeline monotone no matter what whisper reported
            start = max(start, prev_end)
            if end < start:
                end = start + 0.05
            new = TimedCue(cue.text, start, end, grp)
            prev_end = new.end
            out.append(new)
        else:
            start = max(cue.start, prev_end)
            end = max(cue.end, start)
            c = TimedCue(cue.text, start, end, cue.words)
            prev_end = c.end
            out.append(c)
    return out


def cues_from_words(
    sentences: list[str],
    words: list[tuple[str, float, float]],
    audio_span: float,
    *,
    audio_pre_roll: float = DEFAULT_AUDIO_PRE_ROLL,
) -> list[TimedCue] | None:
    """Build cues for providers that returned none (OpenAI/ElevenLabs path).

    The old fallback guessed times proportionally by word count; these are
    MEASURED from the audio. Returns None when the match is too poor.
    """
    groups = map_words_to_sentences(words, sentences)
    if groups is None:
        return None
    out: list[TimedCue] = []
    prev_end = 0.0
    for sent, grp in zip(sentences, groups):
        if grp:
            start = max(0.0, grp[0][1] - audio_pre_roll)
            end = grp[-1][2]
            start = max(start, prev_end)
            if end <= start:
                end = start + 0.5
            out.append(TimedCue(sent, start, end, grp))
            prev_end = out[-1].end
        else:
            # Unmatched sentence: give it the space up to the next match.
            out.append(TimedCue(sent, prev_end, prev_end + 0.5))
            prev_end = out[-1].end
    if out:
        out[-1].end = max(out[-1].end, min(audio_span, out[-1].end + 0.2))
    return out


def align_narration(
    mp3: Path,
    sentences: list[str],
    provider_cues: list[TimedCue],
    audio_span: float,
    workdir: Path,
    code: str = "en",
    *,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
    enabled: bool = True,
) -> tuple[list[TimedCue], bool]:
    """Align one language's narration to its own audio. Returns (cues, aligned).

    Cached: the whisper words are stored in ``<code>.whisper.json`` keyed by
    the mp3's content hash + model + device, so a re-render never re-runs the
    (slow) transcription pass. When alignment is unavailable — no
    faster-whisper, transcription failed, match too poor — the provider's cues
    are returned untouched and ``aligned`` is False.
    """
    if not enabled or not mp3.exists() or not sentences:
        return provider_cues, False

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    words_path = workdir / f"{code}.whisper.json"
    marker = workdir / f".align_{code}.marker.json"
    sig = json.dumps(
        {
            "audio": _hash_file(mp3),
            "model": model_size,
            "device": device or "auto",
            "lang": language or "",
            "align_model": _align_model_name(language or "en") or "",
        },
        sort_keys=True,
    )

    words: list[tuple[str, float, float]] | None = None
    try:
        if marker.exists() and words_path.exists() \
                and json.loads(marker.read_text(encoding="utf-8")).get("sig") == sig:
            raw = json.loads(words_path.read_text(encoding="utf-8"))
            raw_words = raw.get("words", [])
            parsed_words: list[tuple[str, float, float]] = []
            for item in raw_words:
                if isinstance(item, dict):
                    parsed_words.append((str(item.get("word", "")), float(item.get("start", 0.0)), float(item.get("end", 0.0))))
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    parsed_words.append((str(item[0]), float(item[1]), float(item[2])))
            words = parsed_words or None
    except Exception:
        words = None

    if words is None:
        print(f"  * [{code}] whisperx-aligning the narration audio "
              f"({audio_span / 60:.1f} min; model={model_size}) ...", flush=True)
        words = narration_words(mp3, model_size=model_size, device=device,
                                language=language)
        if words is None:
            print(f"  ! [{code}] narration alignment unavailable "
                  "(whisperx/whisper unavailable or transcription failed) — keeping the "
                  "TTS provider's own timing.", flush=True)
            return provider_cues, False
        try:
            words_path.write_text(
                json.dumps(
                    {
                        "engine": "whisperx",
                        "words": [
                            {"word": w, "start": round(s, 3), "end": round(e, 3)}
                            for w, s, e in words
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            marker.write_text(json.dumps({"sig": sig}), encoding="utf-8")
        except OSError:
            pass

    if not provider_cues or not any(c.end > c.start for c in provider_cues):
        # Provider gave nothing measurable (openai/elevenlabs): the whisper
        # words ARE the timing now, replacing the old proportional guess.
        cues = cues_from_words(sentences, words, audio_span)
        if cues:
            print(f"  * [{code}] built {len(cues)} cues from whisper word "
                  "timestamps.", flush=True)
            return cues, True
        return provider_cues, False

    refined = refine_cues(provider_cues, sentences, words)
    if refined is provider_cues:
        return provider_cues, False
    print(f"  * [{code}] cue times refined from "
          f"{len(words)} measured words.", flush=True)
    return refined, True


# ---------------------------------------------------------------------------
# Dead-air trimming — zero silence at the head/tail of every sentence
# ---------------------------------------------------------------------------
_TRIM_SR = 48000


def _decode_pcm(path: Path, sr: int = _TRIM_SR):
    """Decode any audio file to mono float32 PCM with ffmpeg."""
    import subprocess

    import numpy as np

    from .util import which_ffmpeg

    proc = subprocess.run(
        [which_ffmpeg(), "-v", "error", "-i", str(path), "-ac", "1",
         "-ar", str(sr), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _zero_crossing(x, idx: int, direction: int, max_search: int) -> int:
    """Nearest zero crossing from ``idx`` moving ``direction`` (-1 = earlier,
    +1 = later) within ``max_search`` samples. Searching OUTWARD from the
    speech means a snap can only ever add a few samples of the waveform's
    own lead-in/tail, never cut into a word -- and cutting on a zero
    crossing is what prevents the click a mid-wave cut produces."""
    n = len(x)
    idx = min(max(idx, 0), n - 1) if n else 0
    for k in range(max_search):
        j = idx + direction * k
        if j <= 0 or j >= n - 1:
            return min(max(j, 0), n)
        if x[j] == 0.0 or (x[j] > 0) != (x[j + 1] > 0):
            return j + (1 if direction > 0 else 0)
    return idx


def speech_bounds(cue) -> tuple[float, float]:
    """Exact spoken span of a cue: first word start -> last word end (from the
    WhisperX word JSON); the cue's own start/end when it has no words."""
    words = [w for w in (getattr(cue, "words", None) or [])
             if len(w) >= 3 and float(w[2]) > float(w[1])]
    if words:
        return float(words[0][1]), float(words[-1][2])
    return float(cue.start), float(cue.end)


def trim_dead_air(
    mp3: Path,
    cues: list,
    out_mp3: Path,
    *,
    gap_ms: float = 0.0,
    fade_ms: float = 3.0,
    zc_search_ms: float = 8.0,
) -> list | None:
    """Cut every sentence to its exact WhisperX speech span and butt them
    together, so there is no dead air at the start or end of any line.

    Each cut is snapped outward to the nearest waveform zero crossing and
    gets a ``fade_ms`` micro-fade, so the joins never click. ``gap_ms`` of
    silence may be placed between sentences (default 0 = none).

    Returns the new cues (times and word times shifted onto the trimmed
    track, contiguous from 0) or ``None`` when trimming is impossible
    (numpy/ffmpeg missing, decode failed) -- the caller keeps the original.
    """
    try:
        import subprocess

        import numpy as np

        from .util import which_ffmpeg

        x = _decode_pcm(mp3)
    except Exception as exc:
        print(f"  ! dead-air trim skipped ({type(exc).__name__}: {exc})",
              flush=True)
        return None
    if not len(x) or not cues:
        return None
    sr = _TRIM_SR
    zc = max(int(sr * zc_search_ms / 1000.0), 1)
    fade = max(int(sr * fade_ms / 1000.0), 0)
    gap = np.zeros(int(sr * max(gap_ms, 0.0) / 1000.0), dtype=np.float32)

    pieces = []
    new_cues = []
    t = 0.0
    prev_end = 0
    removed = 0.0
    for i, c in enumerate(cues):
        s0, e0 = speech_bounds(c)
        a = _zero_crossing(x, int(s0 * sr), -1, zc)
        b = _zero_crossing(x, int(round(e0 * sr)), +1, zc)
        a = max(a, prev_end)              # never re-use audio (overlap)
        if b <= a:
            b = min(a + int(0.05 * sr), len(x))
        seg = x[a:b].copy()
        if fade and len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        start = t
        shift = start - a / sr
        dur = len(seg) / sr
        words = None
        if getattr(c, "words", None):
            words = [(w[0], round(max(float(w[1]) + shift, start), 3),
                      round(min(float(w[2]) + shift, start + dur), 3))
                     for w in c.words]
        new_cues.append(TimedCue(c.text, round(start, 3),
                                 round(start + dur, 3), words))
        pieces.append(seg)
        t += dur
        if i < len(cues) - 1 and len(gap):
            pieces.append(gap)
            t += len(gap) / sr
        removed += (b - a) / sr
        prev_end = b
    removed = len(x) / sr - removed

    y = np.concatenate(pieces)
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    out_mp3 = Path(out_mp3)
    tmp = out_mp3.with_suffix(".trim_tmp" + out_mp3.suffix)
    try:
        subprocess.run(
            [which_ffmpeg(), "-y", "-v", "error", "-f", "s16le", "-ar",
             str(sr), "-ac", "1", "-i", "-", "-c:a", "libmp3lame", "-b:a",
             "192k", str(tmp)],
            input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        )
        tmp.replace(out_mp3)
    except Exception as exc:
        print(f"  ! dead-air trim encode failed ({exc})", flush=True)
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    print(f"  * dead-air trim: {len(new_cues)} sentences cut to their exact "
          f"spoken span on zero crossings ({removed:.1f}s of silence removed)",
          flush=True)
    return new_cues


# ---------------------------------------------------------------------------
# Per-sentence silence strip (ffmpeg silenceremove) -- BEFORE alignment
# ---------------------------------------------------------------------------
SILENCE_THRESHOLD_DB = -50.0


def _silence_filter(threshold_db: float, head: bool, tail: bool) -> str:
    one = (f"silenceremove=start_periods=1:start_duration=0:"
           f"start_threshold={threshold_db:g}dB")
    parts = []
    if head:
        parts.append(one)
    if tail:
        parts += ["areverse", one, "areverse"]
    return ",".join(parts)


def strip_silence(
    src: Path,
    dst: Path | None = None,
    *,
    threshold_db: float = SILENCE_THRESHOLD_DB,
) -> tuple[float, float] | None:
    """Trim the silence off the head and tail of one TTS clip.

    Runs exactly the ffmpeg filter
    ``silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB,
    areverse,silenceremove=...,areverse`` -- in two passes so the amount cut
    from the HEAD is known (word timings must shift by it).

    Writes ``dst`` (default: overwrite ``src``) and returns
    ``(head_removed_seconds, new_duration)``; ``None`` when ffmpeg is missing
    or the clip could not be decoded (the caller keeps the original).
    """
    import subprocess

    from .util import probe_duration, which_ffmpeg

    src = Path(src)
    dst = Path(dst) if dst else src
    try:
        ff = which_ffmpeg()
        d0 = probe_duration(src)
        if d0 <= 0:
            return None
        head_tmp = dst.with_name(dst.stem + ".head.wav")
        out_tmp = dst.with_name(dst.stem + ".trim" + dst.suffix)
        subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(src), "-af",
             _silence_filter(threshold_db, True, False), str(head_tmp)],
            check=True, capture_output=True)
        d1 = probe_duration(head_tmp)
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"] \
            if dst.suffix.lower() == ".mp3" else []
        subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(head_tmp), "-af",
             _silence_filter(threshold_db, False, True), *codec, str(out_tmp)],
            check=True, capture_output=True)
        d2 = probe_duration(out_tmp)
        try:
            head_tmp.unlink()
        except OSError:
            pass
        if d2 <= 0.05:
            # the whole clip read as silence (threshold too high): keep it
            try:
                out_tmp.unlink()
            except OSError:
                pass
            return None
        out_tmp.replace(dst)
        return max(d0 - d1, 0.0), d2
    except Exception:
        return None
