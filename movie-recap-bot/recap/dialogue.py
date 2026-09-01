"""Dialogue / transcript extraction for auto-recap.

Two sources, tried in order:
  1. An existing subtitle file (.srt/.ass/.vtt) next to the movie — no heavy deps.
  2. Speech-to-text (Whisper) on the movie's audio — optional; requires
     faster-whisper / openai-whisper / whisperx.

Returns a timestamped transcript in a compact text form the LLM can read.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from .util import probe_duration, run, which_ffmpeg


class DialogueError(RuntimeError):
    pass


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _read_lines_from_text(text: str) -> list[dict]:
    out: list[dict] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        out.append({"text": raw})
    return out


def from_srt_path(path: Path) -> list[dict]:
    """Parse an SRT/ASS/VTT subtitle file into timed cues."""
    try:
        import pysubs2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise DialogueError(
            "pysubs2 not installed. `pip install pysubs2` to parse subtitle files."
        ) from exc

    try:
        sub = pysubs2.load(str(path), encoding="utf-8")
    except Exception:
        sub = pysubs2.load(str(path), encoding="latin-1")

    cues: list[dict] = []
    for ev in sub:
        text = ev.plaintext.replace("\n", " ").strip()
        if not text:
            continue
        cues.append({"text": text, "start": ev.start / 1000.0, "end": ev.end / 1000.0})
    return cues


def find_subtitle_near(video: Path, extra: str | None = None) -> Path | None:
    """Look for an SRT/ASS/VTT matching the movie name (or an explicit path)."""
    if extra:
        p = Path(extra)
        if p.exists():
            return p
    candidates = [
        video.with_suffix(x)
        for x in ("", ".srt", ".ass", ".vtt", ".sub", ".txt")
    ]
    for c in candidates:
        if c.exists() and c != video:
            return c
    return None


def _from_whisper(video: Path, model_size: str, device: str, language: str | None) -> list[dict]:
    """Transcribe with Whisper (openai-whisper, faster-whisper, or whisperx)."""
    audio = _extract_audio(video)
    # Try faster-whisper first (fast, CPU-friendly), then openai-whisper, then whisperx.
    try:
        return _faster_whisper(audio, model_size, device, language)
    except ImportError:
        pass
    try:
        return _openai_whisper(audio, model_size, language)
    except ImportError:
        pass
    try:
        return _whisperx(audio, model_size, device, language)
    except ImportError:
        raise DialogueError(
            "No Whisper implementation found for audio transcription. Install one of:\n"
            "  pip install faster-whisper      # recommended (CPU-friendly)\n"
            "  pip install openai-whisper\n"
            "  pip install whisperx            # word-level timestamps + diarization\n"
            "OR provide an existing .srt subtitle file next to the movie."
        )


def _extract_audio(video: Path, tmp: Path | None = None) -> Path:
    audio = tmp or (video.parent / f"{video.stem}_audio.wav")
    run(
        [
            which_ffmpeg(), "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    return audio


def _faster_whisper(audio: Path, model_size: str, device: str, language: str | None) -> list[dict]:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(model_size, device=device, compute_type="int8")
    segments, _info = model.transcribe(str(audio), language=language, beam_size=1)
    cues: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            cues.append({"text": text, "start": float(seg.start), "end": float(seg.end)})
    return cues


def _openai_whisper(audio: Path, model_size: str, language: str | None) -> list[dict]:
    import whisper  # type: ignore

    model = whisper.load_model(model_size)
    out = model.transcribe(str(audio), language=language)
    cues: list[dict] = []
    for s in out.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            cues.append({"text": text, "start": float(s["start"]), "end": float(s["end"])})
    return cues


def _whisperx(audio: Path, model_size: str, device: str, language: str | None) -> list[dict]:
    import whisperx  # type: ignore

    model = whisperx.load_model(model_size, device, compute_type="int8")
    result = model.transcribe(str(audio), language=language)
    cues: list[dict] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            cues.append({"text": text, "start": float(seg["start"]), "end": float(seg["end"])})
    return cues


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_dialogue(
    video: Path,
    srt_path: str | Path | None = None,
    *,
    whisper_model: str = "small",
    whisper_device: str = "cpu",
    whisper_language: str | None = None,
    whitelist: bool = True,
) -> list[dict]:
    """Return a list of timed cues from the film's dialogue.

    Prefers an existing subtitle; else transcribes the audio with Whisper.
    """
    video = Path(video)
    srt = find_subtitle_near(video, str(srt_path) if srt_path else None)
    if srt:
        cues = from_srt_path(srt)
        return _maybe_whitelist(cues, whitelist)
    # No subtitles -> transcribe.
    return _maybe_whitelist(
        _from_whisper(video, whisper_model, whisper_device, whisper_language),
        whitelist,
    )


def _maybe_whitelist(cues: list[dict], whitelist: bool) -> list[dict]:
    if not whitelist:
        return cues
    return [c for c in cues if c.get("text")]


def to_transcript_text(cues: list[dict], max_chars: int | None = None) -> str:
    """Compact, timestamped transcript text for the LLM prompt."""
    parts: list[str] = []
    for c in cues:
        t = c.get("text", "").strip()
        if not t:
            continue
        if "start" in c:
            parts.append(f"[{_fmt(c['start'])}] {t}")
        else:
            parts.append(t)
    text = "\n".join(parts)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


def transcript_markdown(cues: list[dict]) -> str:
    """Nicely formatted transcript for the review/export file."""
    lines = []
    for c in cues:
        t = c.get("text", "").strip()
        if not t:
            continue
        if "start" in c:
            lines.append(f"**{_fmt(c['start'])}**  {t}")
        else:
            lines.append(t)
    return "\n".join(lines)
