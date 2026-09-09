"""Language catalog for the recap pipeline.

The recap engine is multi-language. Every language is either:

* **authored natively** — the recap is WRITTEN in that language from a
  dialogue source in the same language (a subtitle file you provide, or
  Whisper on the movie's own audio for ``en``), or
* **translated** — a line-aligned translation of an authored master recap
  (this is how ``zh`` has always worked: it needs no dialogue source of its
  own because each Chinese line is paired 1:1 with its English sentence and
  reuses that sentence's film window).

Rules for a language's film windows, voices and fonts live here so the
pipeline, prompts and studio all agree on one catalog.
"""
from __future__ import annotations

# Language codes the engine knows how to render.
SUPPORTED: tuple[str, ...] = ("en", "zh", "ar", "es")

# Human names used in LLM prompts ("write the beats in Arabic") and logs.
NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
    "ar": "Arabic",
    "es": "Spanish",
}

# Default edge-tts narrator per language. Warm deep male voices where
# available, matching the English narrator's register (configurable under
# narration.lang_voice).
VOICE_DEFAULTS: dict[str, str] = {
    "en": "en-US-ChristopherNeural",
    "zh": "zh-CN-YunjianNeural",
    "ar": "ar-SA-HamedNeural",      # Modern Standard Arabic (Saudi), male
    "es": "es-MX-JorgeNeural",      # Latin American Spanish (Mexico), male
}

# Default burned-subtitle font per language. Arabic needs a shaped Arabic
# typeface (libass renders it through the font's Arabic tables); the CJK/Latin
# default font falls back to subtitles.font for every other language.
# "Arial" exists on stock Windows and covers Arabic + Latin; swap to a
# Noto/Arial-styled Arabic font if you install one.
SUBTITLE_FONTS: dict[str, str] = {
    "ar": "Arial",
}


def name(code: str) -> str:
    """Human language name for prompts/logs (falls back to the code)."""
    return NAMES.get(code, code)


def voice_default(code: str) -> str:
    return VOICE_DEFAULTS.get(code, VOICE_DEFAULTS["en"])


def font_for(code: str, subtitles_cfg: dict) -> str:
    """Burned-subtitle font for a language code (falls back to the shared one)."""
    if code in SUBTITLE_FONTS:
        return SUBTITLE_FONTS[code]
    return str(subtitles_cfg.get("font") or "Noto Serif CJK SC")


def validate(codes: list[str]) -> list[str]:
    """Return unknown codes among ``codes`` (empty when all are supported)."""
    return [c for c in codes if c not in SUPPORTED]
