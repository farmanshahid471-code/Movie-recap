"""Movie-Recaps-style summary bot.

Turns an (owned) movie into a short, continuously-narrated recap video in the
*Movie Recaps* style — present-tense storytelling over a background montage,
with burned-in subtitles — in English and Simplified Chinese.

Pipeline:
    script   ->  recap script (EN)           (LLM or provided file)
    translate->  Simplified Chinese script   (LLM or provided file)
    narrate  ->  narration audio + timing    (TTS: edge / elevenlabs / openai)
    subtitles->  .srt + .ass subtitle files  (burned-in)
    assemble ->  final .mp4 per language     (ffmpeg)
"""

__version__ = "0.1.0"

# Storage policy runs as early as possible so that no library ever drops a
# model weight or scratch file into the OS user-profile / C: drive. When the
# user exported CACHE_DIR / OUTPUT_DIR (or keeps them in movie-recap-bot/.env)
# we honour them here; otherwise load_config() bootstraps from config.yaml.
try:  # pragma: no cover - best-effort, never break imports over storage
    from . import storage as _storage  # noqa: E402

    if _storage.env_cache_root() or _storage.env_output_root():
        _storage.bootstrap()
except Exception:  # pragma: no cover
    pass
