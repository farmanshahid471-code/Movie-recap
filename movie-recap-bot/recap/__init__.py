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
