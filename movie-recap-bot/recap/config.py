"""Configuration loading: YAML config + environment secrets, with sane defaults."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from . import storage

BASE_DIR = Path(__file__).resolve().parent.parent

_DEFAULTS: dict[str, Any] = {
    "project": {"name": "recap-project", "output_dir": "output", "cache_dir": ""},
    # Languages whose clips to render. "en" is authored from the movie's own
    # audio/dialogue; "zh"/"ar"/"es" are authored natively when a subtitle in
    # that language is provided (language.sources or <movie>.<code>.srt next to
    # the film), otherwise they fall back to a line-aligned translation of the
    # English recap.
    "language": {"target_languages": ["en"], "zh_variant": "zh-CN",
                 "sources": {}},
    "narration": {
        "words_target": 2000,      # ~13-14 min at ~150 wpm (full-length recap)
        "words_min": 600,
        "words_max": 4200,
        "words_per_minute": 150,   # speech rate used for all length maths
        # Default edge-tts narrators per language (all warm, deep male).
        "lang_voice": {"en": "en-US-ChristopherNeural",
                       "zh": "zh-CN-YunjianNeural",
                       "ar": "ar-SA-HamedNeural",    # Modern Standard Arabic
                       "es": "es-MX-JorgeNeural"},   # Latin American Spanish
        "rate": "+0%",
        # edge-tts pitch shift in Hz ("-6Hz" deeper, "+6Hz" brighter; "-0Hz" = off)
        "pitch": "-0Hz",
        "tts_provider": "edge",
        # Re-run the GENERATED narration audio through faster-whisper and lock
        # every cue (and word) to what is actually spoken. Works with ANY TTS
        # provider — for openai/elevenlabs this replaces guessed cue times
        # with measured ones. Cached per audio content; set false to skip.
        "whisper_align": True,
        # Optional smaller/faster model just for the narration alignment
        # (defaults to dialogue.whisper_model, i.e. "small").
        "whisper_align_model": None,
        # Close every video with the channel outro ("If you enjoyed the
        # video, don't forget to leave a like...") like real recap channels.
        "sign_off": True,
        # VISUAL MATCH: size each section's narration to the film time it
        # covers (instead of by beat density), cap dense sections at what
        # their footage can show at 1x, pace sentence anchors so every
        # sentence's window is at least as long as the sentence, and
        # hard-trim sections the writer over-delivers (the budget is a
        # ceiling, not a suggestion). This is what lets the whole recap
        # play at normal speed -- no slow motion, no frozen frames (those
        # remain only as a safety net).
        "visual_match": True,
    },
    # Step D — chronological timeline (replaces semantic vector matching).
    # Beats advance monotonically through the film and every beat's visual is
    # locked to its narration cue, so video length == audio length exactly.
    "timeline": {
        "micro_cut_seconds": 3.0,  # aim for a new shot roughly every 3s
        "max_cuts_per_beat": 3,    # a long sentence becomes up to 3 micro-shots
        "min_cut_seconds": 1.2,    # never flash a shot shorter than this
        "pre_roll": 0.4,           # start each shot slightly before its moment
        # Place the micro-cut points inside a sentence ON MEASURED WORD
        # boundaries (clause breaks: after commas, before and/but/while),
        # so the picture switches exactly when the narrator changes subject.
        "cut_on_words": True,
        # Snap every cut's film position onto the film's REAL shot changes
        # (one cached PySceneDetect pass per movie; `pip install
        # scenedetect[opencv]`), so each visual begins on an actual camera
        # cut the way a human edit does. No scenedetect -> un-snapped.
        "snap_to_scenes": True,
        "snap_tolerance": 0.8,   # max seconds to move a cut onto a boundary
        # How far the visuals may run AHEAD of the moment being narrated
        # (safety valve; group pacing keeps the typical lead near zero).
        "max_lead_seconds": 3.0,
        # MOTION GUARANTEE: in dialogue-dense sections the narration can be
        # longer than the footage behind it. Instead of freezing the picture
        # or running ahead, the footage plays in slow motion down to this
        # speed (0.35x still looks smooth at 30fps). 1.0 disables slow-mo.
        "min_speed": 0.35,
        # A cut must show at least this much NEW film, otherwise it continues
        # the current footage seamlessly (micro-jumps read as stutters).
        "min_new_footage": 0.8,
    },
    # Whisper ASR tuning (auto-recap from the movie's own audio).
    "dialogue": {
        "whisper_model": "small",
        "whisper_device": "auto",
        "whisper_language": None,
        "word_timestamps": True,   # per-word times on every cue (dialogue sidecar)
        "max_chars": None,         # cap for transcript text sent to an LLM
        "srt_path": None,          # optional pre-existing subtitle instead of ASR
    },
    # Step A — contextual chunking of the long transcript so LLM context
    # windows never overflow. Windows of `window_seconds` sliding by
    # `window_seconds - overlap_seconds`, each carrying 30s of context.
    "chunking": {
        "window_seconds": 180.0,   # 3-minute blocks (maximum-precision mode)
        "overlap_seconds": 30.0,   # overlap between adjacent blocks
        "parallel": False,         # Ollama is single-user; keep serial by default
        "model": None,             # optional smaller/faster model for the
                                   # chunk-summary pass, e.g. "qwen2.5:3b"
    },
    # Step A (pass 1.5) — optional VISUAL pass. DeepSeek's chat API is
    # text-only, so narration is blind to silent set-pieces. When a key for a
    # vision-capable provider exists (default: GEMINI_API_KEY, free tier), the
    # film's frames are captioned and the notes merged into each chunk's beat
    # list. Skipped gracefully (text-only) when no key is present.
    "vision": {
        "enabled": True,
        "provider": "gemini",      # gemini (free, multimodal) | openai | groq
        "model": "gemini-3.6-flash",  # Gemini Flash models are multimodal
        "base_url": None,          # None = provider default (Gemini OpenAI-compat)
        "cadence_seconds": 20.0,   # sample ~every 20s of film
        "scene_threshold": 0.35,   # also caption real shot changes above this
        "max_frames": 400,         # hard cap per movie (API-quota friendly)
        "width": 512,              # JPEG width sent to the vision model
        "frames_per_request": 4,   # frames per API call (free-tier economy)
    },
    # LEGACY semantic vector matcher (retired from beat selection — the
    # chronological timeline in recap/timeline.py maps narration lines to film
    # windows now). Only semantic.clip.mode is still read by the pipeline.
    "semantic": {
        "enabled": True,
        "embedding_model": "all-MiniLM-L6-v2",  # 384-dim, runs locally (legacy)
        "store": "auto",           # auto | local | supabase (legacy)
        "top_k": 3,                # candidates considered per recap line (legacy)
        "min_score": 0.10,         # below this -> even-beat fallback (legacy)
        "pre_roll": 0.5,           # seconds of footage before the matched cue (legacy)
        "clip_pad": 0.15,          # extra footage after the narration of a line (legacy)
        "min_clip": 0.8,           # never cut a beat shorter than this (legacy)
        "max_clip": 10.0,          # nor longer than this (legacy)
        "clip": {"mode": "reencode"},  # reencode = frame-exact (required for A/V lock)
    },
    "subtitles": {
        "font": "Noto Serif CJK SC",
        # Per-language burned-subtitle fonts; missing codes fall back to font.
        # Arabic needs a shaped Arabic typeface (default "Arial" ships with
        # Windows); others reuse the CJK font's Latin glyphs.
        "lang_font": {"ar": "Arial"},
        "fontsize": 56,
        "margin_v": 96,
        "margin_x": 40,
        "outline": 3,
        "shadow": 1,
        "max_combo_duration": 6.0,
        "line_width_units": 30,   # ~30 CJK (or ~60 Latin) glyphs per line
        "display_lang": "en",
    },
    "video": {
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "concat_mode": "fit",
        # "scenes" cuts a recap-style montage of beats from the movie;
        # "continuous" plays it straight through and loops to cover narration.
        "montage": "scenes",
        "scene_len": 6.0,          # seconds per beat (even-beat fallback)
        "scene_min_len": 2.0,      # drop detected scenes shorter than this
        "scene_max_len": 20.0,     # trim detected scenes longer than this
        "scene_threshold": 27.0,   # PySceneDetect ContentDetector threshold
        "codec": "libx264",
        "audio_codec": "aac",
        "bgm": "",
        "bgm_volume": 0.12,
    },
    "llm": {"provider": "deepseek", "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # only set from file if not already in environment
        os.environ.setdefault(key, val)


def _coerce_typed(value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(default, list):
        return [x.strip() for x in value.split(",") if x.strip()]
    return value


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (or a given path) merged over defaults + env overrides."""
    _load_dotenv(BASE_DIR / ".env")

    yaml_path = Path(path) if path else BASE_DIR / "config.yaml"
    raw: dict = {}
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}

    cfg = _deep_merge(_DEFAULTS, raw)

    # Runtime env overrides for the most-toggled fields.
    cfg["llm"]["provider"] = os.environ.get(
        "LLM_PROVIDER", cfg["llm"].get("provider", "")
    )
    cfg["llm"]["model"] = os.environ.get(
        "MODEL_NAME", cfg["llm"].get("model", "deepseek-chat")
    )
    # Base URL is per-provider. Previously OLLAMA_BASE_URL was applied to every
    # provider, so a stale env var silently pointed DeepSeek at localhost:11434.
    _prov = (cfg["llm"].get("provider") or "").strip().lower()
    _PROVIDER_BASE_ENV = {
        "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "deepseek": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "groq": ("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "gemini": ("GEMINI_BASE_URL",
                   "https://generativelanguage.googleapis.com/v1beta/openai/"),
    }
    _env_key, _fallback = _PROVIDER_BASE_ENV.get(_prov, ("", ""))
    if _env_key and os.environ.get(_env_key):
        cfg["llm"]["base_url"] = os.environ[_env_key]
    elif not cfg["llm"].get("base_url") and _fallback:
        cfg["llm"]["base_url"] = _fallback
    if _prov == "ollama" and cfg["llm"].get("base_url"):
        os.environ.setdefault("OLLAMA_BASE_URL", cfg["llm"]["base_url"])
    cfg["narration"]["tts_provider"] = os.environ.get(
        "TTS_PROVIDER", cfg["narration"].get("tts_provider", "edge")
    )
    # Narration sync/style toggles (see recap/align.py + recap/script.py).
    if "RECAP_WHISPER_ALIGN" in os.environ:
        cfg["narration"]["whisper_align"] = os.environ[
            "RECAP_WHISPER_ALIGN"
        ].strip().lower() not in ("0", "false", "no", "off")
    if "RECAP_SIGN_OFF" in os.environ:
        cfg["narration"]["sign_off"] = os.environ[
            "RECAP_SIGN_OFF"
        ].strip().lower() not in ("0", "false", "no", "off")
    # Vision pass toggles (see recap/vision.py). Keys come from the provider's
    # env var (gemini -> GEMINI_API_KEY), which _load_dotenv already imported.
    if "VISION_ENABLED" in os.environ:
        cfg["vision"]["enabled"] = os.environ["VISION_ENABLED"].strip().lower() not in (
            "0", "false", "no", "off"
        )
    if os.environ.get("VISION_PROVIDER"):
        cfg["vision"]["provider"] = os.environ["VISION_PROVIDER"].strip().lower()
    if os.environ.get("VISION_MODEL"):
        cfg["vision"]["model"] = os.environ["VISION_MODEL"].strip()
    if os.environ.get("VISION_BASE_URL"):
        cfg["vision"]["base_url"] = os.environ["VISION_BASE_URL"].strip()

    # Language tags: zh -> configured zh_variant
    langs = []
    for lang in cfg["language"]["target_languages"]:
        tag = lang
        if lang.startswith("zh"):
            tag = cfg["language"].get("zh_variant", "zh-CN")
        langs.append({"code": lang, "tag": tag})
    cfg["language"]["_resolved"] = langs

    # Output root: env OUTPUT_DIR > config project.output_dir > default.
    # Relative paths resolve against the bot folder; absolute paths (e.g.
    # "D:\\recap\\output") are honored verbatim so nothing has to live on C:.
    if os.environ.get("OUTPUT_DIR"):
        cfg["project"]["output_dir"] = os.environ["OUTPUT_DIR"]
    out = Path(cfg["project"]["output_dir"])
    out_abs = out if storage.is_abs(out) else BASE_DIR / out
    cfg["project"]["_out"] = out_abs.resolve()

    # Cache root: env CACHE_DIR > config project.cache_dir > sibling "cache"
    # folder of the output root (so all data follows output_dir's drive).
    if os.environ.get("CACHE_DIR"):
        cfg["project"]["cache_dir"] = os.environ["CACHE_DIR"]
    raw_cache = (cfg["project"].get("cache_dir") or "").strip()
    if raw_cache:
        cache = Path(raw_cache)
        if not storage.is_abs(cache):
            cache = BASE_DIR / cache
    else:
        cache = out_abs.resolve().parent / "cache"
    cfg["project"]["_cache"] = cache.resolve()
    storage.bootstrap(cfg["project"]["_cache"])

    cfg["project"]["name"] = cfg["project"].get("name", "recap-project")
    return cfg


def out_dir(cfg: dict) -> Path:
    return Path(cfg["project"]["_out"])


def cache_dir(cfg: dict) -> Path:
    return Path(cfg["project"]["_cache"])


def work_dir(cfg: dict) -> Path:
    d = out_dir(cfg) / "_work"
    d.mkdir(parents=True, exist_ok=True)
    return d
