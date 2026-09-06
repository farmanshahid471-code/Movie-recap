"""Configuration loading: YAML config + environment secrets, with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

_DEFAULTS: dict[str, Any] = {
    "project": {"name": "recap-project", "output_dir": "output"},
    "language": {"target_languages": ["en"], "zh_variant": "zh-CN"},
    "narration": {
        "words_target": 2000,      # ~13-14 min at ~150 wpm (full-length recap)
        "words_min": 600,
        "words_max": 4200,
        "lang_voice": {"en": "en-US-ChristopherNeural", "zh": "zh-CN-YunxiNeural"},
        "rate": "+0%",
        "tts_provider": "edge",
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
        "window_seconds": 300.0,   # 5-minute logical blocks
        "overlap_seconds": 30.0,   # overlap between adjacent blocks
        "parallel": False,         # Ollama is single-user; keep serial by default
    },
    # Step D — semantic timestamp mapping (recap sentence -> movie moment).
    "semantic": {
        "enabled": True,           # auto-recap maps each line to a film moment
        "embedding_model": "all-MiniLM-L6-v2",  # 384-dim, runs locally
        "store": "auto",           # auto | local | supabase (auto: supabase when creds exist)
        "top_k": 3,                # candidates considered per recap line
        "min_score": 0.10,         # below this -> even-beat fallback for that line
        "pre_roll": 0.5,           # seconds of footage before the matched cue
        "clip_pad": 0.15,          # extra footage after the narration of a line
        "min_clip": 0.8,           # never cut a beat shorter than this
        "max_clip": 10.0,          # nor longer than this
        "clip": {"mode": "copy"},  # copy (fast, keyframe) | reencode (frame-exact)
    },
    "subtitles": {
        "font": "Noto Serif CJK SC",
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
    "llm": {"provider": "ollama", "model": "qwen2.5", "base_url": "http://localhost:11434/v1"},
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
    cfg["llm"]["model"] = os.environ.get("MODEL_NAME", cfg["llm"].get("model", "qwen2.5"))
    # Ollama base URL (OpenAI-compatible).
    cfg["llm"]["base_url"] = os.environ.get(
        "OLLAMA_BASE_URL", cfg["llm"].get("base_url", "http://localhost:11434/v1")
    )
    if cfg["llm"]["provider"] == "ollama":
        os.environ.setdefault("OLLAMA_BASE_URL", cfg["llm"]["base_url"])
    cfg["narration"]["tts_provider"] = os.environ.get(
        "TTS_PROVIDER", cfg["narration"].get("tts_provider", "edge")
    )

    # Language tags: zh -> configured zh_variant
    langs = []
    for lang in cfg["language"]["target_languages"]:
        tag = lang
        if lang.startswith("zh"):
            tag = cfg["language"].get("zh_variant", "zh-CN")
        langs.append({"code": lang, "tag": tag})
    cfg["language"]["_resolved"] = langs

    # Resolve output dir absolute
    out = Path(cfg["project"]["output_dir"])
    cfg["project"]["_out"] = (BASE_DIR / out).resolve()
    cfg["project"]["_base"] = BASE_DIR
    cfg["project"]["name"] = cfg["project"].get("name", "recap-project")
    return cfg


def out_dir(cfg: dict) -> Path:
    return Path(cfg["project"]["_out"])


def work_dir(cfg: dict) -> Path:
    d = out_dir(cfg) / "_work"
    d.mkdir(parents=True, exist_ok=True)
    return d
