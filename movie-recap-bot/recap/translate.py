"""Translate the English recap script to Simplified Chinese (简体中文).

Native, idiomatic Mandarin. Each line maps to the English line index so the
subtitle timelines stay in sync across both languages.

Like the script, this either:
  * uses an LLM (with a strong Chinese-copywriting prompt), or
  * loads a pre-written translation file that is line-aligned with the EN.
"""
from __future__ import annotations

from pathlib import Path

from . import llm
from .script import normalize

SYSTEM_ZH = (
    "你是一名专业的电影解说视频中文配音撰稿人。负责把英文电影剧情解说翻译成"
    "地道、自然、口语化的简体中文，用于视频字幕和配音。"
)

PROMPT_ZH = """请把下面这份英文电影剧情解说（每行一个句子）翻译成简体中文。

要求：
- 保持“每个英文句子对应一个中文句子”，行数与原文一一对应，不要合并或拆分。
- 用中文讲电影的、「电影解说」式的口语化叙事风格，自然流畅，不要逐字直译。
- 使用现在时，贴合原文的叙事节奏。
- 人物姓名：保留英文原名（或用中文译名，全文统一）。
- 不要加标题、注释或任何额外说明，只输出译文本身。

=== 英文原文（每行一个句子） ===

{english}

=== 译文开始 ===
"""


def generate_online(english: str, cfg_llm: dict) -> str:
    user = PROMPT_ZH.format(english=english)
    return llm.complete(
        cfg_llm.get("provider", ""),
        cfg_llm.get("model", ""),
        SYSTEM_ZH,
        user,
        base_url=cfg_llm.get("base_url"),
    )


def load_translation(path: Path | str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Translation file not found: {p}")
    return normalize(p.read_text(encoding="utf-8").splitlines())


def write_translation_file(text: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text.strip() + "\n", encoding="utf-8")
    return dest
