"""Step A (pass 2) — summarize the action of each transcript chunk.

Each 5-minute block of raw dialogue is distilled by the LLM into *what
actually happens* in that block (action beats, present tense, third person).
The per-chunk summaries are then concatenated in order and handed to the final
script writer, so a full 2-hour film never has to fit one context window.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import llm

SYSTEM_SUMMARY = (
    "You are a movie plot analyst. You read timestamped film dialogue and "
    "infer the ACTION that is happening on screen. You never quote dialogue "
    "and you never repeat raw lines — you say what the characters do."
)

PROMPT_SUMMARY = """Below is a TIMESTAMPED DIALOGUE BLOCK from a movie (what the characters say, with timecodes).
Read it and summarize the *action implied by the dialogue* — the story beats that are happening on screen.

Rules:
- Output a list of concise, factual beats, one per line, in chronological order.
- Present tense, third person. Never quote dialogue. No "the movie/the scene shows".
- Infer visual action from what is said ("He grabs her arm", not "she says she is scared").
- Keep each beat short (under ~25 words) and dense. Skip filler and small talk.
- A block has roughly {budget} characters of output; use fewer lines if the block is thin.
- If this block continues an earlier scene, pick up where it left off naturally.

=== TIMESTAMPED DIALOGUE BLOCK ===
{transcript}
=== END OF BLOCK ===
"""


def summarize_chunks(
    chunks: list[dict],
    cfg_llm: dict,
    *,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[str]:
    """Summarize every transcript chunk. Returns summaries aligned to chunks."""
    if not chunks:
        return []

    def one(chunk: dict) -> str:
        budget = max(800, min(4500, int(len(chunk.get("text", "")) * 0.35)))
        user = PROMPT_SUMMARY.format(
            transcript=chunk.get("text", ""), budget=budget
        )
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            cfg_llm.get("model", ""),
            SYSTEM_SUMMARY,
            user,
            base_url=cfg_llm.get("base_url"),
        )
        return (raw or "").strip()

    if parallel and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(one, chunks))
    return [one(c) for c in chunks]


def merge_summaries(summaries: list[str]) -> str:
    """Join per-chunk summaries into one chronological story summary."""
    blocks = []
    for i, s in enumerate(summaries):
        s = (s or "").strip()
        if not s:
            continue
        blocks.append(f"--- Chunk {i} ---\n{s}")
    return "\n\n".join(blocks)
