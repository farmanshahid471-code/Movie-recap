"""Step A (pass 2) — summarize the action of each transcript chunk.

Each 5-minute block of raw dialogue is distilled by the LLM into *what
actually happens* in that block (action beats, present tense, third person).
The per-chunk summaries are then concatenated in order and handed to the final
script writer, so a full 2-hour film never has to fit one context window.

Slow-PC notes
-------------
* Progress is printed per chunk (index, chars, elapsed) so the run never looks
  dead during the long LLM pass, and each summary is appended to
  ``out_partial`` as it completes so partial progress survives a crash.
* ``cfg_llm["summary_model"]`` may point at a smaller/faster model than the
  main model (e.g. ``qwen2.5:3b``) — on CPU this is several times faster and
  chunk summaries are simple enough for a small model.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
- Aim for roughly {budget} characters of output — use FEWER lines when the block is thin.
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
    out_partial: str | Path | None = None,
) -> list[str]:
    """Summarize every transcript chunk. Returns summaries aligned to chunks.

    ``out_partial`` — if given, each finished summary is appended to that file
    immediately (so a crash never loses all progress and the file shows live
    progress while the pass runs).
    """
    if not chunks:
        return []

    model = cfg_llm.get("summary_model") or cfg_llm.get("model") or ""
    n = len(chunks)
    started = time.time()

    def one(chunk: dict) -> str:
        text = chunk.get("text", "") or ""
        budget = max(500, min(2200, int(len(text) * 0.22)))
        user = PROMPT_SUMMARY.format(transcript=text, budget=budget)
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            model,
            SYSTEM_SUMMARY,
            user,
            base_url=cfg_llm.get("base_url"),
        )
        return (raw or "").strip()

    results: list[str] = []
    if parallel and n > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for idx, chunk in enumerate(chunks):
                futures.append(pool.submit(one, chunk))
            for idx, fut in enumerate(futures):
                print(f"    ... chunk {idx + 1}/{n} ...", flush=True)
                results.append(fut.result())
    else:
        for idx, chunk in enumerate(chunks):
            el = time.time() - started
            print(
                f"    ... chunk {idx + 1}/{n} via {model} "
                f"({len(chunk.get('text', ''))} chars, {el / 60:.1f} min elapsed) ...",
                flush=True,
            )
            s = one(chunk)
            results.append(s)
            if out_partial:
                try:
                    p = Path(out_partial)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(f"--- Chunk {idx} ---\n{s}\n\n")
                except OSError:
                    pass
            print(f"      -> chunk {idx + 1} done ({(time.time() - started) / 60:.1f} min total)",
                  flush=True)
    return results


def merge_summaries(summaries: list[str]) -> str:
    """Join per-chunk summaries into one chronological story summary."""
    blocks = []
    for i, s in enumerate(summaries):
        s = (s or "").strip()
        if not s:
            continue
        blocks.append(f"--- Chunk {i} ---\n{s}")
    return "\n\n".join(blocks)
