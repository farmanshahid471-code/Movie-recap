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

import hashlib
import re
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
Read it and write out the ACTION that is happening on screen — the full story-beat list of this block, in exact order.

Why this matters: your output is the ONLY source the final recap narration is written from.
Every beat you omit is a moment the recap can never show. Completeness first.

Rules:
- ONE LINE PER STORY BEAT. Cover EVERY distinct moment in order: each arrival, departure,
  decision, discovery, confrontation, reveal, reaction, plan, trick, and scene change.
  Never merge two different moments into one line; never drop a beat to keep it short.
- Name the characters who act (use the name the dialogue uses — "Buzz", "Jessie", "Lilypad").
  Keep proper nouns: places, devices, objects, and app names when they matter.
- Present tense, third person, VISIBLE action only ("Jessie hops onto Bullseye and rides off"),
  inferred from what is said — never quote dialogue verbatim.
- If characters talk about something that happened off screen, say what that was.
- Keep each line dense and short (under ~30 words). No "the movie", "the scene shows", "we see".
- Aim for roughly {budget} characters of output. Dense blocks may use the full budget; thin
  blocks should be short. Never pad with invented events.

=== TIMESTAMPED DIALOGUE BLOCK ===
{transcript}
=== END OF BLOCK ===
"""


def _summary_budget(text_chars: int) -> int:
    """Character budget for one chunk summary (maximum-detail mode).

    The old ~0.22x ratio compressed a dense 5-minute block so hard that whole
    scenes vanished before the script writer ever saw them. Default now keeps
    ~0.42x of the raw transcript as story beats — near-complete beat coverage.
    Tune with RECAP_SUMMARY_RATIO (e.g. 0.25 = lighter / cheaper) without
    touching code; the floor/ceiling keep degenerate inputs sane.
    """
    import os

    try:
        ratio = float(os.environ.get("RECAP_SUMMARY_RATIO", "0.42"))
    except ValueError:
        ratio = 0.42
    return max(350, min(5200, int(text_chars * ratio)))


def _summary_max_tokens(budget_chars: int) -> int:
    """Output cap for one chunk summary, derived from the requested size.

    The prompt asks for ``budget`` *characters* of dense beats. English runs
    ~4 chars/token, Chinese ~1; using ~0.9 tokens/char + padding caps a
    rambling summary at ~2x what the job needs without truncating a good one.
    """
    return max(512, min(4096, int(budget_chars * 0.9) + 256))


def _read_partial(path: Path) -> dict[int, str]:
    """Parse an out_partial file into {chunk_index: summary_text}.

    Format written by this module: a ``--- Chunk N ---`` marker line followed
    by the summary text. Returns the blocks found (so an interrupted run can
    resume exactly where it left off).
    """
    out: dict[int, str] = {}
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    cur: int | None = None
    buf: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^--- Chunk (\d+) ---$", line.strip())
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur = int(m.group(1))
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def _chunk_signature(chunks: list[dict]) -> str:
    """Cheap content signature of the chunk list.

    Two runs resume safely only when the chunks are byte-identical (same movie,
    same transcript cache, same window/overlap). Anything that changes the
    chunks — a different film, re-extracted transcript, window_seconds tweak —
    yields a different signature, so stale summaries are never reused.
    """
    h = hashlib.sha1()
    for c in chunks:
        t = (c.get("text", "") or "")
        h.update(str(len(t)).encode("utf-8", "replace"))
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


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

    # Resume: if an out_partial file already holds completed chunks (from an
    # interrupted run) AND the chunk signature matches (same movie/transcript/
    # chunking), re-use them and only summarize the rest. A signature mismatch
    # means the file is stale — wipe it and start fresh.
    done: dict[int, str] = {}
    skip = 0
    partial_path = Path(out_partial) if out_partial is not None else None
    if partial_path is not None:
        sig_path = Path(str(partial_path) + ".sig")
        sig = _chunk_signature(chunks)
        # Stamp the signature so a future run can validate a resume. Always
        # write it for the *new* chunks (this run or a future one).
        try:
            sig_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            fresh = partial_path.exists() and sig_path.exists() and \
                sig_path.read_text(encoding="utf-8").strip() == sig
        except OSError:
            fresh = False
        if fresh:
            done = _read_partial(partial_path)
            contiguous = 0
            while contiguous in done and contiguous < n:
                contiguous += 1
            if contiguous > 0:
                skip = contiguous
                print(f"  * Resuming: {skip}/{n} chunks already summarized in "
                      f"{out_partial} (delete it to force a full re-run).", flush=True)
        elif partial_path.exists():
            # stale partial from different chunks/movie — clear it
            try:
                partial_path.write_text("", encoding="utf-8")
            except OSError:
                pass
            try:
                sig_path.unlink(missing_ok=True)
            except OSError:
                pass

    def one(chunk: dict) -> str:
        text = chunk.get("text", "") or ""
        budget = _summary_budget(len(text))
        user = PROMPT_SUMMARY.format(transcript=text, budget=budget)
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            model,
            SYSTEM_SUMMARY,
            user,
            base_url=cfg_llm.get("base_url"),
            max_tokens=_summary_max_tokens(budget),
        )
        return (raw or "").strip()

    results: list[str] = [""] * n
    for idx, txt in done.items():
        if idx < n:
            results[idx] = txt

    # Stamp the signature for this chunk set (future runs resume only when the
    # signature matches, so stale summaries are never reused).
    if partial_path is not None:
        try:
            sig_path.write_text(sig, encoding="utf-8")
        except OSError:
            pass

    if skip >= n:
        return results

    remaining = [(idx, chunk) for idx, chunk in enumerate(chunks) if idx >= skip]
    if parallel and len(remaining) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for idx, chunk in remaining:
                futures[pool.submit(one, chunk)] = idx
            for fut in futures:
                idx = futures[fut]
                print(f"    ... chunk {idx + 1}/{n} ...", flush=True)
                results[idx] = fut.result()
    else:
        for idx, chunk in remaining:
            el = time.time() - started
            print(
                f"    ... chunk {idx + 1}/{n} via {model} "
                f"({len(chunk.get('text', ''))} chars, {el / 60:.1f} min elapsed) ...",
                flush=True,
            )
            s = one(chunk)
            results[idx] = s
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
