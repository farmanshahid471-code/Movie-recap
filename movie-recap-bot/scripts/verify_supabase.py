"""Verify the Supabase pgvector setup end-to-end (no movie needed).

Run from the movie-recap-bot folder AFTER:
  1. creating the Supabase project
  2. running migrations/001_pgvector.sql once in the SQL editor (or setting
     SUPABASE_DB_URL so the pipeline bootstraps the schema itself)
  3. setting SUPABASE_URL + SUPABASE_SERVICE_KEY (and/or SUPABASE_DB_URL)
     in movie-recap-bot/.env

Usage:
    pip install numpy            # required
    python scripts/verify_supabase.py

What it checks:
  * credentials are present
  * the vector store picks the Supabase adapter
  * inserts rows through the REST API (embedding arrays -> vector column)
  * runs the match_cues() RPC and gets the right row back
  * cleans up after itself (the table is emptied)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    # Load movie-recap-bot/.env the same way the pipeline does.
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    from recap.config import load_config
    from recap.match import SupabaseVectorStore, make_store, supabase_configured

    if not supabase_configured():
        print("[X] No Supabase credentials found in movie-recap-bot/.env.")
        print("    Set SUPABASE_URL and SUPABASE_SERVICE_KEY (Project Settings ->")
        print("    API -> Project URL / service_role secret), or SUPABASE_DB_URL.")
        return 1

    print("[ok] Supabase credentials detected (URL or DB URL set).")

    cfg = load_config()
    import tempfile

    wd = Path(tempfile.mkdtemp(prefix="supabase-verify-"))
    store = make_store(cfg["semantic"], wd, dim=384)
    if not isinstance(store, SupabaseVectorStore):
        print(f"[X] Expected the Supabase store but got {type(store).__name__}. "
              "Check semantic.store in config.yaml (should be 'auto' or 'supabase').")
        return 1
    print(f"[ok] Vector store resolved to: {type(store).__name__}")

    # If only SUPABASE_DB_URL was set, make_store already ran the migration.
    if store.db_url:
        print("[ok] SUPABASE_DB_URL set -> schema bootstrap ran in the store init.")

    try:
        import numpy as np
    except ImportError:
        print("[X] numpy is required. Run: pip install numpy")
        return 1

    # Deterministic 384-dim "topic" vectors (embedding dim of all-MiniLM-L6-v2).
    cue_a = np.zeros(384, dtype=np.float32); cue_a[0] = 1.0
    cue_b = np.zeros(384, dtype=np.float32); cue_b[1] = 1.0
    cues = [
        {"text": "the detective opens the basement door", "start": 100.0, "end": 105.0},
        {"text": "everyone sings at the wedding", "start": 900.0, "end": 906.0},
    ]
    vecs = np.stack([cue_a, cue_b])

    print("    ... inserting 2 rows via the REST API ...")
    try:
        store.add_cues(cues, vecs)
    except Exception as exc:
        print(f"[X] Insert failed: {exc}")
        print("    - Did you run migrations/001_pgvector.sql in the SQL editor?")
        print("    - Are SUPABASE_URL / SUPABASE_SERVICE_KEY correct?")
        return 1
    print("[ok] Insert succeeded (embedding arrays -> vector column).")

    print("    ... querying match_cues() RPC ...")
    try:
        hits = store.search(cue_b, k=2, min_score=0.0)
    except Exception as exc:
        print(f"[X] RPC failed: {exc}")
        return 1
    if not hits or hits[0]["idx"] != 1 or "wedding" not in hits[0]["text"]:
        print(f"[X] Unexpected RPC result: {hits}")
        return 1
    print(f"[ok] RPC matched the right row: idx={hits[0]['idx']} "
          f"score={hits[0]['score']:.3f} text={hits[0]['text']!r}")

    # Cleanup: empty the table (the pipeline also clears it at each run start).
    try:
        store._rest("transcript_cues", method="DELETE", params={"idx": "gte.0"})
        print("[ok] Cleanup done (table emptied).")
    except Exception:
        pass
    store.close()

    print("\nSUPABASE VERIFICATION PASSED — the pipeline will now log")
    print("'* Vector store: Supabase (pgvector).' on the next auto-recap run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
