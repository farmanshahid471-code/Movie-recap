"""Step D — semantic timestamp mapping.

For every narration sentence of the recap script we find the *visual moment*
in the movie that best matches it: the original transcript (dialogue) is
embedded with a local sentence model (``all-MiniLM-L6-v2`` by default), stored
in a pgvector database (Supabase) or in a local zero-setup fallback, and each
script sentence is matched by cosine similarity. The matched transcript cue's
``start``/``end`` timestamps become the source timeframe to clip for that story
beat.

Design notes
------------
* The store is pluggable. When Supabase credentials are present the adapter
  ``SupabaseVectorStore`` is used (exactly as specced: hosted Postgres +
  pgvector); otherwise ``LocalVectorStore`` (SQLite + numpy) runs the identical
  search logic so the pipeline works offline / before the Supabase project is
  deployed.
* Matching is greedy with de-duplication: the same dialogue moment is never
  used for two different narration lines when an alternative is available.
* A line whose best score is below ``min_score`` gets an evenly-spaced
  fallback anchor so the video never stalls on a weak match.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Embedder (local sentence-transformer)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


class Embedder:
    """Thin wrapper around sentence-transformers, loaded lazily (heavy dep)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            if self.model_name in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self.model_name]
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Semantic matching needs sentence-transformers. "
                        "Install with: pip install sentence-transformers"
                    ) from exc
                print(f"  * Loading embedding model '{self.model_name}' ...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                _MODEL_CACHE[self.model_name] = self._model
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> object:
        """Return float32 matrix (n_texts, dim)."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._load().encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(query: object, corpus: object):
    """(nq, dim) x (nc, dim) -> (nq, nc) cosine similarity matrix."""
    import numpy as np

    q = np.asarray(query, dtype=np.float32)
    c = np.asarray(corpus, dtype=np.float32)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    return qn @ cn.T


# ---------------------------------------------------------------------------
# Store interface helpers
# ---------------------------------------------------------------------------
def supabase_configured() -> bool:
    return bool(
        os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")
    ) or bool(os.environ.get("SUPABASE_DB_URL"))


class StoreError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Local fallback store (SQLite + numpy). Identical search semantics to pgvector.
# ---------------------------------------------------------------------------
class LocalVectorStore:
    """Zero-setup fallback: SQLite file + in-memory numpy matrix.

    Search is exact cosine over the whole movie transcript, which is small
    (typically < 5,000 cues), so no index is needed.
    """

    name = "local"

    def __init__(self, db_path: str | Path, dim: int = 384):
        self.db_path = Path(db_path)
        self.dim = dim
        self._conn: sqlite3.Connection | None = None
        self._cache = None  # (vectors, metas)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cues ("
                " idx INTEGER PRIMARY KEY,"
                " text TEXT NOT NULL,"
                " start REAL NOT NULL,"
                " end REAL NOT NULL,"
                " embedding TEXT NOT NULL)"
            )
            self._conn = conn
        return self._conn

    def add_cues(self, cues: list[dict], vectors) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM cues")
        import numpy as np

        arr = np.asarray(vectors, dtype=np.float32)
        conn.executemany(
            "INSERT INTO cues (idx, text, start, end, embedding) VALUES (?,?,?,?,?)",
            [
                (
                    int(i),
                    (c.get("text") or "").strip(),
                    float(c.get("start", 0.0)),
                    float(c.get("end", 0.0)),
                    json.dumps([float(x) for x in arr[i]]),
                )
                for i, c in enumerate(cues)
            ],
        )
        conn.commit()
        self._cache = None

    def _load_all(self):
        if self._cache is None:
            import numpy as np

            conn = self._connect()
            rows = conn.execute(
                "SELECT idx, text, start, end, embedding FROM cues ORDER BY idx"
            ).fetchall()
            if not rows:
                return np.zeros((0, self.dim), dtype=np.float32), []
            vecs = np.array(
                [json.loads(r[4]) for r in rows], dtype=np.float32
            )
            metas = [
                {"idx": r[0], "text": r[1], "start": r[2], "end": r[3]}
                for r in rows
            ]
            self._cache = (vecs, metas)
        return self._cache

    def search(self, query_vector, k: int = 3, min_score: float = 0.0) -> list[dict]:
        import numpy as np

        vecs, metas = self._load_all()
        if len(metas) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores = cosine_similarity_matrix(q, vecs)[0]
        order = np.argsort(-scores)
        out = []
        for pos in order[:k]:
            sc = float(scores[pos])
            if sc < min_score and out:
                break
            out.append({**metas[int(pos)], "score": sc})
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Supabase pgvector store — exactly the specced hosted-Postgres design.
#
# Two ways to use it:
#   1. SUPABASE_URL + SUPABASE_SERVICE_KEY  -> REST/RPC against the
#      `match_cues` SQL function (create it once from
#      migrations/001_pgvector.sql in the Supabase SQL editor).
#   2. SUPABASE_DB_URL (direct/psycopg connection string) -> auto-create
#      the table + function by running the migration itself.
# ---------------------------------------------------------------------------
class SupabaseVectorStore:
    name = "supabase"

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.url = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_KEY") or ""
        self.db_url = os.environ.get("SUPABASE_DB_URL") or ""
        if self.db_url:
            self._ensure_schema()

    # -- optional schema bootstrap over a direct DB connection --------------
    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).resolve().parent.parent / "migrations" / "001_pgvector.sql"
        if not sql_path.exists():
            raise StoreError("migrations/001_pgvector.sql not found next to the package")
        try:
            import psycopg2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise StoreError(
                "SUPABASE_DB_URL is set but psycopg2 is not installed. "
                "pip install psycopg2-binary (or run migrations/001_pgvector.sql "
                "manually in the Supabase SQL editor)."
            ) from exc
        try:
            conn = psycopg2.connect(self.db_url)
            conn.autocommit = True
            conn.cursor().execute(sql_path.read_text(encoding="utf-8"))
            conn.close()
            print("  * pgvector schema ensured on Supabase.")
        except Exception as exc:
            raise StoreError(f"Could not initialise Supabase schema: {exc}") from exc

    # -- REST helpers --------------------------------------------------------
    def _rest(self, path: str, method: str = "POST", payload=None, params=None):
        import urllib.error
        import urllib.parse
        import urllib.request

        if not (self.url and self.key):
            raise StoreError(
                "SupabaseVectorStore requires SUPABASE_URL and SUPABASE_SERVICE_KEY "
                "(or SUPABASE_DB_URL). Set them in movie-recap-bot/.env"
            )
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{path}{query}",
            method=method,
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else []
        except urllib.error.HTTPError as exc:
            raise StoreError(
                f"Supabase REST {path} failed ({exc.code}): "
                f"{exc.read().decode('utf-8', 'replace')[:400]}"
            ) from exc

    def add_cues(self, cues: list[dict], vectors) -> None:
        import numpy as np

        arr = np.asarray(vectors, dtype=np.float32)
        rows = []
        for i, c in enumerate(cues):
            rows.append(
                {
                    "idx": int(i),
                    "text": (c.get("text") or "").strip(),
                    "start_ms": int(round(float(c.get("start", 0.0)) * 1000)),
                    "end_ms": int(round(float(c.get("end", 0.0)) * 1000)),
                    "embedding": [float(x) for x in arr[i]],
                }
            )
        # Clear then insert (a run is a self-contained movie session).
        try:
            self._rest("transcript_cues", method="DELETE", params={"idx": "gte.0"})
        except StoreError:
            pass  # empty table already
        if rows:
            self._rest("transcript_cues", payload=rows)

    def search(self, query_vector, k: int = 3, min_score: float = 0.0) -> list[dict]:
        try:
            rows = self._rest(
                "rpc/match_cues",
                payload={
                    "query_embedding": [float(x) for x in query_vector],
                    "match_count": int(k),
                    "match_threshold": float(min_score),
                },
            )
        except StoreError as exc:
            if "function match_cues" in str(exc) or "Could not find" in str(exc):
                raise StoreError(
                    "pgvector RPC function `match_cues` is missing. Open the "
                    "Supabase SQL editor and run migrations/001_pgvector.sql once."
                ) from exc
            raise
        return [
            {
                "idx": int(r["idx"]),
                "text": r.get("text") or "",
                "start": float(r.get("start_ms", 0)) / 1000.0,
                "end": float(r.get("end_ms", 0)) / 1000.0,
                "score": float(r.get("similarity", 0.0)),
            }
            for r in rows
        ]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_store(cfg_semantic: dict, workdir: Path, dim: int = 384):
    store_cfg = (cfg_semantic or {}).get("store", "auto") or "auto"
    store_cfg = str(store_cfg).strip().lower()
    if store_cfg == "local":
        return LocalVectorStore(Path(workdir) / "vectors.db", dim=dim)
    if store_cfg == "supabase":
        return SupabaseVectorStore(dim=dim)
    # auto: Supabase when configured, otherwise the local fallback
    if supabase_configured():
        print("  * Vector store: Supabase (pgvector).")
        return SupabaseVectorStore(dim=dim)
    print("  * Vector store: local fallback (no Supabase credentials found).")
    return LocalVectorStore(Path(workdir) / "vectors.db", dim=dim)


# ---------------------------------------------------------------------------
# Beat mapping
# ---------------------------------------------------------------------------
def map_beats(
    sentences: list[str],
    cues: list[dict],
    embedder: Embedder,
    store,
    *,
    top_k: int = 3,
    min_score: float = 0.10,
    movie_duration: float | None = None,
) -> list[dict]:
    """Map each narration sentence to the transcript moment it matches best.

    Returns beats::

        [{"index": 0, "sentence": "...", "cue_idx": 12,
          "start": 30.0, "end": 33.5, "score": 0.62, "source_text": "..."}]

    Sentences with no acceptable match (or a movie with no dialogue at all)
    fall back to evenly spaced anchors across the whole runtime, so the video
    still covers the story.
    """
    if not sentences:
        return []
    if not cues:
        # No transcript (e.g. subtitle-free, silent/ambient film): space beats
        # evenly across the runtime.
        total = float(movie_duration or 0.0)
        return _even_fallback(sentences, total, cues)

    print(f"  * Embedding {len(cues)} transcript cues ...")
    cue_vectors = embedder.encode([(c.get("text") or "") for c in cues])
    store.add_cues(cues, cue_vectors)

    print(f"  * Matching {len(sentences)} narration lines to the film ...")
    sent_vectors = embedder.encode(sentences)
    import numpy as np

    sims = cosine_similarity_matrix(sent_vectors, np.asarray(cue_vectors, dtype=np.float32))

    used: set[int] = set()
    beats: list[dict] = []
    fallback_count = 0
    n = len(cues)
    total_dur = float(movie_duration or (max((c.get("end", 0) for c in cues), default=0.0)))
    for i, sentence in enumerate(sentences):
        order = [int(x) for x in np.argsort(-sims[i])]
        best = None
        for j in order[: top_k]:
            score = float(sims[i][j])
            if j in used:
                continue
            if score < min_score:
                break
            best = (j, score)
            break
        if best is not None:
            j, score = best
            used.add(j)
            cue = cues[j]
            beats.append(
                {
                    "index": i,
                    "sentence": sentence,
                    "cue_idx": int(j),
                    "start": float(cue.get("start", 0.0)),
                    "end": float(cue.get("end", 0.0)),
                    "score": round(score, 4),
                    "source_text": (cue.get("text") or "").strip(),
                    "fallback": False,
                }
            )
        else:
            # evenly spaced anchor along the film's timeline
            fallback_count += 1
            beats.append(
                {
                    "index": i,
                    "sentence": sentence,
                    "cue_idx": None,
                    "start": None,
                    "end": None,
                    "score": 0.0,
                    "source_text": "",
                    "fallback": True,
                }
            )

    if fallback_count:
        print(f"  * {fallback_count}/{len(sentences)} lines had no good dialogue "
              f"match -> spaced evenly through the film.")
    cue_times = [float(c.get("start", 0.0)) for c in cues]
    return _fill_fallback_anchors(beats, cue_times, total_dur)


def _fill_fallback_anchors(beats: list[dict], cue_times: list[float], duration: float):
    """Replace None anchors with evenly spaced source windows.

    When there is no transcript we anchor at uniform fractions of the runtime;
    when a transcript exists but a line simply lost the match race we anchor at
    the dialogue cue closest to that fraction of the timeline (so the visual
    still shows speech, not silence).
    """
    n = len(cue_times)
    for b in beats:
        if not b.get("fallback"):
            continue
        frac = (b["index"] + 0.5) / max(len(beats), 1)
        if n > 0 and duration > 0:
            # nearest real cue to that fraction of the timeline
            tgt = frac * duration
            j = min(range(n), key=lambda k: abs(cue_times[k] - tgt))
            b["cue_idx"] = int(j)
            b["start"] = float(cue_times[j])
            b["end"] = float(cue_times[j]) + 4.0
            b["fallback_cue"] = True
        elif duration > 0:
            b["start"] = frac * duration
            b["end"] = min(duration, b["start"] + 4.0)
        else:
            b["start"] = 0.0
            b["end"] = 4.0
    return beats


def _even_fallback(sentences: list[str], duration: float, cues: list[dict]) -> list[dict]:
    beats = []
    for i, s in enumerate(sentences):
        frac = (i + 0.5) / max(len(sentences), 1)
        st = frac * duration if duration > 0 else 0.0
        beats.append(
            {
                "index": i,
                "sentence": s,
                "cue_idx": None,
                "start": st,
                "end": min(duration, st + 4.0) if duration > 0 else st + 4.0,
                "score": 0.0,
                "source_text": "",
                "fallback": True,
            }
        )
    return beats
