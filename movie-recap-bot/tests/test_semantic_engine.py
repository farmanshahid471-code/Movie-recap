"""Standalone smoke tests for the Step A-F semantic engine logic.

Pure-logic only (no network, no model weights): chunking, JSON-array parsing,
cosine matching with the local store, de-duplication and fallback anchors.

Run from the movie-recap-bot folder:

    python tests/test_semantic_engine.py

The match tests need numpy (`pip install numpy`); everything else is stdlib.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.chunk import chunk_cues  # noqa: E402
from recap.script import parse_sentences_json  # noqa: E402


def test_chunking_overlap() -> None:
    # a 16-minute film, one cue every 8s
    cues = [
        {"text": f"line {i}", "start": i * 8.0, "end": i * 8.0 + 5.0}
        for i in range(120)
    ]
    chunks = chunk_cues(cues, window_seconds=300, overlap_seconds=30)
    assert chunks, "no chunks produced"
    assert abs(chunks[0]["start"]) < 1e-6
    assert abs(chunks[1]["start"] - 270.0) < 1e-6, "second window must start 30s before the first ends"
    overlap = {c["text"] for c in chunks[0]["cues"]} & {c["text"] for c in chunks[1]["cues"]}
    assert overlap, "overlap windows must share cues across the seam"


def test_parse_json_sentences() -> None:
    raw = '''Sure! Here is the recap:
```json
["He wakes up tied to a chair.", "The van races to the airport.",]
```
Hope it helps!'''
    s = parse_sentences_json(raw)
    assert s == ["He wakes up tied to a chair.", "The van races to the airport."]

    # prose fallback keeps one sentence per line
    s2 = parse_sentences_json("She runs.\nHe screams.\n")
    assert s2 == ["She runs.", "He screams."]


def test_match_local_store_and_dedupe() -> None:
    np = __import__("numpy")
    from recap.match import LocalVectorStore, map_beats

    tmp = tempfile.mkdtemp()
    store = LocalVectorStore(os.path.join(tmp, "v.db"), dim=8)
    cues = [
        {"text": "the car crashes into the tree", "start": 100.0, "end": 104.0},
        {"text": "he opens the door of the basement", "start": 400.0, "end": 406.0},
        {"text": "the detective finds the hidden letter", "start": 800.0, "end": 810.0},
        {"text": "everyone sings at the wedding", "start": 1200.0, "end": 1207.0},
    ]
    basis = {
        "crash": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "basement": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "letter": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        "wedding": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    }
    cue_vecs = np.array([basis["crash"], basis["basement"],
                         basis["letter"], basis["wedding"]], dtype=np.float32)
    store.add_cues(cues, cue_vecs)
    hit = store.search(basis["letter"], k=2)
    assert hit and hit[0]["idx"] == 2 and hit[0]["start"] == 800.0

    class FakeEmbedder:
        dim = 8

        def encode(self, texts):
            rows = []
            for t in texts:
                key = ("basement" if "basement" in t else
                       "letter" if "letter" in t else
                       "wedding" if "sing" in t else "crash")
                rows.append(basis[key] + np.random.RandomState(len(t)).normal(0, 0.05, 8))
            return np.array(rows, dtype=np.float32)

    sentences = [
        "A car plows into a tree on the highway.",
        "Later he creeps down into the basement.",
        "A letter is found by the detective.",
        "They all sing at the big wedding.",
    ]
    beats = map_beats(sentences, cues, FakeEmbedder(), store,
                      top_k=4, min_score=0.3, movie_duration=1300.0)
    used = [b["cue_idx"] for b in beats]
    assert len(set(used)) == 4, f"no moment should be reused: {used}"
    assert beats[0]["cue_idx"] == 0 and beats[3]["cue_idx"] == 3

    # hopeless lines fall back to even spacing, in story order
    weak_store = LocalVectorStore(os.path.join(tmp, "v2.db"), dim=8)
    weak = map_beats(["completely unrelated noise", "more unrelated noise"],
                     cues, FakeEmbedder(), weak_store,
                     top_k=3, min_score=0.99, movie_duration=1400.0)
    assert all(b["fallback"] for b in weak)
    assert weak[1]["start"] > weak[0]["start"]


if __name__ == "__main__":
    test_chunking_overlap()
    print("ok: contextual chunking")
    test_parse_json_sentences()
    print("ok: JSON sentence parsing")
    test_match_local_store_and_dedupe()
    print("ok: semantic matching / store / dedupe / fallback")
    print("\nALL SEMANTIC ENGINE SMOKE TESTS PASSED")
