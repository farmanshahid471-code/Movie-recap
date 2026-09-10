"""Step C+ — lock the visuals to the SPOKEN narration with faster-whisper.

Why this module exists
----------------------
The timeline locks each narration sentence to a film window, and the beat
boundaries come from the TTS cue times. But cue times are *estimates* unless
the TTS backend reports word boundaries (only edge-tts does; OpenAI and
ElevenLabs return a bare mp3, and the old fallback guessed cue times
proportionally by word count — drifting seconds away from the voice).

The fix (suggested by the bot's owner, and it is the right one): run the
GENERATED narration audio back through faster-whisper. Whisper measures what
the viewer actually hears, so:

* every sentence cue start becomes the true start of its first spoken word,
* every cue carries word-level timestamps, whatever the TTS provider was,
* the timeline can then place its micro-cut boundaries ON WORDS — the visual
  switches at the exact moment the narrator moves to the next clause/subject,
  instead of at an arbitrary even split.

Cost: one extra faster-whisper pass over the (much shorter) narration mp3 —
a 15-minute voiceover takes a couple of minutes on CPU with the small model,
and the result is cached per audio content hash so re-runs are free.

Everything here degrades gracefully: no faster-whisper installed, a failed
transcription, or a bad word-to-sentence match just keeps the provider's own
cues — the pipeline never breaks because of this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .tts import TimedCue

# Word a narrator naturally has before a clause word where a cut feels good.
# Whisper word starts run a few tens of milliseconds late (onset detection),
# so switching the picture this much earlier makes the cut land *with* the
# voice instead of a hair after it.
DEFAULT_AUDIO_PRE_ROLL = 0.1

_PUNCT_STRIP = re.compile(r"[^\w'\u4e00-\u9fff\u0600-\u06ff\u00c0-\u024f]+")
# CJK punctuation that marks a clause boundary even without spaces.
_CJK_BOUNDARY = "，。！？；：、）】」』"


def _norm(token: str) -> str:
    """Normalize one spoken token for matching (case/punctuation insensitive)."""
    t = (token or "").strip().lower()
    t = _PUNCT_STRIP.sub("", t)
    return t


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def narration_words(
    mp3: Path,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> list[tuple[str, float, float]] | None:
    """Transcribe the narration audio and return every spoken word with times.

    Returns ``[(word, start, end)]`` in order, or ``None`` when faster-whisper
    is unavailable or the transcription fails (the caller keeps whatever
    timings the TTS provider gave it).
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None
    try:
        model = WhisperModel(model_size, device=device or "auto", compute_type="int8")
        segments, _info = model.transcribe(
            str(mp3),
            language=language,
            beam_size=1,
            word_timestamps=True,
        )
        words: list[tuple[str, float, float]] = []
        for seg in segments:
            for w in (getattr(seg, "words", None) or []):
                wt = (getattr(w, "word", "") or "").strip()
                if wt:
                    words.append((wt, float(w.start), float(w.end)))
        return words or None
    except Exception:
        return None


def map_words_to_sentences(
    words: list[tuple[str, float, float]],
    sentences: list[str],
) -> list[list[tuple[str, float, float]]] | None:
    """Assign whisper-measured words to the script sentences they belong to.

    Whisper-on-TTS is near-perfect but not byte-perfect: numbers change form
    ("5" vs "five"), punctuation appears and disappears, tokens merge. A
    bounded-lookahead greedy matcher walks both lists, tolerating insertions
    (whisper said something extra) and deletions (a script word went
    untranscribed) on either side.

    Returns one word-list per sentence (empty lists allowed), or ``None`` when
    the match is too poor to trust (below half the sentences matched) — the
    caller then keeps the provider's cues untouched.
    """
    if not words or not sentences:
        return None

    # Flatten the expected tokens, remembering which sentence each belongs to.
    expected: list[tuple[str, int]] = []
    for si, sent in enumerate(sentences):
        for tok in re.findall(r"[\w'\u4e00-\u9fff\u0600-\u06ff]+", sent or ""):
            n = _norm(tok)
            if n:
                expected.append((n, si))

    groups: list[list[tuple[str, float, float]]] = [[] for _ in sentences]
    LOOKAHEAD = 8
    wi = 0  # index into whisper words
    matched_sentences = 0
    last_sent = -1
    clean = 0        # tokens that matched exactly (directly or in lookahead)
    substituted = 0  # tokens accepted as blind substitutions

    for ei, (tok, si) in enumerate(expected):
        if wi >= len(words):
            break
        # Direct match?
        if _norm(words[wi][0]) == tok:
            groups[si].append(words[wi])
            if si != last_sent:
                matched_sentences += 1
                last_sent = si
            clean += 1
            wi += 1
            continue
        # Insertion: whisper produced extra tokens — look ahead for ours.
        found = None
        for k in range(wi + 1, min(wi + 1 + LOOKAHEAD, len(words))):
            if _norm(words[k][0]) == tok:
                found = k
                break
        if found is not None:
            groups[si].append(words[found])
            if si != last_sent:
                matched_sentences += 1
                last_sent = si
            clean += 1
            wi = found + 1
            continue
        # Deletion or substitution: does the NEXT expected token match the
        # current whisper word? Then this script token was skipped.
        if ei + 1 < len(expected) and _norm(words[wi][0]) == expected[ei + 1][0]:
            continue
        # Last resort: accept the mismatch as a substitution.
        groups[si].append(words[wi])
        if si != last_sent:
            matched_sentences += 1
            last_sent = si
        substituted += 1
        wi += 1

    # Quality gate: at least half the sentences located AND the matches must
    # be dominated by real agreement, not blind substitution.
    if matched_sentences < max(1, len(sentences) // 2):
        return None
    if clean == 0 or substituted > clean:
        return None
    return groups


def refine_cues(
    cues: list[TimedCue],
    sentences: list[str],
    words: list[tuple[str, float, float]],
    *,
    audio_pre_roll: float = DEFAULT_AUDIO_PRE_ROLL,
) -> list[TimedCue]:
    """Rewrite cue start/ends from the measured words; fill word timings.

    * A sentence whose words were located starts exactly when its first word
      is spoken (minus a small pre-roll so the picture/subtitle switch lands
      *with* the voice, not after it) and ends at its last word's end.
    * Sentences without a mapping keep the provider's times.
    * Monotonicity is enforced: a refined start can never move before the
      previous sentence's end.
    """
    groups = map_words_to_sentences(words, sentences)
    if groups is None:
        return cues

    out: list[TimedCue] = []
    prev_end = 0.0
    for cue, grp in zip(cues, groups):
        if grp:
            start = max(0.0, grp[0][1] - audio_pre_roll)
            end = grp[-1][2]
            if end <= start:
                end = start + cue.duration
            # keep the timeline monotone no matter what whisper reported
            start = max(start, prev_end)
            if end < start:
                end = start + 0.05
            new = TimedCue(cue.text, start, end, grp)
            prev_end = new.end
            out.append(new)
        else:
            start = max(cue.start, prev_end)
            end = max(cue.end, start)
            c = TimedCue(cue.text, start, end, cue.words)
            prev_end = c.end
            out.append(c)
    return out


def cues_from_words(
    sentences: list[str],
    words: list[tuple[str, float, float]],
    audio_span: float,
    *,
    audio_pre_roll: float = DEFAULT_AUDIO_PRE_ROLL,
) -> list[TimedCue] | None:
    """Build cues for providers that returned none (OpenAI/ElevenLabs path).

    The old fallback guessed times proportionally by word count; these are
    MEASURED from the audio. Returns None when the match is too poor.
    """
    groups = map_words_to_sentences(words, sentences)
    if groups is None:
        return None
    out: list[TimedCue] = []
    prev_end = 0.0
    for sent, grp in zip(sentences, groups):
        if grp:
            start = max(0.0, grp[0][1] - audio_pre_roll)
            end = grp[-1][2]
            start = max(start, prev_end)
            if end <= start:
                end = start + 0.5
            out.append(TimedCue(sent, start, end, grp))
            prev_end = out[-1].end
        else:
            # Unmatched sentence: give it the space up to the next match.
            out.append(TimedCue(sent, prev_end, prev_end + 0.5))
            prev_end = out[-1].end
    if out:
        out[-1].end = max(out[-1].end, min(audio_span, out[-1].end + 0.2))
    return out


def align_narration(
    mp3: Path,
    sentences: list[str],
    provider_cues: list[TimedCue],
    audio_span: float,
    workdir: Path,
    code: str = "en",
    *,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
    enabled: bool = True,
) -> tuple[list[TimedCue], bool]:
    """Align one language's narration to its own audio. Returns (cues, aligned).

    Cached: the whisper words are stored in ``<code>.whisper.json`` keyed by
    the mp3's content hash + model + device, so a re-render never re-runs the
    (slow) transcription pass. When alignment is unavailable — no
    faster-whisper, transcription failed, match too poor — the provider's cues
    are returned untouched and ``aligned`` is False.
    """
    if not enabled or not mp3.exists() or not sentences:
        return provider_cues, False

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    words_path = workdir / f"{code}.whisper.json"
    marker = workdir / f".align_{code}.marker.json"
    sig = json.dumps(
        {
            "audio": _hash_file(mp3),
            "model": model_size,
            "device": device or "auto",
            "lang": language or "",
        },
        sort_keys=True,
    )

    words: list[tuple[str, float, float]] | None = None
    try:
        if marker.exists() and words_path.exists() \
                and json.loads(marker.read_text(encoding="utf-8")).get("sig") == sig:
            raw = json.loads(words_path.read_text(encoding="utf-8"))
            words = [(w[0], float(w[1]), float(w[2])) for w in raw.get("words", [])] or None
    except Exception:
        words = None

    if words is None:
        print(f"  * [{code}] whisper-aligning the narration audio "
              f"({audio_span / 60:.1f} min; model={model_size}) ...", flush=True)
        words = narration_words(mp3, model_size=model_size, device=device,
                                language=language)
        if words is None:
            print(f"  ! [{code}] narration alignment unavailable "
                  "(no faster-whisper or transcription failed) — keeping the "
                  "TTS provider's own timing.", flush=True)
            return provider_cues, False
        try:
            words_path.write_text(
                json.dumps(
                    {"words": [[w, round(s, 3), round(e, 3)] for w, s, e in words]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            marker.write_text(json.dumps({"sig": sig}), encoding="utf-8")
        except OSError:
            pass

    if not provider_cues or not any(c.end > c.start for c in provider_cues):
        # Provider gave nothing measurable (openai/elevenlabs): the whisper
        # words ARE the timing now, replacing the old proportional guess.
        cues = cues_from_words(sentences, words, audio_span)
        if cues:
            print(f"  * [{code}] built {len(cues)} cues from whisper word "
                  "timestamps.", flush=True)
            return cues, True
        return provider_cues, False

    refined = refine_cues(provider_cues, sentences, words)
    if refined is provider_cues:
        return provider_cues, False
    print(f"  * [{code}] cue times refined from "
          f"{len(words)} measured words.", flush=True)
    return refined, True
