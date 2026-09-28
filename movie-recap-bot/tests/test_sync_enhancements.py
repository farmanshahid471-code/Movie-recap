"""Tests for frame-to-audio sync enhancements:
1. WhisperX forced alignment engine and JSON output structure
2. Dynamic audio & video retiming (atempo, freeze-framing, silence padding)
3. Strict spatial constraints to the LLM in script generation & summarization
4. Segmented TTS generation (beat_001.mp3, beat_002.mp3, ...)
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from recap import align, summarize, timeline, tts, video
from recap.tts import TimedCue
from recap.util import probe_duration


# ==============================================================================
# 1. WhisperX Forced Alignment Engine Tests
# ==============================================================================
def test_whisperx_alignment_json_structure() -> None:
    """Verify that align_narration writes WhisperX JSON with precise start/end word times."""
    tmp = Path(tempfile.mkdtemp(prefix="test-align-"))
    try:
        mp3 = tmp / "test.mp3"
        mp3.write_bytes(b"ID3mockaudio")

        # Mock narration_words to return sample word timestamps
        words_sample = [("hello", 0.12, 0.45), ("world", 0.50, 0.85)]
        sentences = ["Hello world"]
        provider_cues = [TimedCue("Hello world", 0.0, 1.0)]

        with patch("recap.align.narration_words", return_value=words_sample):
            cues, aligned = align.align_narration(
                mp3, sentences, provider_cues, audio_span=1.0, workdir=tmp, code="en"
            )
            assert aligned is True
            words_file = tmp / "en.whisper.json"
            assert words_file.exists()

            data = json.loads(words_file.read_text(encoding="utf-8"))
            assert data.get("engine") == "whisperx"
            assert "words" in data
            assert len(data["words"]) == 2
            assert data["words"][0]["word"] == "hello"
            assert abs(data["words"][0]["start"] - 0.12) < 1e-4
            assert abs(data["words"][0]["end"] - 0.45) < 1e-4

            # Verify that re-reading cached WhisperX JSON works seamlessly
            cues2, aligned2 = align.align_narration(
                mp3, sentences, provider_cues, audio_span=1.0, workdir=tmp, code="en"
            )
            assert aligned2 is True
            assert len(cues2) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_whisperx_word_structure_compatibility() -> None:
    """Verify that WhisperX JSON parser handles both dictionary and list word entries."""
    tmp = Path(tempfile.mkdtemp(prefix="test-align-compat-"))
    try:
        words_path = tmp / "en.whisper.json"
        marker = tmp / ".align_en.marker.json"
        mp3 = tmp / "test.mp3"
        mp3.write_bytes(b"ID3mockaudio")

        sig = json.dumps(
            {
                "audio": align._hash_file(mp3),
                "model": "small",
                "device": "auto",
                "lang": "",
            },
            sort_keys=True,
        )
        marker.write_text(json.dumps({"sig": sig}), encoding="utf-8")

        # Dict format
        words_path.write_text(
            json.dumps({"engine": "whisperx", "words": [{"word": "test", "start": 0.2, "end": 0.6}]}),
            encoding="utf-8",
        )
        cues, ok = align.align_narration(
            mp3, ["Test"], [TimedCue("Test", 0.0, 1.0)], audio_span=1.0, workdir=tmp
        )
        assert ok is True
        assert abs(cues[0].start - (0.2 - align.DEFAULT_AUDIO_PRE_ROLL)) < 1e-3

        # Legacy list format
        words_path.write_text(
            json.dumps({"words": [["test", 0.3, 0.7]]}),
            encoding="utf-8",
        )
        cues_legacy, ok_legacy = align.align_narration(
            mp3, ["Test"], [TimedCue("Test", 0.0, 1.0)], audio_span=1.0, workdir=tmp
        )
        assert ok_legacy is True
        assert abs(cues_legacy[0].start - (0.3 - align.DEFAULT_AUDIO_PRE_ROLL)) < 1e-3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# 2. Dynamic Audio & Video Retiming Tests
# ==============================================================================
def test_dynamic_retiming_logic() -> None:
    """Test compute_dynamic_retiming for atempo, freeze, and silence padding."""
    # Case 1: Audio slightly longer than video beat (5.5s audio for 5.0s clip)
    # Ratio = 5.5 / 5.0 = 1.10 <= 1.15 -> speed up by 1.10x with atempo
    r1 = timeline.compute_dynamic_retiming(audio_dur=5.5, video_dur=5.0, max_atempo=1.15)
    assert r1["action"] == "atempo"
    assert abs(r1["atempo"] - 1.10) < 0.01
    assert r1["freeze"] == 0.0
    assert r1["final_dur"] == 5.0

    # Case 2: Narration heavily overruns visual scene (10.0s audio for 5.0s clip)
    # Overrun: audio sped up by 1.15x -> eff_audio = 10.0 / 1.15 ≈ 8.696s.
    # Video freezes final frame for remainder: freeze = 8.696 - 5.0 = 3.696s.
    r2 = timeline.compute_dynamic_retiming(audio_dur=10.0, video_dur=5.0, max_atempo=1.15)
    assert r2["action"] == "freeze"
    assert abs(r2["atempo"] - 1.15) < 0.01
    assert abs(r2["freeze"] - (10.0 / 1.15 - 5.0)) < 0.05
    assert abs(r2["final_dur"] - (10.0 / 1.15)) < 0.05

    # Case 3: Video beat is longer than audio (7.0s video for 5.0s audio)
    # Silence padding: pad 2.0s silence at the end so it snaps to the next scene
    r3 = timeline.compute_dynamic_retiming(audio_dur=5.0, video_dur=7.0, max_atempo=1.15)
    assert r3["action"] == "silence_padding"
    assert r3["atempo"] == 1.0
    assert r3["freeze"] == 0.0
    assert abs(r3["silence_pad"] - 2.0) < 0.01
    assert r3["final_dur"] == 7.0


def test_timeline_apply_beat_retiming() -> None:
    """Verify that apply_beat_retiming_to_timeline modifies cut freezes for heavy overruns."""
    beats = [
        {"index": 0, "duration": 5.0, "cuts": [(10.0, 5.0, 0.0, 1.0)]},
        {"index": 1, "duration": 5.0, "cuts": [(20.0, 5.0, 0.0, 1.0)]},
        {"index": 2, "duration": 8.0, "cuts": [(30.0, 8.0, 0.0, 1.0)]},
    ]
    # Beat 0: slight overrun (5.5s audio -> atempo)
    # Beat 1: heavy overrun (9.2s audio -> freeze)
    # Beat 2: video longer (6.0s audio -> silence pad)
    audio_durs = [5.5, 9.2, 6.0]

    retimed = timeline.apply_beat_retiming_to_timeline(beats, audio_durs, max_atempo=1.15)
    assert retimed[0]["retiming"]["action"] == "atempo"
    assert retimed[0]["duration"] == 5.0
    assert retimed[0]["cuts"][0][2] == 0.0  # no freeze

    assert retimed[1]["retiming"]["action"] == "freeze"
    assert retimed[1]["cuts"][0][2] > 0.0   # freeze set on the cut!
    assert retimed[1]["duration"] > 5.0

    assert retimed[2]["retiming"]["action"] == "silence_padding"
    assert retimed[2]["retiming"]["silence_pad"] == 2.0
    assert retimed[2]["duration"] == 8.0


def test_ffmpeg_retime_audio_and_freeze_video() -> None:
    """Test actual FFmpeg execution for atempo, silence padding, and video freeze."""
    tmp = Path(tempfile.mkdtemp(prefix="test-retime-ffmpeg-"))
    try:
        sine_wav = tmp / "sine.wav"
        # 2.0s audio tone
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=2.0", str(sine_wav)],
            capture_output=True, check=True
        )

        # 1. Audio stretching: target 1.8s (ratio = 2.0 / 1.8 = 1.11x <= 1.15x)
        out_fast = tmp / "fast.wav"
        dur_fast = video.retime_audio(sine_wav, out_fast, target_duration=1.8, max_atempo=1.15)
        assert abs(dur_fast - 1.8) < 0.15

        # 2. Silence padding: target 3.0s (pad 1.0s silence)
        out_padded = tmp / "padded.wav"
        dur_pad = video.retime_audio(sine_wav, out_padded, target_duration=3.0, max_atempo=1.15)
        assert abs(dur_pad - 3.0) < 0.15

        # 3. Video freeze frame: 1.0s video frozen for 1.0s -> 2.0s total
        vid = tmp / "vid.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1.0:r=25", str(vid)],
            capture_output=True, check=True
        )
        vid_frozen = tmp / "vid_frozen.mp4"
        video.freeze_frame_video(vid, vid_frozen, freeze_duration=1.0)
        assert vid_frozen.exists()
        assert abs(probe_duration(vid_frozen) - 2.0) < 0.2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# 3. Strict Spatial Constraints to LLM Tests
# ==============================================================================
def test_spatial_constraint_formula() -> None:
    """Verify Beat Duration * 2.5 = Max Words formula."""
    # 8 second scene: 8 * 2.5 = 20 words
    assert summarize.calculate_beat_max_words(8.0) == 20
    # 4 second beat: 4 * 2.5 = 10 words
    assert summarize.calculate_beat_max_words(4.0) == 10
    # 1.5 second cut: 1.5 * 2.5 = 3.75 -> 4 words
    assert summarize.calculate_beat_max_words(1.5) == 4


def test_summarize_chunk_prompt_contains_strict_constraint() -> None:
    """Verify system prompt and visual block contain the mathematical constraint."""
    visual = [
        {"t": 10.0, "text": "Bonnie arrives with her toys."},
        {"t": 18.0, "text": "Jessie looks at the window."},
    ]
    block = summarize._visual_block(visual, chunk_start=0.0, chunk_end=26.0)
    assert "Beat Duration: 8.0s" in block
    assert "Limit: exactly 15 to 20 words" in block
    assert "Formula applied: Beat Duration (seconds) * 2.5 Words Per Second = Max Words" in block

    # Verify that summarize_chunks injects the strict limit into system prompt
    captured_sys_prompts = []

    def mock_complete(prov, model, sys_prompt, user_prompt, **kwargs):
        captured_sys_prompts.append(sys_prompt)
        return "[00:00:10] Bonnie plays. [00:00:18] Jessie watches."

    with patch("recap.llm.complete", side_effect=mock_complete):
        chunk = {
            "index": 0,
            "start": 0.0,
            "end": 8.0,
            "text": "[00:00:00] Bonnie speaks.",
            "visual": visual,
        }
        summarize.summarize_chunks([chunk], {"provider": "mock", "model": "m"})

    assert len(captured_sys_prompts) == 1
    sys_p = captured_sys_prompts[0]
    # Check for the specified instruction format:
    # "This scene is 8 seconds long. Your summary must be exactly 15 to 20 words."
    assert "This scene is 8 seconds long." in sys_p
    assert "Your summary must be exactly 15 to 20 words." in sys_p
    assert "Formula: Beat Duration" in sys_p


# ==============================================================================
# 4. Segmented TTS Generation Tests
# ==============================================================================
def test_segmented_tts_generation() -> None:
    """Verify generation of individual audio files per beat (beat_001.mp3, beat_002.mp3, ...)."""
    tmp = Path(tempfile.mkdtemp(prefix="test-seg-tts-"))
    try:
        sentences = [
            "The story begins in an abandoned castle.",
            "A brave explorer opens the heavy gate.",
            "Inside awaits an ancient relic of immense power.",
        ]

        class DummyProvider:
            name = "dummy"

            def synthesize(self, sents, voice, out_mp3):
                out_mp3.parent.mkdir(parents=True, exist_ok=True)
                out_mp3.write_bytes(b"ID3dummy")
                cues = []
                t = 0.0
                for s in sents:
                    cues.append(TimedCue(s, t, t + 2.0))
                    t += 2.0
                return cues

        prov = DummyProvider()
        mp3, cues = tts.synthesize_language(
            sentences, {"code": "en", "voice": "alloy"}, tmp, prov
        )

        assert mp3.exists()
        assert len(cues) == 3

        # Verify individual beat files were created
        beat_files = tts.get_beat_files(tmp, "en")
        assert len(beat_files) == 3
        assert beat_files[0].name == "beat_001.mp3"
        assert beat_files[1].name == "beat_002.mp3"
        assert beat_files[2].name == "beat_003.mp3"
        for bf in beat_files:
            assert bf.exists()
            assert bf.stat().st_size > 0

        # Verify beats.json manifest
        manifest_file = tmp / "en.beats.json"
        assert manifest_file.exists()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert len(manifest) == 3
        assert manifest[0]["file"] == "beat_001.mp3"
        assert manifest[0]["beat"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# 5. Target Duration Constraint & Subtitle Discovery / Cache Tests
# ==============================================================================
def test_select_beats_for_target_duration() -> None:
    """Verify that select_beats_for_target constrains full-movie beats to target duration."""
    from recap import beats

    # Simulate an 87-minute movie (5220s) with 100 beats of ~52s each
    # At 175 wpm, each beat is ~150 words -> 15,000 words total (87 minutes)
    movie_beats = []
    t = 0.0
    for i in range(100):
        movie_beats.append({
            "start_ts": t,
            "end_ts": t + 52.0,
            "duration": 52.0,
            "transcript_lines": [{"text": f"Character {i} talks about the quest."}] if i % 2 == 0 else [],
            "vision_notes": [{"t": t + 10.0, "text": f"Visual action {i}", "confidence": "high"}] if i % 3 == 0 else [],
            "shot_count": 4,
        })
        t += 52.0

    # User sets duration to 1200s (20 minutes) -> target_words = 3000 words at 150 wpm
    target_words = 3000
    chosen = beats.select_beats_for_target(movie_beats, target_words, wpm=150)

    assert len(chosen) < len(movie_beats)
    # Check that opening beat and ending beat are always included
    assert chosen[0]["start_ts"] == 0.0
    assert chosen[-1]["end_ts"] == movie_beats[-1]["end_ts"]

    # Total word budget of chosen beats should match ~target_words (not 15,000 words!)
    total_chosen_words = sum(
        beats.word_budget_for_duration(b["duration"], wpm=150)[1] for b in chosen
    )
    assert total_chosen_words <= target_words * 1.25
    assert total_chosen_words >= target_words * 0.70


def test_find_subtitle_near_with_and_without_subtitles() -> None:
    """Verify that explicitly empty subtitle ('') or auto_discover=False does NOT grab nearby srt."""
    from recap import dialogue

    tmp = Path(tempfile.mkdtemp(prefix="test-sub-find-"))
    try:
        movie = tmp / "my_movie.mp4"
        movie.write_bytes(b"mockvideo")
        srt = tmp / "my_movie.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

        # 1. Normal discovery finds the nearby .srt
        found = dialogue.find_subtitle_near(movie, None)
        assert found == srt

        # 2. When user explicitly specifies empty string ("without subtitles"), it must NOT auto-discover
        found_empty = dialogue.find_subtitle_near(movie, extra="")
        assert found_empty is None

        # 3. When auto_discover is False, it must NOT auto-discover
        found_no_auto = dialogue.find_subtitle_near(movie, None, auto_discover=False)
        assert found_no_auto is None

        # 4. Explicit valid path is always respected
        found_explicit = dialogue.find_subtitle_near(movie, extra=str(srt))
        assert found_explicit == srt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

