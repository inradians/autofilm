"""scripts/run_loop_smoke_test.py — behavior tests for the autoresearch loop.

Exercises every piece of the loop infrastructure end-to-end WITHOUT hitting
any API. Builds a synthetic experiment with the full set of artifacts,
then verifies:

  1. production_bible.build_production_bible_json() captures every stage
     and indexes artifacts by their canonical keys.

  2. carryover.plan_carryover() correctly maps the critic's `changes`
     array to invalidation flags for every supported keyword family,
     respects priority thresholds, and routes vague changes to
     manual_review.

  3. Experiment.new_iteration() copies forward exactly the artifacts
     NOT flagged for regen, applies cascade rules, inherits seed, and
     stamps parent_exp / carryover.json.

  4. The CLIs respond to --help, --list, etc. without imports failing.

Usage
-----
    python scripts/run_loop_smoke_test.py

    # Verbose (per-test detail):
    python scripts/run_loop_smoke_test.py -v

    # Stop on first failure:
    python scripts/run_loop_smoke_test.py --fail-fast
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

# Project root on path so we can import prepare/produce/etc.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Test result tracking ──────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.status = "PENDING"
        self.note = ""

    def __repr__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}.get(self.status, "?")
        note = f"  {self.note}" if self.note else ""
        return f"  {icon}  {self.name}{note}"


_results: list[TestResult] = []
_verbose = False
_fail_fast = False


def test(name: str):
    """Decorator: register a function as a smoke test."""
    def deco(fn: Callable[..., None]) -> Callable[..., None]:
        def wrapper(*args, **kwargs) -> TestResult:
            r = TestResult(name)
            _results.append(r)
            if _verbose:
                print(f"\n  ▸ {name}")
            try:
                fn(*args, **kwargs)
                r.status = "PASS"
            except AssertionError as e:
                r.status = "FAIL"
                r.note = str(e)[:200]
                if _verbose:
                    traceback.print_exc()
                if _fail_fast:
                    raise
            except Exception as e:  # noqa: BLE001
                r.status = "FAIL"
                r.note = f"{type(e).__name__}: {str(e)[:180]}"
                if _verbose:
                    traceback.print_exc()
                if _fail_fast:
                    raise
            return r
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# ── Synthetic artifact builders ───────────────────────────────────────────────

def _tiny_png() -> bytes:
    """A 1x1 transparent PNG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _tiny_wav() -> bytes:
    """A 1-sample silent WAV."""
    import struct
    # RIFF header for a 1-sample, 16-bit, mono, 44.1kHz file.
    sample = b"\x00\x00"
    pcm    = sample
    fmt    = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt   += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
    fmt   += b"data" + struct.pack("<I", len(pcm)) + pcm
    return fmt


def _tiny_mp4() -> bytes:
    """Minimal valid MP4 header (ftyp box only). Enough for a smoke test
    that just checks file existence — not playback."""
    # ftyp box: size(4) + 'ftyp' + brand 'mp42' + minor + compat brands
    return (
        b"\x00\x00\x00\x20"      # box size = 32 bytes
        b"ftyp"
        b"mp42"                  # major brand
        b"\x00\x00\x00\x00"      # minor version
        b"mp42isom"              # compat brands
        b"\x00\x00\x00\x08mdat"  # empty mdat box
    )


def _build_fixture(exp_root: Path, scene_id: str = "scene_001",
                   shot_id: str = "shot_001", char_id: str = "jeff") -> None:
    """Create every artifact a finished experiment would have."""
    (exp_root / "book.txt").write_text("smoke_book")
    (exp_root / "seed.txt").write_text("424242")

    # Stage 1: script
    script = {
        "title": "Smoke Test",
        "scenes": [{
            "id": scene_id,
            "location": "Test Forest",
            "characters": [char_id],
            "elements": [
                {"type": "action", "text": "A clearing in the forest."},
                {"type": "dialogue", "text": "Hello.", "character": char_id},
                {"type": "narration", "text": "The story begins.",
                 "character": "narrator"},
            ],
        }],
        "characters": [{"id": char_id, "name": "Jeff",
                        "description": "A weary traveler"}],
    }
    (exp_root / "script.json").write_text(json.dumps(script))

    # Stage 2: cast & locations
    cast = [{"id": char_id, "name": "Jeff",
             "actor": "fictional", "description": "A weary traveler"}]
    (exp_root / "cast.json").write_text(json.dumps(cast))

    moodboard_path = "location_moodboards/forest/00.png"
    (exp_root / moodboard_path).parent.mkdir(parents=True, exist_ok=True)
    (exp_root / moodboard_path).write_bytes(_tiny_png())

    locations = [{
        "slug": "forest",
        "description": "A misty forest",
        "color_palette": ["amber", "teal"],
        "moodboard_paths": [str(exp_root / moodboard_path)],
    }]
    (exp_root / "locations.json").write_text(json.dumps(locations))

    # Stage 3: lookbook
    lookbook = {
        "grade_description": "warm shadows, lifted blacks, teal highlights",
        "ffmpeg_grade":      "eq=contrast=1.1:saturation=0.9",
        "style_keywords":    ["35mm", "anamorphic", "grain"],
    }
    (exp_root / "lookbook.json").write_text(json.dumps(lookbook))
    (exp_root / "lookbook").mkdir(exist_ok=True)
    (exp_root / "lookbook" / "style_frame.png").write_bytes(_tiny_png())

    # Stage 4: references — references/{char_id}/{scene_id}.png
    (exp_root / "references" / char_id).mkdir(parents=True, exist_ok=True)
    (exp_root / "references" / char_id / f"{scene_id}.png").write_bytes(_tiny_png())

    # Stage 5: storyboard
    storyboard = {
        scene_id: [{
            "shot_id":    shot_id,
            "shot_size":  "MS",
            "angle":      "eye-level",
            "duration_seconds": 6,
            "action":     "Jeff walks.",
        }],
    }
    (exp_root / "storyboard.json").write_text(json.dumps(storyboard))

    # Stage 6: music
    (exp_root / "music").mkdir(exist_ok=True)
    (exp_root / "music" / f"{scene_id}.wav").write_bytes(_tiny_wav())

    # Stage 6.5: narration
    (exp_root / "narration").mkdir(exist_ok=True)
    (exp_root / "narration" / f"{scene_id}.mp3").write_bytes(_tiny_wav())

    # Stage 7: frames — frames/{scene_id}/{shot_id}.png
    (exp_root / "frames" / scene_id).mkdir(parents=True, exist_ok=True)
    (exp_root / "frames" / scene_id / f"{shot_id}.png").write_bytes(_tiny_png())

    # Stage 8: clips — clips/{scene_id}/{shot_id}/take_0.mp4
    clip_dir = exp_root / "clips" / scene_id / shot_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "take_0.mp4").write_bytes(_tiny_mp4())
    (clip_dir / "take_1.mp4").write_bytes(_tiny_mp4())

    # Stage 9: edl
    edl = {scene_id: [{"shot_id": shot_id, "take": 0}]}
    (exp_root / "edl.json").write_text(json.dumps(edl))

    # SFX
    (exp_root / "sfx" / scene_id).mkdir(parents=True, exist_ok=True)
    (exp_root / "sfx" / scene_id / "ambient.wav").write_bytes(_tiny_wav())

    # Stage 10: final
    (exp_root / "final.mp4").write_bytes(_tiny_mp4())

    # Snapshot of produce.py for the bible's config section.
    src = PROJECT_ROOT / "produce.py"
    if src.exists():
        (exp_root / "produce.py").write_text(src.read_text())


def _make_metric(changes: list[dict], film_loss: float = 0.42) -> dict:
    return {
        "film_loss": film_loss,
        "scores": {
            "cinematography": 0.5, "color": 0.4, "sound": 0.3,
            "acting": 0.5, "continuity": 0.4, "fidelity": 0.4,
        },
        "weights": {
            "cinematography": 0.20, "color": 0.15, "sound": 0.15,
            "acting": 0.20, "continuity": 0.15, "fidelity": 0.15,
        },
        "changes": changes,
    }


# ── Test fixtures ─────────────────────────────────────────────────────────────

class _IsolatedExperimentsDir:
    """Context manager that overrides EXPERIMENTS_DIR at module level so
    Experiment.new_iteration() works against a temp dir without polluting
    the real experiments tree."""
    def __init__(self):
        self.tmp:  tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None
        self.orig_exp_dir: Path | None = None

    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory(prefix="autofilm_smoke_")
        self.path = Path(self.tmp.name) / "experiments"
        self.path.mkdir()
        # Swap module-level EXPERIMENTS_DIR for the duration of the test.
        import prepare
        self.orig_exp_dir = prepare.EXPERIMENTS_DIR
        prepare.EXPERIMENTS_DIR = self.path
        return self.path

    def __exit__(self, *exc) -> None:
        import prepare
        if self.orig_exp_dir is not None:
            prepare.EXPERIMENTS_DIR = self.orig_exp_dir
        if self.tmp:
            self.tmp.cleanup()


# ── Tests: production_bible.json ──────────────────────────────────────────────

@test("production_bible_json: captures all stages")
def t_pb_all_stages() -> None:
    from prepare import Experiment
    from production_bible import build_production_bible_json

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)

        exp = Experiment(exp_id="exp_001", root=exp_root)
        out_path = build_production_bible_json(exp)

        bible = json.loads(out_path.read_text())

        assert bible["exp_id"]    == "smoke_book/exp_001"
        assert bible["book_slug"] == "smoke_book"
        assert bible["seed"]      == 424242
        assert bible["parent_exp"] is None

        stages = bible["stages"]
        for required in ("script", "cast", "locations", "lookbook",
                         "references", "storyboard", "music", "narration",
                         "frames", "clips", "edl", "sfx", "final"):
            assert required in stages, f"missing stage: {required}"

        assert stages["script"]["n_scenes"]              == 1
        assert stages["script"]["n_characters"]          == 1
        assert stages["script"]["n_narration_elements"]  == 1
        assert stages["lookbook"]["style_frame"]         == "lookbook/style_frame.png"
        assert "scene_001:jeff"                          in stages["references"]["by_scene_char"]
        assert "scene_001"                               in stages["music"]["by_scene"]
        assert "scene_001"                               in stages["narration"]["by_scene"]
        assert "scene_001:shot_001"                      in stages["frames"]["by_scene_shot"]
        assert "scene_001:shot_001:take_0"               in stages["clips"]["by_scene_shot_take"]
        assert "scene_001:shot_001:take_1"               in stages["clips"]["by_scene_shot_take"]
        assert stages["final"]["size_bytes"]              > 0


@test("production_bible_json: includes config from snapshotted produce.py")
def t_pb_config() -> None:
    from prepare import Experiment
    from production_bible import build_production_bible_json

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)

        exp = Experiment(exp_id="exp_001", root=exp_root)
        out_path = build_production_bible_json(exp)
        bible = json.loads(out_path.read_text())

        # Config should include at least these key constants from produce.py.
        cfg = bible.get("config", {})
        for k in ("TAKES_PER_SHOT", "MAX_WORKERS"):
            assert k in cfg, f"config missing {k}: {list(cfg.keys())}"


@test("production_bible_json: includes metric and carryover when present")
def t_pb_metric_and_carryover() -> None:
    from prepare import Experiment
    from production_bible import build_production_bible_json

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)

        # Add metric.json + carryover.json
        metric = _make_metric([{
            "axis": "color", "priority": "high",
            "target": "LOOKBOOK_GRADE",
            "current_behavior": "crushed shadows",
            "suggested_change": "lift shadows",
            "expected_impact":  "more detail",
        }])
        (exp_root / "metric.json").write_text(json.dumps(metric))
        (exp_root / "carryover.json").write_text(json.dumps({"regen_lookbook": True}))
        (exp_root / "parent_exp.txt").write_text("exp_000")

        exp = Experiment(exp_id="exp_001", root=exp_root)
        out_path = build_production_bible_json(exp)
        bible = json.loads(out_path.read_text())

        assert bible["parent_exp"] == "smoke_book/exp_000"
        assert bible["metric"]["film_loss"] == 0.42
        assert bible["carryover_from_parent"]["regen_lookbook"] is True


# ── Tests: carryover planner ──────────────────────────────────────────────────

@test("carryover: lookbook keyword cascades to refs/frames/clips")
def t_carryover_lookbook_cascades() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "color", "priority": "high",
        "target": "LOOKBOOK_GRADE",
        "current_behavior": "crushed shadows",
        "suggested_change": "lift shadows in the color grade",
        "expected_impact":  "more detail",
    }])

    plan = plan_carryover(metric, priority_threshold="medium")

    assert plan["regen_lookbook"]    is True
    assert plan["regen_references"]  == "all"
    assert plan["regen_frames"]      == "all"
    assert plan["regen_clips"]       == "all"
    assert len(plan["applied_changes"]) == 1


@test("carryover: script keyword cascades everything")
def t_carryover_script_cascades() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "fidelity", "priority": "high",
        "target": "regenerate the script",
        "current_behavior": "scenes are out of order",
        "suggested_change": "rewrite the screenplay structure",
        "expected_impact":  "fix narrative",
    }])

    plan = plan_carryover(metric)

    assert plan["regen_script"]      is True
    assert plan["regen_lookbook"]    is True
    assert plan["regen_storyboard"]  is True
    assert plan["regen_references"]  == "all"
    assert plan["regen_clips"]       == "all"


@test("carryover: storyboard keyword regens shot list + frames + clips + edl")
def t_carryover_storyboard() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "cinematography", "priority": "high",
        "target": "shot_list_for_scene",
        "current_behavior": "Coverage too tight",
        "suggested_change": "Add wider shots, expand the shot list",
        "expected_impact":  "Better spatial clarity",
    }])

    plan = plan_carryover(metric)

    assert plan["regen_storyboard"] is True
    assert plan["regen_frames"]     == "all"
    assert plan["regen_clips"]      == "all"
    assert plan["regen_edl"]        is True


@test("carryover: narrow shot regen with scene+shot ID in target")
def t_carryover_narrow_shot() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "cinematography", "priority": "high",
        "target": "first frame composition for scene_002 shot_001",
        "current_behavior": "Subject off-center",
        "suggested_change": "Recompose first frame for scene_002 shot_001",
        "expected_impact":  "Better composition",
    }])

    plan = plan_carryover(metric)

    # Should target only that one shot, not all frames.
    assert plan["regen_frames"] != "all"
    assert ["scene_002", "shot_001"] in plan["regen_frames"]
    assert plan["regen_lookbook"] is False
    assert plan["regen_storyboard"] is False


@test("carryover: music change scoped to a specific scene")
def t_carryover_music_narrow() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "sound", "priority": "high",
        "target": "music for scene_003",
        "current_behavior": "wrong tone",
        "suggested_change": "regenerate music cue for scene_003 with darker mood",
        "expected_impact":  "fits scene better",
    }])

    plan = plan_carryover(metric)

    assert plan["regen_music"] != "all"
    assert ["scene_003"] in plan["regen_music"]


@test("carryover: low-priority change is skipped at default threshold")
def t_carryover_priority_filter() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "sound", "priority": "low",
        "target": "MUSIC_STYLE",
        "current_behavior": "generic",
        "suggested_change": "more period-appropriate music",
        "expected_impact":  "minor mood improvement",
    }])

    plan = plan_carryover(metric, priority_threshold="medium")

    assert len(plan["skipped_changes"]) == 1
    assert plan["regen_music"] == []


@test("carryover: low-priority change is applied at low threshold")
def t_carryover_priority_low_threshold() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "sound", "priority": "low",
        "target": "MUSIC_STYLE",
        "current_behavior": "generic",
        "suggested_change": "more period-appropriate music",
        "expected_impact":  "minor mood improvement",
    }])

    plan = plan_carryover(metric, priority_threshold="low")

    assert len(plan["skipped_changes"]) == 0
    assert plan["regen_music"] == "all"


@test("carryover: vague change goes to manual_review")
def t_carryover_manual_review() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "acting", "priority": "high",
        "target": "overall vibe",
        "current_behavior": "feels off",
        "suggested_change": "make it more atmospheric somehow",
        "expected_impact":  "?",
    }])

    plan = plan_carryover(metric)

    # No keyword matched → routed to manual_review
    assert len(plan["manual_review"]) == 1
    assert plan["regen_lookbook"] is False
    assert plan["regen_clips"] == []


@test("carryover: narration change recognized")
def t_carryover_narration() -> None:
    from carryover import plan_carryover

    metric = _make_metric([{
        "axis": "sound", "priority": "high",
        "target": "narration for scene_001",
        "current_behavior": "voice-over too monotone",
        "suggested_change": "regenerate the narrator V.O. with warmer tone",
        "expected_impact":  "more engaging",
    }])

    plan = plan_carryover(metric)

    assert plan["regen_narration"] != "all"
    assert ["scene_001"] in plan["regen_narration"]


# ── Tests: Experiment.new_iteration ──────────────────────────────────────────

@test("new_iteration: empty plan inherits everything")
def t_iter_empty_plan() -> None:
    from prepare import Experiment

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)
        prev = Experiment(exp_id="exp_001", root=exp_root)

        new = Experiment.new_iteration(prev, carryover={})

        # All artifacts copied forward.
        assert (new.root / "script.json").exists()
        assert (new.root / "lookbook.json").exists()
        assert (new.root / "lookbook" / "style_frame.png").exists()
        assert (new.root / "references" / "jeff" / "scene_001.png").exists()
        assert (new.root / "frames" / "scene_001" / "shot_001.png").exists()
        assert (new.root / "clips" / "scene_001" / "shot_001" / "take_0.mp4").exists()
        assert (new.root / "music" / "scene_001.wav").exists()
        assert (new.root / "narration" / "scene_001.mp3").exists()

        # Final/metric not copied (would skew the next critic run).
        assert not (new.root / "final.mp4").exists()

        # Parent linkage stamped.
        assert new.parent_exp_id == "exp_001"
        assert new.seed == prev.seed
        assert (new.root / "carryover.json").exists()

        # Seed inheritance.
        assert new.seed == 424242


@test("new_iteration: regen_lookbook cascades references/frames/clips")
def t_iter_lookbook_cascade() -> None:
    from prepare import Experiment

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)
        prev = Experiment(exp_id="exp_001", root=exp_root)

        new = Experiment.new_iteration(prev, carryover={"regen_lookbook": True})

        # Lookbook NOT copied
        assert not (new.root / "lookbook.json").exists()
        assert not (new.root / "lookbook").exists()

        # References + frames + clips also NOT copied (cascade)
        assert not (new.root / "references" / "jeff" / "scene_001.png").exists()
        assert not (new.root / "frames" / "scene_001" / "shot_001.png").exists()
        assert not (new.root / "clips" / "scene_001" / "shot_001" / "take_0.mp4").exists()

        # But upstream stages (script/cast) still copied.
        assert (new.root / "script.json").exists()
        assert (new.root / "cast.json").exists()


@test("new_iteration: narrow regen keeps untouched siblings")
def t_iter_narrow_regen() -> None:
    from prepare import Experiment

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root, scene_id="scene_001", shot_id="shot_001")

        # Add a second shot in scene_001 + a second scene's frame so we
        # can verify only the targeted shot is invalidated.
        (exp_root / "frames" / "scene_001" / "shot_002.png").write_bytes(_tiny_png())
        (exp_root / "frames" / "scene_002").mkdir(parents=True, exist_ok=True)
        (exp_root / "frames" / "scene_002" / "shot_001.png").write_bytes(_tiny_png())
        (exp_root / "clips" / "scene_001" / "shot_002").mkdir(parents=True, exist_ok=True)
        (exp_root / "clips" / "scene_001" / "shot_002" / "take_0.mp4").write_bytes(_tiny_mp4())

        prev = Experiment(exp_id="exp_001", root=exp_root)

        new = Experiment.new_iteration(prev, carryover={
            "regen_frames": [["scene_001", "shot_001"]],
            "regen_clips":  [["scene_001", "shot_001"]],
        })

        # The targeted shot is gone.
        assert not (new.root / "frames" / "scene_001" / "shot_001.png").exists()
        assert not (new.root / "clips" / "scene_001" / "shot_001").exists()

        # Untouched siblings preserved.
        assert (new.root / "frames" / "scene_001" / "shot_002.png").exists()
        assert (new.root / "frames" / "scene_002" / "shot_001.png").exists()
        assert (new.root / "clips" / "scene_001" / "shot_002" / "take_0.mp4").exists()


@test("new_iteration: regen_music='all' skips entire music dir")
def t_iter_music_all() -> None:
    from prepare import Experiment

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)
        # Add a second scene's music
        (exp_root / "music" / "scene_002.wav").write_bytes(_tiny_wav())
        prev = Experiment(exp_id="exp_001", root=exp_root)

        new = Experiment.new_iteration(prev, carryover={"regen_music": "all"})

        # No music files copied
        assert not (new.root / "music" / "scene_001.wav").exists()
        assert not (new.root / "music" / "scene_002.wav").exists()

        # But narration and clips still there
        assert (new.root / "narration" / "scene_001.mp3").exists()
        assert (new.root / "clips" / "scene_001" / "shot_001" / "take_0.mp4").exists()


@test("new_iteration: regen_script cascades to full pipeline")
def t_iter_script_cascade() -> None:
    from prepare import Experiment

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)
        prev = Experiment(exp_id="exp_001", root=exp_root)

        new = Experiment.new_iteration(prev, carryover={"regen_script": True})

        # Almost nothing should remain: no script, no cast, no lookbook,
        # no refs, no frames, no clips, no music.
        for relpath in ("script.json", "cast.json", "locations.json",
                        "lookbook.json",
                        "references/jeff/scene_001.png",
                        "frames/scene_001/shot_001.png",
                        "clips/scene_001/shot_001/take_0.mp4",
                        "music/scene_001.wav",
                        "narration/scene_001.mp3",
                        "edl.json"):
            assert not (new.root / relpath).exists(), \
                f"regen_script cascade should have invalidated {relpath}"

        # But scaffolding (book, seed, parent, snapshot) still set.
        assert (new.root / "book.txt").exists()
        assert (new.root / "seed.txt").exists()
        assert (new.root / "parent_exp.txt").exists()
        assert new.seed == prev.seed


# ── Tests: end-to-end carryover → new_iteration ──────────────────────────────

@test("end_to_end: critic recommendation → plan → new exp dir")
def t_end_to_end() -> None:
    from prepare import Experiment
    from carryover import plan_carryover

    with _IsolatedExperimentsDir() as exp_dir:
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        _build_fixture(exp_root)

        metric = _make_metric([
            # Should narrow regen to one shot's frame + clips.
            {
                "axis": "cinematography", "priority": "high",
                "target": "first frame composition for scene_001 shot_001",
                "current_behavior": "Subject off-center",
                "suggested_change": "Recompose first frame for scene_001 shot_001",
                "expected_impact":  "Better composition",
            },
            # Should regen narration for scene_001.
            {
                "axis": "sound", "priority": "high",
                "target": "narration for scene_001",
                "current_behavior": "monotone",
                "suggested_change": "regenerate the narrator V.O. with warmer tone",
                "expected_impact":  "more engaging",
            },
        ])
        (exp_root / "metric.json").write_text(json.dumps(metric))

        prev = Experiment(exp_id="exp_001", root=exp_root)
        plan = plan_carryover(metric)
        new  = Experiment.new_iteration(prev, carryover=plan)

        # The targeted frame + narration are GONE
        assert not (new.root / "frames" / "scene_001" / "shot_001.png").exists()
        assert not (new.root / "narration" / "scene_001.mp3").exists()

        # Everything else preserved.
        assert (new.root / "lookbook.json").exists()
        assert (new.root / "references" / "jeff" / "scene_001.png").exists()
        assert (new.root / "music" / "scene_001.wav").exists()
        assert (new.root / "script.json").exists()


# ── Tests: CLI smoke ──────────────────────────────────────────────────────────

@test("CLI: api_smoke_test --list runs without errors")
def t_cli_api_list() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/api_smoke_test.py", "--list"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr[:300]}"
    out = proc.stdout
    # Sanity: every category should appear with at least one endpoint.
    for header in ("Image endpoints:", "Video endpoints:", "Audio endpoints:"):
        assert header in out, f"missing header: {header}"
    for endpoint in ("gpt_image", "gen4_image", "stable_audio", "runway_tts",
                     "google_veo", "ltx-2-3-pro"):
        assert endpoint in out, f"missing endpoint: {endpoint}"


@test("CLI: run_loop --help runs without errors")
def t_cli_run_loop_help() -> None:
    proc = subprocess.run(
        [sys.executable, "run_loop.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr[:300]}"
    for flag in ("--iterations", "--target", "--threshold",
                 "--plateau", "--resume", "--history"):
        assert flag in proc.stdout, f"missing flag: {flag}"


@test("CLI: production_bible --help runs without errors")
def t_cli_pb_help() -> None:
    proc = subprocess.run(
        [sys.executable, "production_bible.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr[:300]}"


@test("CLI: carryover --help runs without errors")
def t_cli_carryover_help() -> None:
    proc = subprocess.run(
        [sys.executable, "carryover.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr[:300]}"


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    t_pb_all_stages,
    t_pb_config,
    t_pb_metric_and_carryover,
    t_carryover_lookbook_cascades,
    t_carryover_script_cascades,
    t_carryover_storyboard,
    t_carryover_narrow_shot,
    t_carryover_music_narrow,
    t_carryover_priority_filter,
    t_carryover_priority_low_threshold,
    t_carryover_manual_review,
    t_carryover_narration,
    t_iter_empty_plan,
    t_iter_lookbook_cascade,
    t_iter_narrow_regen,
    t_iter_music_all,
    t_iter_script_cascade,
    t_end_to_end,
    t_cli_api_list,
    t_cli_run_loop_help,
    t_cli_pb_help,
    t_cli_carryover_help,
]


def main() -> int:
    global _verbose, _fail_fast
    parser = argparse.ArgumentParser(
        description="Smoke test for the autoresearch run loop infrastructure."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    _verbose   = args.verbose
    _fail_fast = args.fail_fast

    print("Running run-loop smoke tests (no API calls)...")

    for fn in ALL_TESTS:
        fn()

    # Summary
    print(f"\n{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    for r in _results:
        print(r)
    print(f"{'─' * 60}")
    n_pass = sum(1 for r in _results if r.status == "PASS")
    n_fail = sum(1 for r in _results if r.status == "FAIL")
    print(f"  {n_pass} PASS  ·  {n_fail} FAIL  ·  {len(_results)} total")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
