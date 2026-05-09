"""production_bible.py — write a machine-readable production bible JSON.

The PDF bible (bible.py) is for humans reading a binder. THIS file is the
machine version: every artifact's path, every config value, every prompt,
every model used. The autoresearch loop (run_loop.py) reads it during the
next iteration's planning step to know what was done and what to change.

Layout:
{
  "exp_id": "jurassic_park/exp_002",
  "book_slug": "jurassic_park",
  "parent_exp": "jurassic_park/exp_001",
  "seed": 12345,
  "timestamp_iso": "2026-05-09T19:30:00",
  "stages": {
    "script":        {"path": "script.json", "n_scenes": 8, "n_characters": 5},
    "cast":          {"path": "cast.json", "n_cast": 5},
    "locations":     {"path": "locations.json", "n_locations": 4,
                       "moodboards": {"forest": "location_moodboards/forest/00.png"}},
    "lookbook":      {"path": "lookbook.json",
                       "style_frame": "lookbook/style_frame.png",
                       "grade": "...", "style_keywords": [...]},
    "references":    {"by_scene_char": {"scene_001:jeff": "references/jeff/scene_001.png"}},
    "storyboard":    {"path": "storyboard.json", "n_shots": 24},
    "music":         {"by_scene": {"scene_001": "music/scene_001.wav"}},
    "narration":     {"by_scene": {"scene_001": "narration/scene_001.mp3"}},
    "frames":        {"by_scene_shot": {"scene_001:shot_001": "frames/scene_001/shot_001.png"}},
    "clips":         {"by_scene_shot_take": {
                         "scene_001:shot_001:take_0": "clips/scene_001/shot_001/take_0.mp4"
                       }},
    "edl":           {"path": "edl.json"},
    "sfx":           {"by_scene": {"scene_001": "sfx/scene_001/ambient.wav"}},
    "final":         {"path": "final.mp4"}
  },
  "config": {
    "DIRECTOR": "...",
    "CINEMATOGRAPHER": "...",
    "MUSIC_STYLE": "...",
    "TAKES_PER_SHOT": 2,
    "SHOT_DURATION_SECONDS": 8,
    "VIDEO_BACKEND": "seedance",
    "MAX_WORKERS": 2,
    ...
  },
  "metric": { ... full contents of metric.json if present ... },
  "carryover_from_parent": { ... carryover.json if present ... }
}

Run as:
    python production_bible.py exp_001
    python production_bible.py latest
    python production_bible.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare import EXPERIMENTS_DIR, Experiment, iter_all_experiments


def _list_files(d: Path, suffix_set: set[str]) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in suffix_set]


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def build_production_bible_json(exp: Experiment) -> Path:
    """Write production_bible.json to the experiment dir. Returns its path."""
    root = exp.root
    bible: dict[str, Any] = {
        "exp_id":        f"{exp.book_slug}/{exp.exp_id}",
        "book_slug":     exp.book_slug,
        "parent_exp":    (
            f"{exp.book_slug}/{exp.parent_exp_id}"
            if exp.parent_exp_id else None
        ),
        "seed":          exp.seed,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages":        {},
        "config":        {},
    }

    # ── Stage 1: script ─────────────────────────────────────────────────
    if exp.has("script.json"):
        script = exp.read_json("script.json")
        bible["stages"]["script"] = {
            "path":          "script.json",
            "title":         script.get("title"),
            "n_scenes":      len(script.get("scenes", [])),
            "n_characters":  len(script.get("characters", [])),
            "n_elements": sum(
                len(s.get("elements") or []) for s in script.get("scenes", [])
            ),
            "n_narration_elements": sum(
                1
                for s in script.get("scenes", [])
                for e in (s.get("elements") or [])
                if e.get("type") == "narration"
            ),
        }

    # ── Stage 2: cast & locations ──────────────────────────────────────
    if exp.has("cast.json"):
        cast = exp.read_json("cast.json")
        bible["stages"]["cast"] = {
            "path":     "cast.json",
            "n_cast":   len(cast),
            "by_id":    {c["id"]: c.get("name") for c in cast},
        }
    if exp.has("locations.json"):
        locations = exp.read_json("locations.json")
        moodboards: dict[str, str] = {}
        for loc in locations:
            slug = loc.get("slug")
            mb_paths = loc.get("moodboard_paths") or []
            if slug and mb_paths:
                # Store the first (and currently only) moodboard path,
                # relative to the exp root.
                p = Path(mb_paths[0])
                if p.exists():
                    moodboards[slug] = _rel(p, root)
        bible["stages"]["locations"] = {
            "path":         "locations.json",
            "n_locations":  len(locations),
            "moodboards":   moodboards,
        }

    # ── Stage 3: lookbook ──────────────────────────────────────────────
    if exp.has("lookbook.json"):
        lb = exp.read_json("lookbook.json")
        sf = root / "lookbook" / "style_frame.png"
        bible["stages"]["lookbook"] = {
            "path":            "lookbook.json",
            "style_frame":     "lookbook/style_frame.png" if sf.exists() else None,
            "grade":           lb.get("ffmpeg_grade"),
            "style_keywords":  lb.get("style_keywords", []),
            "director":        lb.get("director"),
            "cinematographer": lb.get("cinematographer"),
        }

    # ── Stage 4: references ────────────────────────────────────────────
    refs_by: dict[str, str] = {}
    refs_dir = root / "references"
    if refs_dir.exists():
        for png in refs_dir.rglob("*.png"):
            # path = references/{char_id}/{scene_id}.png
            parts = png.relative_to(refs_dir).parts
            if len(parts) == 2:
                char_id, scene_file = parts
                scene_id = Path(scene_file).stem
                key = f"{scene_id}:{char_id}"
                refs_by[key] = _rel(png, root)
    if refs_by:
        bible["stages"]["references"] = {
            "by_scene_char": refs_by,
            "n":             len(refs_by),
        }

    # ── Stage 5: storyboard ────────────────────────────────────────────
    if exp.has("storyboard.json"):
        storyboard = exp.read_json("storyboard.json")
        n_shots = sum(len(v) for v in storyboard.values())
        bible["stages"]["storyboard"] = {
            "path":          "storyboard.json",
            "n_shots":       n_shots,
            "shots_by_scene": {sid: len(v) for sid, v in storyboard.items()},
        }

    # ── Stage 6: music ─────────────────────────────────────────────────
    music_dir = root / "music"
    if music_dir.exists():
        music_by: dict[str, str] = {
            p.stem: _rel(p, root)
            for p in sorted(music_dir.glob("*.wav"))
        }
        if music_by:
            bible["stages"]["music"] = {
                "by_scene": music_by,
                "n":        len(music_by),
            }

    # ── Stage 6.5: narration ───────────────────────────────────────────
    narr_dir = root / "narration"
    if narr_dir.exists():
        narr_by: dict[str, str] = {}
        for p in sorted(list(narr_dir.glob("*.mp3")) + list(narr_dir.glob("*.wav"))):
            narr_by[p.stem] = _rel(p, root)
        if narr_by:
            bible["stages"]["narration"] = {
                "by_scene": narr_by,
                "n":        len(narr_by),
            }

    # ── Stage 7: first frames ──────────────────────────────────────────
    frames_dir = root / "frames"
    if frames_dir.exists():
        frames_by: dict[str, str] = {}
        for png in frames_dir.rglob("*.png"):
            parts = png.relative_to(frames_dir).parts
            if len(parts) == 2:
                scene_id, shot_file = parts
                shot_id = Path(shot_file).stem
                frames_by[f"{scene_id}:{shot_id}"] = _rel(png, root)
        if frames_by:
            bible["stages"]["frames"] = {
                "by_scene_shot": frames_by,
                "n":             len(frames_by),
            }

    # ── Stage 8: clips ─────────────────────────────────────────────────
    clips_dir = root / "clips"
    if clips_dir.exists():
        clips_by: dict[str, str] = {}
        for mp4 in clips_dir.rglob("*.mp4"):
            parts = mp4.relative_to(clips_dir).parts
            if len(parts) == 3:
                scene_id, shot_id, take_file = parts
                take_id = Path(take_file).stem  # e.g. take_0
                clips_by[f"{scene_id}:{shot_id}:{take_id}"] = _rel(mp4, root)
        if clips_by:
            bible["stages"]["clips"] = {
                "by_scene_shot_take": clips_by,
                "n":                  len(clips_by),
            }

    # ── Stage 9: edl ───────────────────────────────────────────────────
    if exp.has("edl.json"):
        edl = exp.read_json("edl.json")
        bible["stages"]["edl"] = {
            "path":   "edl.json",
            "n_picks": (
                len(edl) if isinstance(edl, list)
                else sum(len(v) for v in edl.values()) if isinstance(edl, dict)
                else None
            ),
        }

    # ── SFX ─────────────────────────────────────────────────────────────
    sfx_dir = root / "sfx"
    if sfx_dir.exists():
        sfx_by: dict[str, str] = {}
        for wav in sorted(sfx_dir.rglob("*.wav")):
            parts = wav.relative_to(sfx_dir).parts
            if parts:
                sfx_by[parts[0]] = _rel(wav, root)
        if sfx_by:
            bible["stages"]["sfx"] = {"by_scene": sfx_by, "n": len(sfx_by)}

    # ── Stage 10: final ────────────────────────────────────────────────
    final = root / "final.mp4"
    if final.exists():
        bible["stages"]["final"] = {
            "path":           "final.mp4",
            "size_bytes":     final.stat().st_size,
        }

    # ── Captured config (snapshotted from produce.py constants + runtime) ──
    # Read from this experiment's snapshotted produce.py for produce.py-defined
    # values, and from the prepare module for prepare.py-defined values
    # (which are shared, version-controlled, and never agent-modified).
    cfg: dict[str, Any] = {}
    snapshot = root / "produce.py"
    if snapshot.exists():
        src = snapshot.read_text()
        # produce.py-level constants (top-level assignments).
        # Match plain `KEY = ...`, `KEY=...`, and type-annotated `KEY: type = ...`.
        for key in ("DIRECTOR", "CINEMATOGRAPHER", "MUSIC_STYLE",
                    "LOOKBOOK_STYLE_KEYWORDS", "LOOKBOOK_GRADE",
                    "VIDEO_BACKEND", "IMAGE_BACKEND", "MAX_WORKERS"):
            for line in src.splitlines():
                stripped = line.strip()
                if (stripped.startswith(f"{key} ")
                        or stripped.startswith(f"{key}=")
                        or stripped.startswith(f"{key}:")):
                    cfg[key] = stripped[:240]
                    break
    # prepare.py-defined runtime constants — these are constants used by the
    # pipeline that aren't redefined per-experiment, so reading the live
    # module is correct.
    try:
        import prepare as _prepare
        for key in ("TAKES_PER_SHOT", "SHOT_DURATION_SECONDS", "VEO_TIER"):
            if hasattr(_prepare, key):
                cfg[key] = getattr(_prepare, key)
    except Exception:  # noqa: BLE001
        pass
    bible["config"] = cfg

    # ── Metric (if evaluated) ──────────────────────────────────────────
    if exp.has("metric.json"):
        bible["metric"] = exp.read_json("metric.json")

    # ── Carryover info (if this exp was forked from a parent) ─────────
    if exp.has("carryover.json"):
        bible["carryover_from_parent"] = exp.read_json("carryover.json")

    out = exp.write_json("production_bible.json", bible)
    return out


def _print_summary(p: Path) -> None:
    data = json.loads(p.read_text())
    print(f"\n=== {data['exp_id']} ===")
    print(f"  parent:  {data.get('parent_exp')}")
    print(f"  seed:    {data['seed']}")
    print(f"  stages:")
    for name, info in data.get("stages", {}).items():
        n = info.get("n", info.get("n_shots", info.get("n_scenes", "")))
        n_str = f"  ({n})" if n != "" else ""
        print(f"    {name}{n_str}")
    if "metric" in data:
        loss = data["metric"].get("film_loss")
        print(f"  film_loss: {loss}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build production_bible.json")
    ap.add_argument("exp_id", nargs="?", help="exp_NNN, book/exp_NNN, or 'latest'")
    ap.add_argument("--all", action="store_true", help="Build for all completed experiments")
    args = ap.parse_args()

    if args.all:
        for exp_path in iter_all_experiments():
            try:
                exp = Experiment(exp_id=exp_path.name, root=exp_path)
                p = build_production_bible_json(exp)
                _print_summary(p)
            except Exception as e:  # noqa: BLE001
                print(f"  {exp_path.name}: failed — {e}")
        return

    if not args.exp_id:
        ap.print_help()
        sys.exit(1)

    if args.exp_id == "latest":
        exp = Experiment.latest()
    else:
        exp = Experiment.load(args.exp_id)

    p = build_production_bible_json(exp)
    _print_summary(p)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
