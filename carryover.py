"""carryover.py — translate critic recommendations into a carryover plan.

The critic returns a `changes` array via the CRITIC_TOOL_SCHEMA; each entry
has `axis`, `priority`, `target`, `current_behavior`, `suggested_change`,
and `expected_impact`. This module reads those recommendations and decides
which artifacts the next experiment should regenerate vs inherit from the
parent.

The plan is a dict matching the schema documented on
``Experiment.new_iteration``:

  {
    "regen_script":      bool,
    "regen_cast":        bool,
    "regen_lookbook":    bool,
    "regen_references":  list[[scene_id, char_id]] | "all",
    "regen_storyboard":  bool,
    "regen_music":       list[scene_id] | "all",
    "regen_narration":   list[scene_id] | "all",
    "regen_frames":      list[[scene_id, shot_id]] | "all",
    "regen_clips":       list[[scene_id, shot_id]] | "all",
    "regen_edl":         bool,
    "applied_changes":   list[dict],   # which changes triggered which flags
  }

Heuristics — keyword-based mapping from change.target / change.suggested_change:

  Mentions "script", "screenplay", "narrative", "scene structure", "fidelity"
    → regen_script (cascades everything).

  Mentions "lookbook", "grade", "color palette", "style frame",
  "color treatment", "LUT", "LOOKBOOK_GRADE", "style keywords"
    → regen_lookbook (cascades references + frames + clips).

  Mentions "casting", "actor", "cast", "replace casting"
    → regen_cast.

  Mentions "moodboard", "location", "set design"
    → regen_cast (locations are part of the cast stage).

  Mentions "shot list", "storyboard", "shot composition", "coverage",
  "shot_list_for_scene"
    → regen_storyboard (cascades frames+clips for the whole show).

  Mentions "music", "score", "MUSIC_STYLE"
    → regen_music = "all" (could be narrowed to specific scenes if cited).

  Mentions "narration", "voice over", "VO", "voice-over"
    → regen_narration = "all".

  Mentions "edit", "EDL", "pacing", "cut", "TAKES_PER_SHOT"
    → regen_edl + (clip-level) regen_clips when takes/coverage cited.

  Cites a specific scene_id or shot_id in target / suggested_change
    → narrow to that scene / shot.

  Otherwise — falls back to axis-based routing using the critic's
  `axis` field (cinematography / color / sound / acting / continuity /
  fidelity). Every change drives a concrete pipeline action; nothing
  is ever flagged for manual review. The autoresearch loop is fully
  automated.

Run as a module:
    from carryover import plan_carryover
    plan = plan_carryover(metric_dict)
"""
from __future__ import annotations

import json
import re
from typing import Any


# ── Keyword sets (lowercased; matched against the concatenated change text)

_SCRIPT_KW = (
    "screenplay", "script structure", "narrative", "scene order",
    "scene structure", "story beats", "fidelity to source",
    "regenerate the script", "rewrite the script",
)
_LOOKBOOK_KW = (
    "lookbook", "look book", "ffmpeg_grade", "lookbook_grade",
    "color grade", "color palette", "color treatment", "lut",
    "style frame", "style_frame", "style keywords",
    "lookbook_style_keywords",
)
_CAST_KW = (
    "casting", "recast", "replace casting", "actor choice",
    "different actor", "actor for",
)
_LOC_KW = (
    "moodboard", "location reference", "set design", "production design",
    "location dressing",
)
_STORY_KW = (
    "shot list", "storyboard", "coverage", "shot count",
    "shot_list_for_scene", "more coverage", "wider coverage",
    "tighter shot list",
)
_FRAME_KW = (
    "first frame", "frame composition", "framing of shot",
    "composition of frame",
)
_CLIP_KW = (
    "take", "takes_per_shot", "performance", "camera move",
    "regenerate clip", "rerun shot", "veo_prompt",
)
_MUSIC_KW = (
    "music", "score", "soundtrack", "music_style", "music cue",
)
_NARR_KW = (
    "narration", "narrator", "voice over", "voice-over", " v.o.",
    "voiceover",
)
_EDL_KW = (
    "edl", "edit decisions", "cut points", "pacing", "rhythm",
    "edit timing",
)


_SCENE_ID_RE = re.compile(r"\b(scene[_-]?\d{1,3})\b", re.IGNORECASE)
_SHOT_ID_RE  = re.compile(r"\b(shot[_-]?\d{1,3})\b", re.IGNORECASE)
_CHAR_HINT_RE = re.compile(r"\bcharacter[_-]?id[\s:=]+([a-z][a-z0-9_-]*)", re.IGNORECASE)


def _change_text(change: dict) -> str:
    """Concatenated lowercased text of a change for keyword matching."""
    return " ".join(
        str(change.get(k, ""))
        for k in ("target", "current_behavior", "suggested_change",
                  "expected_impact", "axis")
    ).lower()


def _scene_ids_in(text: str) -> list[str]:
    return sorted({m.group(1).lower().replace("-", "_") for m in _SCENE_ID_RE.finditer(text)})


def _shot_ids_in(text: str) -> list[str]:
    return sorted({m.group(1).lower().replace("-", "_") for m in _SHOT_ID_RE.finditer(text)})


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def plan_carryover(
    metric: dict[str, Any],
    *,
    priority_threshold: str = "medium",
) -> dict[str, Any]:
    """Build a carryover plan from the critic's metric.json contents.

    ``priority_threshold`` filters out changes below the given priority.
    Allowed values: "low" (apply all), "medium" (default — apply medium+high),
    "high" (apply only high-priority).
    """
    priority_rank = {"low": 0, "medium": 1, "high": 2}
    threshold_rank = priority_rank.get(priority_threshold, 1)

    changes = metric.get("changes") or []

    plan: dict[str, Any] = {
        "regen_script":      False,
        "regen_cast":        False,
        "regen_lookbook":    False,
        "regen_references":  [],
        "regen_storyboard":  False,
        "regen_music":       [],
        "regen_narration":   [],
        "regen_frames":      [],
        "regen_clips":       [],
        "regen_edl":         False,
        "applied_changes":   [],
        "skipped_changes":   [],
        "manual_review":     [],
    }

    # Helpers to merge into list-valued fields, promoting to "all" when
    # appropriate.
    def _add_scenes(field: str, scenes: list[str]) -> None:
        cur = plan[field]
        if cur == "all":
            return
        if not scenes:
            plan[field] = "all"
            return
        existing = {tuple([s]) if isinstance(s, str) else tuple(s) for s in cur}
        for s in scenes:
            existing.add((s,))
        plan[field] = [list(t) for t in sorted(existing)]

    def _add_shots(field: str, pairs: list[tuple[str, str]]) -> None:
        cur = plan[field]
        if cur == "all":
            return
        if not pairs:
            plan[field] = "all"
            return
        existing = {tuple(x) for x in cur}
        for p in pairs:
            existing.add(p)
        plan[field] = [list(t) for t in sorted(existing)]

    def _add_refs(scene_ids: list[str], char_id: str | None) -> None:
        cur = plan["regen_references"]
        if cur == "all":
            return
        if not scene_ids and not char_id:
            return
        existing = {tuple(x) for x in cur}
        for s in (scene_ids or [""]):
            existing.add((s, char_id or ""))
        plan["regen_references"] = [list(t) for t in sorted(existing)]

    for change in changes:
        priority = (change.get("priority") or "medium").lower()
        if priority_rank.get(priority, 1) < threshold_rank:
            plan["skipped_changes"].append({
                "change":  change,
                "reason":  f"priority {priority} below threshold {priority_threshold}",
            })
            continue

        text       = _change_text(change)
        scene_ids  = _scene_ids_in(text)
        shot_ids   = _shot_ids_in(text)
        char_match = _CHAR_HINT_RE.search(text)
        char_id    = char_match.group(1) if char_match else None

        triggered: list[str] = []

        # Highest-precedence cascades first.
        if _matches_any(text, _SCRIPT_KW):
            plan["regen_script"] = True
            triggered.append("regen_script (cascades)")

        elif _matches_any(text, _LOOKBOOK_KW):
            plan["regen_lookbook"] = True
            triggered.append("regen_lookbook (cascades refs+frames+clips)")

        elif _matches_any(text, _STORY_KW):
            plan["regen_storyboard"] = True
            # Storyboard regen invalidates frames+clips for whole show.
            plan["regen_frames"] = "all"
            plan["regen_clips"]  = "all"
            plan["regen_edl"]    = True
            triggered.append("regen_storyboard (+ frames/clips/edl)")

        elif _matches_any(text, _CAST_KW) or _matches_any(text, _LOC_KW):
            plan["regen_cast"] = True
            triggered.append("regen_cast")

        else:
            # Narrower targets — combine multiple if applicable.
            if _matches_any(text, _MUSIC_KW):
                _add_scenes("regen_music", scene_ids)
                triggered.append(
                    f"regen_music({'all' if not scene_ids else scene_ids})"
                )

            if _matches_any(text, _NARR_KW):
                _add_scenes("regen_narration", scene_ids)
                triggered.append(
                    f"regen_narration({'all' if not scene_ids else scene_ids})"
                )

            if _matches_any(text, _FRAME_KW):
                pairs = [(s, sh) for s in (scene_ids or [""]) for sh in (shot_ids or [""])]
                pairs = [p for p in pairs if all(p)]   # drop blanks
                _add_shots("regen_frames", pairs)
                triggered.append(
                    f"regen_frames({'all' if not pairs else len(pairs)})"
                )

            if _matches_any(text, _CLIP_KW):
                pairs = [(s, sh) for s in (scene_ids or [""]) for sh in (shot_ids or [""])]
                pairs = [p for p in pairs if all(p)]
                _add_shots("regen_clips", pairs)
                triggered.append(
                    f"regen_clips({'all' if not pairs else len(pairs)})"
                )

            if _matches_any(text, _EDL_KW):
                plan["regen_edl"] = True
                triggered.append("regen_edl")

            # Vague reference change?
            if "reference" in text and char_id:
                _add_refs(scene_ids, char_id)
                triggered.append(f"regen_references({char_id})")

        if triggered:
            plan["applied_changes"].append({
                "change":      change,
                "triggered":   triggered,
            })
        else:
            # Axis-based fallback: the critic always tags every change
            # with an axis (cinematography / color / sound / acting /
            # continuity / fidelity) — even when the suggestion text
            # is too vague to keyword-match. Use the axis to pick a
            # default invalidation surface. This guarantees every
            # change drives a concrete pipeline action — no human in
            # the loop.
            axis = (change.get("axis") or "").lower()
            axis_routes: list[str] = []
            if axis == "fidelity":
                plan["regen_script"] = True
                axis_routes.append("regen_script (cascades)")
            elif axis == "color":
                plan["regen_lookbook"] = True
                axis_routes.append("regen_lookbook (cascades refs+frames+clips)")
            elif axis == "cinematography":
                plan["regen_storyboard"] = True
                plan["regen_frames"] = "all"
                plan["regen_clips"]  = "all"
                plan["regen_edl"]    = True
                axis_routes.append("regen_storyboard (+ frames/clips/edl)")
            elif axis == "continuity":
                plan["regen_storyboard"] = True
                plan["regen_frames"] = "all"
                plan["regen_clips"]  = "all"
                axis_routes.append("regen_storyboard (+ frames/clips for continuity)")
            elif axis == "acting":
                # Re-render takes with the (now-updated) prompt direction.
                plan["regen_clips"] = "all"
                axis_routes.append("regen_clips (acting → new takes)")
            elif axis == "sound":
                plan["regen_music"]     = "all"
                plan["regen_narration"] = "all"
                axis_routes.append("regen_music + regen_narration")
            else:
                # Critic gave no axis or a novel one — last-resort default
                # is regen_lookbook, the most "atmospheric" surface that
                # cascades enough to produce a meaningfully different cut.
                plan["regen_lookbook"] = True
                axis_routes.append(f"regen_lookbook (no-axis fallback)")
            plan["applied_changes"].append({
                "change":    change,
                "triggered": axis_routes,
                "via":       "axis_fallback",
            })

    # Cascade rules — keep in sync with Experiment.new_iteration:
    #   regen_script    → cascades everything
    #   regen_lookbook  → cascades references + frames + clips
    # Apply here so the plan dict accurately reflects what will happen.
    if plan["regen_script"]:
        plan["regen_cast"]       = True
        plan["regen_lookbook"]   = True
        plan["regen_storyboard"] = True
        plan["regen_references"] = "all"
        plan["regen_music"]      = "all"
        plan["regen_narration"]  = "all"
        plan["regen_frames"]     = "all"
        plan["regen_clips"]      = "all"
        plan["regen_edl"]        = True
    if plan["regen_lookbook"]:
        plan["regen_references"] = "all"
        plan["regen_frames"]     = "all"
        plan["regen_clips"]      = "all"

    return plan


def plan_summary(plan: dict[str, Any]) -> str:
    """Human-readable one-line summary of a carryover plan."""
    bits: list[str] = []
    if plan["regen_script"]:
        return "REGEN ENTIRE SCRIPT (cascades everything)"
    if plan["regen_lookbook"]:
        bits.append("lookbook→cascade")
    if plan["regen_cast"]:
        bits.append("cast")
    if plan["regen_storyboard"]:
        bits.append("storyboard→cascade")
    refs = plan["regen_references"]
    if refs == "all":
        bits.append("refs:all")
    elif refs:
        bits.append(f"refs:{len(refs)}")
    music = plan["regen_music"]
    if music == "all":
        bits.append("music:all")
    elif music:
        bits.append(f"music:{len(music)}")
    narr = plan["regen_narration"]
    if narr == "all":
        bits.append("narration:all")
    elif narr:
        bits.append(f"narration:{len(narr)}")
    frames = plan["regen_frames"]
    if frames == "all":
        bits.append("frames:all")
    elif frames:
        bits.append(f"frames:{len(frames)}")
    clips = plan["regen_clips"]
    if clips == "all":
        bits.append("clips:all")
    elif clips:
        bits.append(f"clips:{len(clips)}")
    if plan["regen_edl"]:
        bits.append("edl")
    if not bits:
        # Every change was below the priority threshold and got
        # skipped — nothing left to regenerate. The next iteration
        # will produce the same artifacts as the parent.
        return "no regen needed (all changes below threshold)"
    return ", ".join(bits)


if __name__ == "__main__":
    import argparse
    from prepare import Experiment

    ap = argparse.ArgumentParser(description="Plan carryover from a metric.json")
    ap.add_argument("exp_id", help="exp_NNN, book/exp_NNN, or 'latest'")
    ap.add_argument(
        "--threshold", default="medium", choices=["low", "medium", "high"],
        help="Minimum priority to apply (default medium)",
    )
    args = ap.parse_args()

    exp = Experiment.latest() if args.exp_id == "latest" else Experiment.load(args.exp_id)
    if not exp.has("metric.json"):
        raise SystemExit(f"No metric.json for {exp.exp_id} — run evaluate.py first.")
    metric = exp.read_json("metric.json")
    plan = plan_carryover(metric, priority_threshold=args.threshold)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"\nSummary: {plan_summary(plan)}")
