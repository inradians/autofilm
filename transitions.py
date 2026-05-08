"""transitions.py — shot-level video transitions for autofilm.

This module is the backend for the optional `transition_out` field on
shots in the storyboard / EDL. It ships a curated catalog of named
transitions, a validator/normalizer, and a single renderer function that
takes an ordered list of clip paths plus an N-1 list of transitions and
produces one composed mp4.

## Implementation note: ffmpeg, not GLSL

ffmpeg's `xfade` filter ships ~50 named transitions covering everything
a short-film cut needs: cross-dissolve, fade-through-black, wipes,
slides, iris open/close, pixelize, smooth radial, etc. It runs headless
without a GPU. The matching `acrossfade` filter handles audio.

A Python-and-GLSL alternative (e.g. moderngl driving fragment shaders)
would add a heavy dep tree and only matters for transitions xfade can't
express. If you ever need something genuinely custom — a kaleidoscope
push, a glitch shatter — render the *transition itself* as a separate
Veo or Aleph shot inserted between the two adjacent shots, and use
`cut` on either side. That's cheaper and more controllable than building
a shader pipeline you'll iterate on twice.

## EDL field shape

Each shot in the storyboard / EDL may carry:

    "transition_out": {"type": "fade", "duration": 0.5}

The `type` is one of `TRANSITIONS` keys (default `"cut"`). The
`duration` is in seconds; capped at 2.0 to keep transitions from
overwhelming short clips. Missing field == hard cut == default.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


# ============================================================================
# Catalog
# ============================================================================
# Keys are the transition type names that appear in the storyboard / EDL
# JSON. Values map to the ffmpeg xfade transition param plus a short
# editorial description (used in prompts and surfaced in the bible).
#
# Curated subset of xfade's catalog — the ones with clear editorial
# meaning. Adding more is fine; just match an ffmpeg xfade transition
# name (`ffmpeg -h filter=xfade` on your machine for the full list).
TRANSITIONS: dict[str, dict[str, Any]] = {
    # Hard cut — special, no overlap, no xfade call needed.
    "cut":         {"xfade": None,         "desc": "Hard cut. No overlap. The default; use for almost everything."},

    # Cross-dissolves and fades — the universal editorial vocabulary.
    "fade":        {"xfade": "fade",         "desc": "Cross-dissolve. Continuity within a scene; passage of mood."},
    "fadeblack":   {"xfade": "fadeblack",    "desc": "Fade through black. End of a chapter / time jump / death."},
    "fadewhite":   {"xfade": "fadewhite",    "desc": "Fade through white. Flashback, revelation, dream entry."},
    "dissolve":    {"xfade": "dissolve",     "desc": "Pixel dissolve. Gentler than crossfade; texture change."},
    "fadegrays":   {"xfade": "fadegrays",    "desc": "Desaturate then fade. Memory / archival / past-tense."},

    # Wipes — period drama, action, vintage vocabulary.
    "wipeleft":    {"xfade": "wipeleft",     "desc": "Hard wipe right-to-left."},
    "wiperight":   {"xfade": "wiperight",    "desc": "Hard wipe left-to-right. Geographical move (\"meanwhile\")."},
    "wipeup":      {"xfade": "wipeup",       "desc": "Hard wipe bottom-to-top."},
    "wipedown":    {"xfade": "wipedown",     "desc": "Hard wipe top-to-bottom."},

    # Slides — modern documentary, push-cuts.
    "slideleft":   {"xfade": "slideleft",    "desc": "Push next shot in from the right; old shot slides off left."},
    "slideright":  {"xfade": "slideright",   "desc": "Push next shot in from the left; old shot slides off right."},

    # Smooth sweeps — softer than wipes, less mechanical than slides.
    "smoothleft":  {"xfade": "smoothleft",   "desc": "Soft horizontal sweep right-to-left."},
    "smoothright": {"xfade": "smoothright",  "desc": "Soft horizontal sweep left-to-right."},

    # Iris — silent-film, stage, period homage.
    "circleopen":  {"xfade": "circleopen",   "desc": "Iris opening. Beginning of a sequence; reveal."},
    "circleclose": {"xfade": "circleclose",  "desc": "Iris closing. Sequence end / button / chapter close."},

    # Effect-driven — special-purpose, use sparingly.
    "radial":      {"xfade": "radial",       "desc": "Sweeping arc. Time passage, clock-wipe."},
    "pixelize":    {"xfade": "pixelize",     "desc": "Digital glitch. Dream sequence, surveillance, broken reality."},
    "hblur":       {"xfade": "hblur",        "desc": "Soft blur. Sleep, unconsciousness, intoxication, haze."},
}


DEFAULT_DURATION = 0.5    # seconds; mid-range for editorial transitions
MIN_NONCUT_DURATION = 0.10
MAX_DURATION = 2.0
# When a clip is shorter than 2 × the proposed transition duration we
# clamp the transition so it never overlaps more than 40% of either clip.
SHORTCLIP_TRANSITION_CAP = 0.40


# ============================================================================
# Validation
# ============================================================================

def transition_names() -> list[str]:
    """Public list of valid transition type names (for JSON-schema enums)."""
    return list(TRANSITIONS.keys())


def validate_transition(t: dict | None) -> dict:
    """Coerce a (possibly missing or malformed) transition dict into a
    well-formed `{type, duration, xfade, desc}` record. Unknown types
    fall back to `cut` rather than raising — pipeline robustness over
    strict validation, since this runs on agent-produced JSON.
    """
    if not t or not isinstance(t, dict):
        return _cut()
    name = str(t.get("type", "cut")).strip()
    if name not in TRANSITIONS:
        return _cut()
    if name == "cut":
        return _cut()
    raw_dur = t.get("duration", DEFAULT_DURATION)
    try:
        dur = float(raw_dur)
    except (TypeError, ValueError):
        dur = DEFAULT_DURATION
    dur = max(MIN_NONCUT_DURATION, min(MAX_DURATION, dur))
    return {
        "type":     name,
        "duration": dur,
        "xfade":    TRANSITIONS[name]["xfade"],
        "desc":     TRANSITIONS[name]["desc"],
    }


def _cut() -> dict:
    return {"type": "cut", "duration": 0.0, "xfade": None, "desc": TRANSITIONS["cut"]["desc"]}


def clamp_for_clip_durations(t: dict, left_dur: float, right_dur: float) -> dict:
    """Shorten a transition if either neighboring clip is too short to
    absorb it without disappearing. Never extends, only clamps down."""
    if t["type"] == "cut":
        return t
    cap = min(left_dur, right_dur) * SHORTCLIP_TRANSITION_CAP
    if cap < MIN_NONCUT_DURATION:
        return _cut()  # Both clips too short to transition meaningfully.
    if t["duration"] > cap:
        return {**t, "duration": cap}
    return t


# ============================================================================
# Prompt guidance — Claude reads this when planning the storyboard
# ============================================================================

def prompt_guidance() -> str:
    """Return a markdown blurb describing the transition catalog. Inject
    into the SHOTLIST_SYSTEM prompt so Claude knows what's available."""
    lines = [
        "Available transitions for `transition_out` (between this shot and the next):",
        "",
    ]
    for name, meta in TRANSITIONS.items():
        lines.append(f"  - `{name}` — {meta['desc']}")
    lines += [
        "",
        f"Default is `cut` (hard cut, no overlap). When you do specify a",
        f"non-cut transition, also set `duration` in seconds — typical",
        f"editorial range is 0.3-1.0s; cap is {MAX_DURATION}s. Use",
        f"transitions sparingly: a film with a transition on every shot",
        f"is amateurish. Reserve them for: scene endings (fadeblack),",
        f"time jumps (fadeblack/fadegrays), revelation beats (fadewhite),",
        f"montage interiors (fade/dissolve). Within a scene, default to",
        f"hard cuts.",
    ]
    return "\n".join(lines)


# ============================================================================
# Rendering
# ============================================================================

def _ffprobe_duration(path: Path) -> float:
    """Read a clip's duration via ffprobe (already required by the project)."""
    if not shutil.which("ffprobe"):
        # Fallback to ffmpeg's stderr parsing — works but slower.
        return _ffmpeg_probe_duration(path)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def _ffmpeg_probe_duration(path: Path) -> float:
    """Fallback duration probe using ffmpeg itself."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    )
    # ffmpeg writes to stderr like "Duration: 00:00:08.00, ..."
    for line in r.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            ts = line.split(",", 1)[0].split("Duration:", 1)[1].strip()
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"Could not probe duration of {path}")


def _build_filter_complex(
    durations: list[float],
    transitions: list[dict],
    width: int,
    height: int,
    fps: int,
) -> tuple[str, str, str]:
    """Build the filter_complex for chained xfade + acrossfade.

    Returns (filter_complex, last_video_label, last_audio_label).

    Math: each xfade overlaps `t.duration` seconds, so the running
    cumulative output duration after the i-th xfade equals
    output_so_far + clip[i+1].duration - t.duration. The xfade `offset`
    parameter is the time at which the fade BEGINS in the running stream.
    """
    n = len(durations)
    if n < 2:
        raise ValueError("Need at least 2 clips to apply a transition")
    if len(transitions) != n - 1:
        raise ValueError(
            f"Expected {n - 1} transitions for {n} clips, got {len(transitions)}"
        )

    parts: list[str] = []

    # Step 1 — normalize every input stream to identical (width, height,
    # fps, sar). xfade requires this; otherwise it errors.
    for i in range(n):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]"
        )
        parts.append(
            f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=48000,"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )

    # Step 2 — chain the xfades. After each xfade the running stream is
    # named v01, v02, ... vNN_out for video and similarly for audio.
    last_v = "v0"
    last_a = "a0"
    output_so_far = durations[0]

    for i, t in enumerate(transitions):
        clamped = clamp_for_clip_durations(t, durations[i], durations[i + 1])
        is_last = i == len(transitions) - 1
        out_v = "vout" if is_last else f"vx{i + 1:02d}"
        out_a = "aout" if is_last else f"ax{i + 1:02d}"

        if clamped["xfade"] is None:
            # Cut: concat the two streams head-to-tail. We keep using
            # filter_complex so the rest of the chain stays uniform.
            parts.append(
                f"[{last_v}][v{i + 1}]concat=n=2:v=1:a=0[{out_v}]"
            )
            parts.append(
                f"[{last_a}][a{i + 1}]concat=n=2:v=0:a=1[{out_a}]"
            )
            output_so_far = output_so_far + durations[i + 1]
        else:
            offset = output_so_far - clamped["duration"]
            # Floor offset at 0 in case of rounding / very short first clip.
            offset = max(0.0, offset)
            parts.append(
                f"[{last_v}][v{i + 1}]"
                f"xfade=transition={clamped['xfade']}"
                f":duration={clamped['duration']:.3f}:offset={offset:.3f}[{out_v}]"
            )
            parts.append(
                f"[{last_a}][a{i + 1}]"
                f"acrossfade=d={clamped['duration']:.3f}[{out_a}]"
            )
            output_so_far = output_so_far + durations[i + 1] - clamped["duration"]

        last_v = out_v
        last_a = out_a

    return ";".join(parts), last_v, last_a


def render_clips_with_transitions(
    clip_paths: list[Path],
    transitions: list[dict] | None,
    output_path: Path,
    *,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
    bitrate: str = "8000k",
) -> Path:
    """Render an ordered list of clips into one mp4, applying the given
    transitions between consecutive pairs.

    Args:
        clip_paths: ordered list of input mp4 paths (already trimmed if
            you want EDL in/out applied).
        transitions: list of len(clip_paths) - 1 transition dicts.
            Each dict comes from `validate_transition()`. Pass None or []
            to default to all hard cuts.
        output_path: where to write the result.
        fps: target framerate. All inputs are normalized to this.
        width, height: target dimensions; inputs are scaled-and-padded
            to fit (no crop, no stretch). 1280x720 is the production
            default for 720p; bump to 1920x1080 for hero runs.
        bitrate: libx264 target bitrate. 8000k matches compile_final's
            existing pre-grade output.

    Returns:
        The output_path.

    Notes:
        - Single-clip case: just copies the input through.
        - All-cuts case: still routed through filter_complex with
          concat. This is slightly slower than ffmpeg's stream-copy
          concat but produces a single re-encoded output that's safe to
          feed back into compile_final's color-grade pass. The speedup
          isn't worth the special-case branching.
    """
    if not clip_paths:
        raise ValueError("clip_paths is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Single clip — copy through.
    if len(clip_paths) == 1:
        shutil.copy(str(clip_paths[0]), str(output_path))
        return output_path

    # Validate every transition (defensive — caller may have skipped it).
    n_trans = len(clip_paths) - 1
    transitions = transitions or [{"type": "cut"} for _ in range(n_trans)]
    if len(transitions) < n_trans:
        transitions = list(transitions) + [{"type": "cut"} for _ in range(n_trans - len(transitions))]
    transitions = [validate_transition(t) for t in transitions[:n_trans]]

    # Probe each clip's duration. Required for xfade offset math.
    durations = [_ffprobe_duration(Path(p)) for p in clip_paths]

    filter_complex, last_v, last_a = _build_filter_complex(
        durations, transitions, width, height, fps,
    )

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in clip_paths:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{last_v}]", "-map", f"[{last_a}]",
        "-c:v", "libx264", "-preset", "medium", "-b:v", bitrate,
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


# ============================================================================
# Convenience — used by produce.py to thread storyboard transitions into EDL
# ============================================================================

def transitions_for_scene(scene_decisions: list[dict]) -> list[dict]:
    """Given an in-order list of EDL decisions for one scene, return the
    list of N-1 transitions that should be applied between consecutive
    shots. The transition between shots i and i+1 comes from shot i's
    `transition_out` field (default cut)."""
    if not scene_decisions:
        return []
    out: list[dict] = []
    for d in scene_decisions[:-1]:  # last shot's transition_out is for cross-scene
        out.append(validate_transition(d.get("transition_out")))
    return out


def any_non_cut(transitions: list[dict]) -> bool:
    """True if at least one transition in the list is not a hard cut.
    Lets compile_final fast-path the common case (everything's a cut →
    use moviepy's existing concatenate, no re-encode)."""
    return any(t.get("type", "cut") != "cut" for t in transitions)
