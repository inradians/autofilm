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
    """Public list of valid transition type names (for JSON-schema enums).

    Includes both the built-in ffmpeg-xfade catalog and any GLSL
    transitions registered via ``register_glsl_transition()``.
    """
    return list(TRANSITIONS.keys()) + list(GLSL_TRANSITIONS.keys())


def validate_transition(t: dict | None) -> dict:
    """Coerce a (possibly missing or malformed) transition dict into a
    well-formed ``{type, duration, xfade, glsl, desc}`` record. Unknown
    types fall back to ``cut`` rather than raising — pipeline robustness
    over strict validation, since this runs on agent-produced JSON.

    Looks up ``type`` in ``TRANSITIONS`` (ffmpeg-xfade catalog) first,
    then in ``GLSL_TRANSITIONS`` (custom shader registry, populated by
    ``register_glsl_transition``). GLSL entries return a dict with
    ``glsl`` populated and ``xfade`` set to ``None``; the renderer
    branches on which is present.
    """
    if not t or not isinstance(t, dict):
        return _cut()
    name = str(t.get("type", "cut")).strip()
    if name == "cut":
        return _cut()
    if name in TRANSITIONS:
        meta = TRANSITIONS[name]
        is_glsl = False
    elif name in GLSL_TRANSITIONS:
        meta = GLSL_TRANSITIONS[name]
        is_glsl = True
    else:
        return _cut()
    raw_dur = t.get("duration", meta.get("default_duration", DEFAULT_DURATION))
    try:
        dur = float(raw_dur)
    except (TypeError, ValueError):
        dur = DEFAULT_DURATION
    dur = max(MIN_NONCUT_DURATION, min(MAX_DURATION, dur))
    if is_glsl:
        return {
            "type":     name,
            "duration": dur,
            "xfade":    None,
            "glsl":     meta["shader"],
            "desc":     meta["desc"],
        }
    return {
        "type":     name,
        "duration": dur,
        "xfade":    meta["xfade"],
        "glsl":     None,
        "desc":     meta["desc"],
    }


def _cut() -> dict:
    return {
        "type":     "cut",
        "duration": 0.0,
        "xfade":    None,
        "glsl":     None,
        "desc":     TRANSITIONS["cut"]["desc"],
    }


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
    if GLSL_TRANSITIONS:
        lines += [
            "",
            "Custom GLSL shader transitions (slower than ffmpeg ones — use sparingly):",
            "",
        ]
        for name, meta in GLSL_TRANSITIONS.items():
            lines.append(f"  - `{name}` — {meta['desc']}  *(GLSL)*")
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


def _ffprobe_video_stream_duration(path: Path) -> float | None:
    """Read the VIDEO STREAM duration (not container duration).

    Used to defend against audio/video drift in muxed files: the
    container reports max(audio_duration, video_duration), so when
    audio runs longer (acrossfade output), moviepy's `clip.duration`
    is too long and trying to read past the actual frame count
    produces 'using last valid frame' warnings — visible as a frozen
    tail. Trimming to this value avoids that.

    Returns None on probe failure so callers can fall through.
    """
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(r.stdout).get("streams", [])
        if streams and streams[0].get("duration"):
            return float(streams[0]["duration"])
    except Exception:                                            # noqa: BLE001
        pass
    return None


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
    #
    # We use xfade for EVERY pair, even hard cuts. Cuts are encoded as a
    # 1-frame (1/fps) fade — visually indistinguishable from a hard cut
    # at 24fps but keeping the filter graph homogeneous. Mixing xfade
    # with concat in a single graph fails because concat outputs at
    # timebase 1/1000000 while xfade wants 1/fps.
    cut_duration = 1.0 / float(fps)
    last_v = "v0"
    last_a = "a0"
    output_so_far = durations[0]

    for i, t in enumerate(transitions):
        clamped = clamp_for_clip_durations(t, durations[i], durations[i + 1])
        is_last = i == len(transitions) - 1
        out_v = "vout" if is_last else f"vx{i + 1:02d}"
        out_a = "aout" if is_last else f"ax{i + 1:02d}"

        if clamped["xfade"] is None:
            # Hard cut: use a 1-frame fade. Visually identical to a cut
            # at this fps, but keeps the filter graph all-xfade so the
            # timebases stay consistent.
            xfade_name = "fade"
            xfade_duration = cut_duration
        else:
            xfade_name = clamped["xfade"]
            xfade_duration = clamped["duration"]

        offset = output_so_far - xfade_duration
        offset = max(0.0, offset)
        parts.append(
            f"[{last_v}][v{i + 1}]"
            f"xfade=transition={xfade_name}"
            f":duration={xfade_duration:.4f}:offset={offset:.4f}[{out_v}]"
        )
        parts.append(
            f"[{last_a}][a{i + 1}]"
            f"acrossfade=d={xfade_duration:.4f}[{out_a}]"
        )
        output_so_far = output_so_far + durations[i + 1] - xfade_duration

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

    # GLSL routing: if any transition uses a custom shader, switch to
    # pairwise rendering. The single-filter-graph fast path can't hold
    # custom shader stages.
    if _any_glsl(transitions):
        return _render_pairwise(
            clip_paths, transitions, output_path,
            fps=fps, width=width, height=height, bitrate=bitrate,
        )

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
        # -shortest is critical here: ffmpeg's acrossfade output runs
        # a hair longer than xfade's video output, so without this the
        # muxer's duration metadata is longer than the actual video.
        # moviepy then tries to read N "frames" past the real end of
        # the video and fills them with the last valid frame — a
        # visible 1-3s freeze at the tail of every transitioned scene.
        "-shortest",
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


# ============================================================================
# OPTIONAL GLSL EXTENSION
# ============================================================================
# Custom transitions written as fragment shaders following the
# gl-transitions.com convention. *Opt-in* — an experiment only pays the
# moderngl import cost and slower pairwise rendering when the agent
# actually uses a GLSL transition. The default catalog stays
# ffmpeg-xfade only.
#
# The original module-level note (top of file) argued against GLSL on
# dependency-cost grounds, and that argument still applies for the 90%
# case. This extension exists for the 10% case: chromatic aberration,
# displacement maps, custom warps — things xfade can't express. Use it
# deliberately, not by default.

GLSL_TRANSITIONS: dict[str, dict[str, Any]] = {}
"""Registry of GLSL transitions. Populated by ``register_glsl_transition``."""


def register_glsl_transition(
    name: str,
    shader: str,
    *,
    default_duration: float = 1.0,
    desc: str = "",
) -> None:
    """Register a custom transition implemented as a fragment shader.

    Args:
        name: storyboard-visible name (e.g. ``"chromatic_glitch"``). Must
            not collide with an ffmpeg-xfade entry; collisions raise.
        shader: a gl-transitions style fragment-shader BODY. Must define a
            ``vec4 transition(vec2 uv)`` function. The harness adds
            ``#version 330``, the uniforms (``from``, ``to``, ``progress``,
            ``ratio``), a ``texture2D`` macro, a ``random()`` helper, and
            the ``main()`` wrapper — don't redeclare any of those.
        default_duration: seconds; used when storyboards omit ``duration``.
            Same clamping rules apply as for ffmpeg transitions.
        desc: short editorial description, surfaced in the bible and in
            the SHOTLIST_SYSTEM prompt guidance.
    """
    if name in TRANSITIONS:
        raise ValueError(
            f"GLSL transition name '{name}' collides with an ffmpeg-xfade "
            f"entry. Pick a different name."
        )
    GLSL_TRANSITIONS[name] = {
        "shader":           shader,
        "default_duration": float(default_duration),
        "desc":             desc or f"Custom GLSL transition ({name})",
    }


def _any_glsl(transitions: list[dict]) -> bool:
    return any(t.get("glsl") for t in transitions)


# ----- Default GLSL transitions shipped out of the box ---------------------
# Follow the gl-transitions.com convention. Agents can add more by
# calling register_glsl_transition() from produce.py with their own
# shader source — the registration goes into the experiment's produce.py
# snapshot so it's reproducible.

_GLSL_CHROMATIC_GLITCH = """
// RGB-split glitch dissolve. Shifts R and B channels horizontally with
// rising amplitude through the middle of the transition, then settles.
// Reads as a digital intrusion / signal-loss / surveillance-camera moment.
vec4 transition(vec2 uv) {
    float p = progress;
    float bell = sin(p * 3.14159);                 // 0 at edges, 1 in middle
    float jitter = (random(uv + vec2(p, 0.0)) - 0.5) * 0.06 * bell;
    vec2 r_uv = uv + vec2(jitter,  0.0);
    vec2 b_uv = uv + vec2(-jitter, 0.0);
    float r = mix(texture2D(from, r_uv).r, texture2D(to, r_uv).r, p);
    float g = mix(texture2D(from, uv  ).g, texture2D(to, uv  ).g, p);
    float b = mix(texture2D(from, b_uv).b, texture2D(to, b_uv).b, p);
    return vec4(r, g, b, 1.0);
}
"""

_GLSL_DISPLACEMENT_PUSH = """
// Displacement-mapped push. The luminance of `from` drives a per-pixel
// displacement of `to`'s sample coordinate, so bright pixels of A push
// B's image around as B fades in. Reads as a "dissolution" — good for
// memory beats or fluid POV transitions.
vec4 transition(vec2 uv) {
    float p = progress;
    vec3 ref = texture2D(from, uv).rgb;
    float disp = (ref.r + ref.g + ref.b) / 3.0;     // luminance proxy
    vec2 dir = vec2(disp - 0.5, disp - 0.5) * 0.15 * p;
    vec4 a = texture2D(from, uv);
    vec4 b = texture2D(to,   uv + dir);
    return mix(a, b, smoothstep(0.0, 1.0, p));
}
"""

register_glsl_transition(
    "chromatic_glitch",
    _GLSL_CHROMATIC_GLITCH,
    default_duration=0.8,
    desc="RGB-split glitch dissolve. Use for digital intrusion / signal-loss beats.",
)
register_glsl_transition(
    "displacement_push",
    _GLSL_DISPLACEMENT_PUSH,
    default_duration=1.0,
    desc="Luminance-driven displacement push. Use for memory or fluid POV shifts.",
)


# ----- GLSL renderer (moderngl + ffmpeg piping) ----------------------------
# Headless via the EGL backend — no X server required. Falls back to the
# default standalone backend if EGL isn't available. Per-frame: upload
# both source frames as textures, set ``progress``, render to FBO, read
# back. Roughly real-time at 720p on llvmpipe; effectively instant on a
# real GPU.

_GLSL_VERTEX = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_GLSL_FRAGMENT_TEMPLATE = """
#version 330
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D from;
uniform sampler2D to;
uniform float progress;
uniform float ratio;

#define texture2D texture

float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

%USER_BODY%

void main() {
    fragColor = transition(v_uv);
}
"""


def _read_video_frames(path: Path, ss: float, t: float, w: int, h: int, fps: int) -> bytes:
    """Read [ss, ss+t] of `path`, scaled to WxH at `fps`, as rgb24 bytes."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", f"{ss:.3f}", "-t", f"{t:.3f}",
        "-i", str(path),
        "-vf", f"scale={w}:{h}:flags=bilinear,fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _write_rgb24_video(buf: bytes, w: int, h: int, fps: int, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        str(out),
    ]
    subprocess.run(cmd, input=buf, check=True)


def _has_audio_stream(path: Path) -> bool:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return bool(json.loads(r.stdout).get("streams"))


def _concat_files(clips: list[Path], out: Path, *, width: int, height: int, fps: int) -> Path:
    """Re-encoding concat with WxH/fps normalization. Used by the GLSL
    pairwise renderer to glue (head_a, transition, tail_b) together and
    to handle plain cuts on the GLSL pairwise path."""
    if len(clips) == 1:
        shutil.copy(str(clips[0]), str(out))
        return out
    n = len(clips)
    have_audio = all(_has_audio_stream(c) for c in clips)
    parts: list[str] = []
    for i in range(n):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]"
        )
        if have_audio:
            parts.append(
                f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=48000[a{i}]"
            )
    if have_audio:
        parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
        maps = ["-map", "[v]"]
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(parts),
        *maps,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _render_glsl_pair(
    a: Path,
    b: Path,
    shader_body: str,
    duration: float,
    out: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Render the GLSL transition between A's tail and B's head, mux audio
    crossfade onto the result, then concat with head_of_a + tail_of_b."""
    try:
        import moderngl
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "GLSL transitions require moderngl + numpy. "
            "Install with `uv add moderngl numpy` or use a non-GLSL transition."
        ) from e

    a_dur = _ffprobe_duration(a)
    b_dur = _ffprobe_duration(b)
    duration = max(MIN_NONCUT_DURATION, min(MAX_DURATION, duration, a_dur, b_dur))
    transition_start_in_a = max(0.0, a_dur - duration)

    tail_buf = _read_video_frames(a, transition_start_in_a, duration, width, height, fps)
    head_buf = _read_video_frames(b, 0.0,                    duration, width, height, fps)
    frame_bytes = width * height * 3
    actual_n = min(len(tail_buf), len(head_buf)) // frame_bytes
    if actual_n < 1:
        return _concat_files([a, b], out, width=width, height=height, fps=fps)

    tail = np.frombuffer(tail_buf[: actual_n * frame_bytes], dtype=np.uint8).reshape(actual_n, height, width, 3)
    head = np.frombuffer(head_buf[: actual_n * frame_bytes], dtype=np.uint8).reshape(actual_n, height, width, 3)

    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception:
        ctx = moderngl.create_standalone_context()

    fragment = _GLSL_FRAGMENT_TEMPLATE.replace("%USER_BODY%", shader_body)
    prog = ctx.program(vertex_shader=_GLSL_VERTEX, fragment_shader=fragment)

    quad = np.array(
        [
            -1, -1, 0.0, 1.0,
             1, -1, 1.0, 1.0,
            -1,  1, 0.0, 0.0,
             1,  1, 1.0, 0.0,
        ],
        dtype="f4",
    )
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos", "in_uv")

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 3)])
    fbo.use()
    ctx.viewport = (0, 0, width, height)

    tex_from = ctx.texture((width, height), 3)
    tex_to   = ctx.texture((width, height), 3)
    tex_from.use(location=0)
    tex_to.use(location=1)
    if "from" in prog: prog["from"].value = 0
    if "to"   in prog: prog["to"].value   = 1
    if "ratio" in prog: prog["ratio"].value = width / height

    rendered = bytearray()
    for i in range(actual_n):
        # ffmpeg gives top-down rgb24; flip rows so gl-transitions UVs
        # land where the shader expects them.
        tex_from.write(np.ascontiguousarray(tail[i][::-1]).tobytes())
        tex_to.write(  np.ascontiguousarray(head[i][::-1]).tobytes())
        if "progress" in prog:
            prog["progress"].value = i / max(1, actual_n - 1)
        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        pixels = fbo.color_attachments[0].read()
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1]
        rendered.extend(arr.tobytes())

    import tempfile
    with tempfile.TemporaryDirectory(prefix="autofilm_glsl_") as tmp:
        tmpdir = Path(tmp)
        trans_clip = tmpdir / "trans.mp4"
        _write_rgb24_video(bytes(rendered), width, height, fps, trans_clip)

        a_has = _has_audio_stream(a)
        b_has = _has_audio_stream(b)
        if a_has and b_has:
            trans_with_audio = tmpdir / "trans_with_audio.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(trans_clip),
                    "-ss", f"{transition_start_in_a:.3f}", "-t", f"{duration:.3f}", "-i", str(a),
                    "-t", f"{duration:.3f}", "-i", str(b),
                    "-filter_complex",
                    f"[1:a][2:a]acrossfade=duration={duration:.3f}:overlap=1[a]",
                    "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    str(trans_with_audio),
                ],
                check=True,
            )
            trans_clip = trans_with_audio

        head_a = tmpdir / "head_a.mp4"
        tail_b = tmpdir / "tail_b.mp4"
        if transition_start_in_a > 0.01:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(a), "-t", f"{transition_start_in_a:.3f}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k",
                    str(head_a),
                ],
                check=True,
            )
        if b_dur > duration + 0.01:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{duration:.3f}", "-i", str(b),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k",
                    str(tail_b),
                ],
                check=True,
            )

        parts = [p for p in (head_a, trans_clip, tail_b) if p.exists()]
        return _concat_files(parts, out, width=width, height=height, fps=fps)


def _render_pairwise(
    clip_paths: list[Path],
    transitions: list[dict],
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    bitrate: str,
) -> Path:
    """Pairwise renderer used when at least one transition is GLSL.

    Walks (clip_a, clip_b) pairs in order, applying ffmpeg-xfade or GLSL
    per transition. After each pair the rolling output becomes the new
    clip_a for the next iteration. Slower than the single-filter-graph
    fast path (~N file rewrites per scene), but it's the only way to
    interleave shader stages with xfade stages cleanly.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="autofilm_pairwise_") as tmp:
        tmpdir = Path(tmp)
        rolling = Path(clip_paths[0])
        for i, t in enumerate(transitions):
            stage = tmpdir / f"stage_{i:03d}.mp4"
            nxt = Path(clip_paths[i + 1])
            t = clamp_for_clip_durations(
                t, _ffprobe_duration(rolling), _ffprobe_duration(nxt)
            )
            if t.get("glsl"):
                _render_glsl_pair(
                    rolling, nxt, t["glsl"], t["duration"], stage,
                    width=width, height=height, fps=fps,
                )
            elif t.get("xfade"):
                a_dur = _ffprobe_duration(rolling)
                b_dur = _ffprobe_duration(nxt)
                fc, lv, la = _build_filter_complex([a_dur, b_dur], [t], width, height, fps)
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(rolling), "-i", str(nxt),
                    "-filter_complex", fc,
                    "-map", f"[{lv}]", "-map", f"[{la}]",
                    "-c:v", "libx264", "-preset", "medium", "-b:v", bitrate,
                    "-pix_fmt", "yuv420p", "-r", str(fps),
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    str(stage),
                ]
                subprocess.run(cmd, check=True)
            else:
                _concat_files([rolling, nxt], stage, width=width, height=height, fps=fps)
            rolling = stage

        shutil.copy(str(rolling), str(output_path))
    return output_path
