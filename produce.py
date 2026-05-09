"""produce.py — the single file the agent edits.

Everything taste-shaped lives here. Everything pipe-shaped lives in
prepare.py. The agent's job is to iterate on this file to drive
film_loss down across experiments.

The pipeline runs sequentially: parse → cast → lookbook → references →
shotlist → frames → video → edit → mix → done. Each step writes its
artifact and the next step reads it. State is persisted under
experiments/exp_NNN/ so a crash mid-pipeline is recoverable.

If you (the agent) want to change creative direction:
  - To change the visual style globally → edit LOOKBOOK_PROMPT or LOOKBOOK_GRADE.
  - To change shot density / coverage → edit shot_list_for_scene().
  - To change performance variation → edit veo_prompt() and TAKES_PER_SHOT.
  - To change casting strategy → edit CAST_SYSTEM.
  - To change the dialogue read → edit veo_prompt() (the dialogue block).
  - To favor one model over another for a step → swap which helper you call.

Structural changes (new pipeline steps, new metrics) belong in prepare.py.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from prepare import (
    Experiment,
    BFL_API_BASE,
    FLUX2_PRO_MODEL,
    FLUX_PRO_MODEL,
    GEMINI_FLASH_MODEL,
    GEN4_IMAGE_MODEL,
    GOOGLE_IMAGE_MODEL,
    GOOGLE_VEO_MODEL,
    GPT_IMAGE_MODEL,
    LTX_API_BASE,
    LTX_FAST_MODEL,
    LTX_PRO_MODEL,
    LTX_VIDEO_MODEL,
    MAX_PLANNED_SHOT_SECONDS,
    MAX_SCENES,
    NANO_BANANA_MODEL,
    REVE_API_BASE,
    SEEDANCE2_MODEL,
    SHOT_DURATION_SECONDS,
    TAKES_PER_SHOT,
    VEO_MODEL_LITE,
    VEO_TIER,
    VIDEO_MODELS,
    book_chunks,
    claude_text,
    claude_tool,
    elevenlabs_sfx,
    extract_video_frame,
    ffmpeg,
    flux_image,
    gen4_image,
    gpt_image,
    google_nano_banana,
    google_veo,
    ltx_video,
    nano_banana,
    openai_image,
    plan_shot_durations,
    REVE_API_BASE,
    reve_image,
    route_shot,
    runway_image,
    seedance,
    stable_audio,
    veo,
    veo_final_model,
)
from transitions import (
    DEFAULT_DURATION as TRANSITION_DEFAULT_DURATION,
    any_non_cut,
    prompt_guidance as transitions_prompt_guidance,
    render_clips_with_transitions,
    transition_names,
    transitions_for_scene,
    validate_transition,
)


# ============================================================================
# CREATIVE KNOBS (the agent should iterate on these between experiments)
# ============================================================================

# OPTIONAL: name a director and/or DP whose body of work should bias the
# look book. Real working artists. Leave blank ("" or unset) for the
# pipeline's default neutral cinematic style. When set, the look book
# stage derives concrete craft markers (lens choice, lighting approach,
# palette, framing patterns) from their published filmography and bakes
# those into the lookbook.json — the Veo prompts then use the derived
# markers, not the names themselves.
DIRECTOR = os.getenv("DIRECTOR", "").strip()
CINEMATOGRAPHER = os.getenv("CINEMATOGRAPHER", "").strip()

# Locked visual style. Lower number of distinct visual decisions = more
# coherent film. The agent overrides this between experiments.
LOOKBOOK_STYLE_KEYWORDS = [
    "anamorphic",
    "practical light",
    "teal-and-orange grade",
    "35mm grain",
    "shallow depth of field",
]

# ffmpeg -vf chain applied at compile time to lock the grade across shots.
# Contrast slightly raised, saturation slightly muted, midtones pushed
# toward warm/orange and shadows toward cool/teal as a neutral baseline.
LOOKBOOK_GRADE = (
    "eq=contrast=1.08:saturation=0.92:gamma=0.97,"
    "colorbalance=rs=0.04:gs=0:bs=-0.06:rm=0:gm=0:bm=-0.02:rh=-0.04:gh=-0.02:bh=0.04"
)

# Music style — shifts whole emotional tone of the film.
MUSIC_STYLE = (
    "orchestral cinematic score, sweeping strings and low brass, "
    "restrained, building tension, no vocals"
)


# ============================================================================
# CONCURRENCY
# ============================================================================

# How many image/video/audio generation tasks to run in parallel.
# Each task holds a live Runway API call polling for completion, so the
# effective limit is whatever Runway allows concurrently on your account
# (typically 10-20 tasks). Set to 1 to disable parallelism entirely
# (useful for debugging; every print statement appears in order).
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))

# Generation backend selector. Set in .env or shell to switch away from
# the default without changing any other code.
#
# VIDEO_BACKEND=seedance — Runway SeedDance 2 (default)
#                          36 credits/sec, identity-consistent, up to 15s,
#                          native reference images, RUNWAYML_API_SECRET
# VIDEO_BACKEND=runway   — Runway Veo 3.1 (previous default)
# VIDEO_BACKEND=google   — Google Veo 3.1 direct (GOOGLE_AI_API_KEY)
# VIDEO_BACKEND=ltx      — LTX 2.3 (LTX_API_KEY, console.ltx.video)
#
# IMAGE_BACKEND=google   — Google Imagen 3 instead of gpt_image / nano_banana
IMAGE_BACKEND: str = os.getenv("IMAGE_BACKEND", "runway")
VIDEO_BACKEND: str = os.getenv("VIDEO_BACKEND", "seedance")

# Thread-safe print — prevents interleaved output from parallel workers.
_PRINT_LOCK = threading.Lock()

# Set to True the first time any worker detects a Runway daily task limit.
# Once set, all subsequent workers check this flag and bail out immediately
# rather than consuming more quota on attempts that will also fail.
_daily_limit_hit = threading.Event()


def _is_daily_limit(exc: Exception) -> bool:
    """Return True if the exception is a Runway 429 daily task limit."""
    msg = str(exc)
    return "429" in msg and (
        "daily" in msg.lower() or "task limit" in msg.lower()
    )


def _check_daily_limit() -> None:
    """Raise RuntimeError if the daily task limit has been hit this run."""
    if _daily_limit_hit.is_set():
        raise RuntimeError(
            "Runway daily task limit reached — skipping (re-run tomorrow "
            "or top up credits; the experiment resumes from this point)."
        )


def _record_daily_limit(exc: Exception, context: str) -> None:
    """Flag the daily limit and print a prominent one-time warning."""
    if _daily_limit_hit.is_set():
        return
    _daily_limit_hit.set()
    _tprint(
        f"\n  {'='*60}\n"
        f"  ⛔  RUNWAY DAILY TASK LIMIT REACHED at {context}\n"
        f"  All remaining generation tasks will be skipped.\n"
        f"  Re-run tomorrow (or top up credits at dev.runwayml.com).\n"
        f"  The experiment resumes automatically from this point.\n"
        f"  {'='*60}\n"
    )


def _tprint(*args, **kwargs) -> None:
    """Thread-safe print. Use inside parallel worker functions."""
    with _PRINT_LOCK:
        print(*args, **kwargs)


def _parallel_run(
    label: str,
    work_items: list,
    worker_fn,
    *,
    workers: int | None = None,
) -> list:
    """Submit all work items to a thread pool, then block at an explicit
    barrier until every job has finished OR failed before returning.

    The barrier guarantees that the next pipeline stage never starts while
    any job from the current stage is still running. It uses
    ``concurrent.futures.wait(return_when=ALL_COMPLETED)`` rather than
    ``as_completed`` so the semantics are unambiguous: we wait for the
    complete set, not just for items as they trickle in.

    Falls back to serial execution when MAX_WORKERS == 1 or work_items
    has only one item (easier to debug; logs appear in order).

    Args:
        label:      Short stage name used in progress and failure messages.
        work_items: One argument per worker_fn call.
        worker_fn:  Thread-safe callable; receives one item, returns a result.
        workers:    Pool size override. Defaults to MAX_WORKERS.

    Returns:
        List of successful return values (order matches completion order,
        not input order). Failed items are logged and omitted.
    """
    n = workers if workers is not None else MAX_WORKERS

    # ── Serial path ──────────────────────────────────────────────────────
    if n <= 1 or len(work_items) <= 1:
        results = []
        for i, item in enumerate(work_items, 1):
            print(f"  [{label}] {i}/{len(work_items)}")
            try:
                results.append(worker_fn(item))
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ [{label}] job {i} failed: {e}")
        return results

    # ── Parallel path ────────────────────────────────────────────────────
    n_jobs = len(work_items)
    print(f"  ┌─ [{label}] submitting {n_jobs} job(s) "
          f"(MAX_WORKERS={n}) ─────────────────")

    with ThreadPoolExecutor(max_workers=n) as pool:
        future_to_item = {pool.submit(worker_fn, item): item
                          for item in work_items}

        # ── BARRIER ──────────────────────────────────────────────────────
        # Block here until every submitted future has either completed
        # successfully or raised an exception. The next stage only starts
        # after this returns.
        done, not_done = wait(future_to_item, return_when=ALL_COMPLETED)
        # ─────────────────────────────────────────────────────────────────

    # With ALL_COMPLETED, not_done is always empty — but handle it
    # defensively in case of executor edge cases.
    for f in not_done:
        item = future_to_item[f]
        _tprint(f"  ✗ [{label}] job did not complete (cancelled?): {item}")
        try:
            f.cancel()
        except Exception:
            pass

    results: list = []
    n_ok = n_fail = 0
    for f in done:
        try:
            results.append(f.result())
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            item = future_to_item[f]
            _tprint(f"  ✗ [{label}] failed: {e}")

    status = "✓ all succeeded" if n_fail == 0 else f"✓ {n_ok} ok  ✗ {n_fail} failed"
    print(f"  └─ [{label}] barrier cleared — {status} "
          f"({n_jobs} job(s)) ────────────────")
    return results


# ============================================================================
# STAGE 1 — Parse book → screenplay
# ============================================================================

PARSE_SYSTEM = """You are a screenwriter analyzing a novel. Extract:
  - characters[]: every named, speaking character
  - scenes[]: every distinct screen-worthy scene
For each scene give id, location, time_of_day, page_start, page_end,
character ids present, summary, mood, and (optionally) up to 3 short
dialogue_excerpts — each at most 12 words, capturing the dramatic shape
of a key line, NOT a verbatim transcription. Return only via the tool."""

PARSE_TOOL_SCHEMA = {
    "description": "Submit characters and scene index for this chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "arc": {"type": "string"},
                    },
                    "required": ["id", "name", "description"],
                },
            },
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "location": {"type": "string"},
                        "time_of_day": {"type": "string"},
                        "page_start": {"type": "integer"},
                        "page_end": {"type": "integer"},
                        "characters": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "mood": {"type": "string"},
                        "dialogue_excerpts": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "object", "properties": {
                                "character_id": {"type": "string"},
                                "line": {
                                    "type": "string",
                                    "description": "≤12 words; dramatic shape, not transcription.",
                                },
                            }, "required": ["character_id", "line"]},
                        },
                    },
                    "required": ["id", "location", "time_of_day", "page_start", "page_end", "summary"],
                },
            },
        },
        "required": ["characters", "scenes"],
    },
}


def parse_script(exp: Experiment) -> dict:
    if exp.has("script.json"):
        return exp.read_json("script.json")

    chunks = book_chunks(pages_per_chunk=25)
    characters: list[dict] = []
    scenes: list[dict] = []

    for s, e, text in chunks:
        prior = json.dumps([{"id": c["id"], "name": c["name"]} for c in characters])
        result = claude_tool(
            system=PARSE_SYSTEM,
            user_content=f"BOOK pp.{s}-{e}:\n\n{text}\n\nPrior characters (reuse IDs):\n{prior}",
            tool_name="submit_extraction",
            tool_schema=PARSE_TOOL_SCHEMA,
            max_tokens=8000,
        )
        # Merge characters by id.
        by_id = {c["id"]: c for c in characters}
        for c in result.get("characters", []):
            if c["id"] not in by_id:
                by_id[c["id"]] = c
        characters = list(by_id.values())
        scenes.extend(result.get("scenes", []))

    if MAX_SCENES:
        scenes = scenes[:MAX_SCENES]

    script = {
        "title": "Jurassic Park",
        "source": "Jurassic Park by Michael Crichton (1990)",
        "characters": characters,
        "scenes": scenes,
    }
    exp.write_json("script.json", script)
    return script


# ============================================================================
# STAGE 1.5 — Format each scene as screenplay elements
# ============================================================================
# The parse stage extracts scene summaries; this stage converts each summary
# into proper screenplay format (slug / action / character / dialogue /
# parenthetical) so the bible has a real readable screenplay section.
#
# Input is the *paraphrased scene summary* the parse stage already produced
# — not the raw book pages. The output is original screenplay-format prose
# in Claude's own words: terse visual action lines, short indicative
# dialogue beats. This mirrors how real adaptation works: a screenwriter
# reads the source and writes screenplay action and dialogue from their own
# craft, not by copying novel paragraphs.

SCREENPLAY_FORMAT_SYSTEM = """You are a working screenwriter formatting a
scene from a treatment. The scene summary, mood, location, and characters
are given. Produce a sequence of SCREENPLAY ELEMENTS in your own words:

  - "action" elements: terse, third-person, present-tense visual prose.
    What the camera sees. One element per beat. Each element MUST be at
    most 25 words. Write in your own voice; do NOT transcribe or quote
    from any source novel — interpret the summary as a screenwriter
    adapting it.

  - "dialogue" elements: short character lines that capture the dramatic
    shape of the beat. Each line MUST be at most 15 words. Use the
    character_id field for the speaker. Optional parenthetical for tone
    (one or two words: "quietly", "amused", "warning").

  - "transition" elements: rare; only where the scene needs a hard
    transition like CUT TO BLACK or SMASH CUT.

Aim for 4-10 elements per scene. Mix action and dialogue. Open with an
action line establishing the visual; close with the strongest beat of
the summary. Return only via the tool."""

SCREENPLAY_FORMAT_TOOL_SCHEMA = {
    "description": "Submit formatted screenplay elements for one scene.",
    "input_schema": {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "minItems": 3,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["action", "dialogue", "transition"],
                        },
                        "text": {
                            "type": "string",
                            "description": "Action line (≤25 words) or dialogue line (≤15 words).",
                        },
                        "character": {
                            "type": "string",
                            "description": "character_id; required for dialogue.",
                        },
                        "parenthetical": {
                            "type": "string",
                            "description": "Optional tone hint, 1-2 words.",
                        },
                    },
                    "required": ["type", "text"],
                },
            },
        },
        "required": ["elements"],
    },
}


def format_screenplay(exp: Experiment, script: dict) -> dict:
    """Populate scene["elements"] for every scene in script.json.

    Cached: skips scenes that already have elements. Re-saves script.json
    in place so the bible's _screenplay_section picks them up.
    """
    if all(scene.get("elements") for scene in script.get("scenes", [])):
        return script

    char_by_id = {c["id"]: c for c in script.get("characters", [])}
    changed = False

    for scene in script.get("scenes", []):
        if scene.get("elements"):
            continue
        scene_chars = [
            {"id": cid, "name": char_by_id.get(cid, {}).get("name", cid),
             "description": char_by_id.get(cid, {}).get("description", "")[:80]}
            for cid in scene.get("characters", [])
        ]
        payload = {
            "scene_id":     scene["id"],
            "location":     scene.get("location", ""),
            "time_of_day":  scene.get("time_of_day", ""),
            "summary":      scene.get("summary", ""),
            "mood":         scene.get("mood", ""),
            "characters":   scene_chars,
        }
        result = claude_tool(
            system=SCREENPLAY_FORMAT_SYSTEM,
            user_content=(
                f"Scene to format:\n{json.dumps(payload, indent=2)}\n\n"
                "Format it as screenplay elements."
            ),
            tool_name="submit_screenplay",
            tool_schema=SCREENPLAY_FORMAT_TOOL_SCHEMA,
            max_tokens=3000,
        )
        scene["elements"] = result.get("elements", [])
        changed = True

    if changed:
        exp.write_json("script.json", script)
    return script


# ============================================================================
# STAGE 2 — Casting + location moodboards
# ============================================================================

def _rephrase_prompt(original: str) -> str:
    """Ask Claude to reword an image prompt using neutral/architectural
    language without changing the scene content. Used when a prompt is
    rejected by an image model's content filter."""
    return claude_text(
        user=(
            "Rewrite this image-generation prompt so it passes content-policy "
            "filters. Do NOT change the subject, location, objects, or visual "
            "content — only reword using neutral, architectural, or clinical "
            "language. Avoid anything that sounds graphic, medical-emergency, "
            "or disturbing. Return ONLY the rewritten prompt. No explanation.\n\n"
            f"Original prompt:\n{original}"
        ),
        model="claude-haiku-4-5-20251001",
    )


# ── Unified generation helpers with automatic model fallback ─────────────────
#
# _generate_video() and _generate_image() try each model in priority order.
# On any failure (rate limit, content filter, timeout, API error) they log
# and move to the next model. The pipeline never fails because one model
# is having issues.
#
# Reference images and moodboards are passed to every model that accepts
# them so visual consistency is preserved across shots regardless of which
# model ends up generating the clip or frame.

def _generate_video(
    prompt: str,
    first_frame: bytes,
    ref_imgs: list[bytes],
    moodboard: bytes | None,
    duration: int,
    seed: int,
    context: str,
) -> bytes:
    """Generate a video clip with automatic model fallback.

    Attempt order:
      1. SeedDance 2 (ByteDance via Runway) — primary, accepts ref images
      2. Veo 3.1 fast (Google via Runway)   — Runway fallback
      3. LTX 2.3 (Lightricks API)           — non-Runway fallback
      4. Google Veo 3.1 (direct)            — last resort

    Reference images (character refs + optional moodboard) are passed to
    every model that accepts them. Models that don't support refs (Veo via
    Runway, Google Veo, LTX) receive only the first frame for guidance.

    Raises RuntimeError only if every model fails.
    """
    # Build reference list: character refs first, moodboard last
    refs = ref_imgs[:3]
    if moodboard and len(refs) < 3:
        refs = refs + [moodboard]

    # Check Runway daily limit before attempting Runway models
    runway_ok = not _daily_limit_hit.is_set()

    attempts: list[tuple[str, Any]] = []

    if runway_ok:
        attempts.append((
            "seedance",
            lambda: seedance(prompt, first_frame,
                             reference_images=refs or None,
                             duration_seconds=duration, seed=seed),
        ))
        attempts.append((
            "veo",
            lambda: veo(prompt, first_frame,
                        duration_seconds=duration, seed=seed),
        ))

    # Non-Runway fallbacks — always available regardless of Runway limits.
    if os.environ.get("GOOGLE_AI_API_KEY"):
        attempts.append((
            "google_veo",
            lambda: google_veo(prompt, first_frame,
                               duration_seconds=duration,
                               resolution="720p"),
        ))
    # Both LTX models: Pro first (quality), Fast second (speed/fallback).
    if os.environ.get("LTX_API_KEY"):
        attempts.append((
            "ltx-2-3-pro",
            lambda: ltx_video(prompt, first_frame,
                               duration_seconds=duration,
                               resolution="1080p", seed=seed,
                               model=LTX_PRO_MODEL),
        ))
        attempts.append((
            "ltx-2-3-fast",
            lambda: ltx_video(prompt, first_frame,
                               duration_seconds=duration,
                               resolution="1080p", seed=seed,
                               model=LTX_FAST_MODEL),
        ))

    last_exc: Exception | None = None
    for label, fn in attempts:
        try:
            _tprint(f"    [{label}] video {context}")
            result = fn()
            _tprint(f"    ✓ [{label}] {context} ({len(result)//1024}kB)")
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                # Mark Runway as unavailable and skip remaining Runway attempts
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if l not in ("seedance", "veo")]
            _tprint(f"    ✗ [{label}] {context}: {e}")
            last_exc = e

    raise RuntimeError(
        f"All video models failed for {context}. Last: {last_exc}"
    )


def _generate_image_t2i(
    prompt: str,
    size: str = "1792x1024",
    quality: str = "high",
    context: str = "",
) -> bytes:
    """Pure text-to-image. No reference images.

    Used to CREATE the source/canonical images (character references,
    location moodboards, lookbook style frames, frame compositions).
    These outputs are themselves what later identity-locked frames use as
    refs — so this stage cannot rely on having refs available.

    Chain (every model is text-to-image friendly):
      1. gpt_image           Runway / GPT Image 2 (best instruction following)
      2. openai_image        OpenAI direct                (OPENAI_API_KEY)
      3. nano_banana         Runway / Imagen 3 text-to-image
      4. nano_banana*        same, Claude-rephrased prompt
         ↓ Runway daily limit hit, drop Runway-named attempts
      5. google_nano_banana  Gemini 2.5 Flash Image direct (GOOGLE_AI_API_KEY)
      6. reve_create         Reve text-to-image           (REVE_API_KEY)
      7. flux-pro            BFL FLUX.1 Pro                (BFL_API_KEY)

    Raises RuntimeError only if every available model fails.
    """
    runway_ok  = not _daily_limit_hit.is_set()
    rephrased: str | None = None

    def get_rephrased() -> str:
        nonlocal rephrased
        if rephrased is None:
            rephrased = _rephrase_prompt(prompt)
        return rephrased

    attempts: list[tuple[str, Any]] = []

    if runway_ok:
        attempts += [
            ("gpt_image",
             lambda: gpt_image(prompt, size=size, quality=quality)),
            ("nano_banana",
             lambda: nano_banana(prompt[:950])),
            ("nano_banana*",
             lambda: nano_banana(get_rephrased()[:950])),
        ]

    if os.environ.get("OPENAI_API_KEY"):
        attempts.insert(
            1,  # right after gpt_image (Runway)
            ("openai_image",
             lambda: openai_image(prompt, size="1536x1024", quality="medium")),
        )

    if os.environ.get("GOOGLE_AI_API_KEY"):
        attempts.append((
            "google_nano_banana",
            lambda: google_nano_banana(prompt[:950]),
        ))

    if os.environ.get("REVE_API_KEY"):
        attempts.append((
            "reve_create",
            lambda: reve_image(prompt, aspect_ratio="16:9"),
        ))

    if os.environ.get("BFL_API_KEY"):
        attempts.append((
            "flux-pro",
            lambda: flux_image(prompt, width=1344, height=768,
                               model=FLUX_PRO_MODEL),
        ))

    last_exc: Exception | None = None
    for label, fn in attempts:
        try:
            _tprint(f"    [{label}] t2i {context}")
            result = fn()
            _tprint(f"    ✓ [{label}] {context} ({len(result)//1024}kB)")
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if not l.startswith(("gpt_image", "nano_banana"))]
            _tprint(f"    ✗ [{label}] {context}: {e}")
            last_exc = e

    raise RuntimeError(
        f"All t2i image models failed for {context}. Last: {last_exc}"
    )


def _generate_image_with_refs(
    prompt: str,
    refs: list[bytes],
    size: str = "1344x768",
    quality: str = "high",
    context: str = "",
) -> bytes:
    """Text + reference images — identity-locked image generation.

    Used for FRAME generation where character / location identity must
    be preserved from the canonical reference images created by
    _generate_image_t2i(). Every model in the chain is reference-aware
    and uses the refs to lock identity.

    Chain (all models accept and use refs):
      1. gen4_image+refs        Runway Gen4 (purpose-built ID lock)
      2. nano_banana+refs       Runway / Imagen 3 with refs
      3. nano_banana*+refs      same, rephrased prompt
      4. gen4_image_turbo+refs  Runway Gen4 Turbo (cheaper)
         ↓ Runway daily limit hit, drop Runway-named attempts
      5. google_nano_banana+refs Gemini 2.5 Flash Image direct (GOOGLE_AI_API_KEY)
      6. reve_remix+refs        Reve remix, up to 6 refs    (REVE_API_KEY)
      7. flux-2-pro+refs        BFL FLUX.2 multi-ref         (BFL_API_KEY)

    Raises RuntimeError only if every available model fails. If `refs`
    is empty, raises immediately — use _generate_image_t2i() instead.
    """
    if not refs:
        raise RuntimeError(
            "_generate_image_with_refs requires at least one reference image. "
            "Use _generate_image_t2i() for text-only generation."
        )

    runway_ok  = not _daily_limit_hit.is_set()
    rephrased: str | None = None

    def get_rephrased() -> str:
        nonlocal rephrased
        if rephrased is None:
            rephrased = _rephrase_prompt(prompt)
        return rephrased

    attempts: list[tuple[str, Any]] = []

    if runway_ok:
        attempts += [
            ("gen4_image",
             lambda: gen4_image(prompt[:950], reference_images=refs)),
            ("nano_banana",
             lambda: nano_banana(prompt[:950], reference_images=refs)),
            ("nano_banana*",
             lambda: nano_banana(get_rephrased()[:950], reference_images=refs)),
            ("gen4_image_turbo",
             lambda: gen4_image(prompt[:950], reference_images=refs, turbo=True)),
        ]

    if os.environ.get("GOOGLE_AI_API_KEY"):
        attempts.append((
            "google_nano_banana",
            lambda: google_nano_banana(prompt[:950], reference_images=refs),
        ))

    if os.environ.get("REVE_API_KEY"):
        attempts.append((
            "reve_remix",
            lambda: reve_image(prompt, reference_images=refs,
                               aspect_ratio="16:9"),
        ))

    if os.environ.get("BFL_API_KEY"):
        attempts.append((
            "flux-2-pro",
            lambda: flux_image(prompt, reference_images=refs,
                               width=1344, height=768),
        ))

    last_exc: Exception | None = None
    for label, fn in attempts:
        try:
            _tprint(f"    [{label}] image+refs {context}")
            result = fn()
            _tprint(f"    ✓ [{label}] {context} ({len(result)//1024}kB)")
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if not l.startswith(("gen4_image", "nano_banana"))]
            _tprint(f"    ✗ [{label}] {context}: {e}")
            last_exc = e

    raise RuntimeError(
        f"All ref-aware image models failed for {context}. Last: {last_exc}"
    )




def _veo_first_frame(image_prompt: str) -> bytes:
    """Generate the shortest Veo clip and extract its first frame as PNG.
    Used as the final moodboard fallback when image models reject a prompt.
    """
    import subprocess, tempfile
    mp4 = veo(image_prompt, duration_seconds=4, resolution="720p")
    with tempfile.TemporaryDirectory(prefix="autofilm_mb_") as tmp:
        mp4_path = Path(tmp) / "clip.mp4"
        png_path = Path(tmp) / "frame.png"
        mp4_path.write_bytes(mp4)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(mp4_path), "-vframes", "1", "-q:v", "2", str(png_path)],
            check=True,
        )
        return png_path.read_bytes()


def _generate_moodboard(prompt_text: str, slug: str,
                        ref_imgs: list[bytes] | None = None) -> bytes:
    """Generate a location moodboard image.

    Moodboards are themselves reference material — they're created from
    text alone and later passed AS refs into frame generation. So this
    delegates to the t2i function (no refs in the chain). If ref_imgs is
    provided (e.g. a previously generated style frame to maintain
    cross-location visual consistency), it switches to the ref-aware
    chain instead.
    """
    if ref_imgs:
        return _generate_image_with_refs(
            prompt=prompt_text,
            refs=ref_imgs,
            size="1792x1024",
            quality="high",
            context=f"moodboard/{slug}",
        )
    return _generate_image_t2i(
        prompt=prompt_text,
        size="1792x1024",
        quality="high",
        context=f"moodboard/{slug}",
    )


CAST_SYSTEM = """You are a casting director for a virtual film adaptation.
For each character, suggest one currently-working real actor. Provide a
short rationale and an alternative. No deceased actors. No reuse across
roles. Return only via the tool."""

CAST_TOOL_SCHEMA = {
    "description": "Submit casting choices.",
    "input_schema": {
        "type": "object",
        "properties": {
            "casting": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "character_id": {"type": "string"},
                        "actor": {"type": "string"},
                        "rationale": {"type": "string"},
                        "alternative": {"type": "string"},
                    },
                    "required": ["character_id", "actor"],
                },
            },
        },
        "required": ["casting"],
    },
}

LOCATIONS_SYSTEM = """You are a production designer. For each unique
location, write one rich sensory paragraph: real-world inspiration,
light quality, surfaces/materials, palette in 3-5 words, camera-friendly
notes. Group scenes that share a location. Return only via the tool."""

LOCATIONS_TOOL_SCHEMA = {
    "description": "Submit unique locations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "color_palette": {"type": "array", "items": {"type": "string"}},
                        "scene_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["slug", "name", "description", "scene_ids"],
                },
            },
        },
        "required": ["locations"],
    },
}


def cast_and_locations(exp: Experiment, script: dict) -> tuple[list[dict], list[dict]]:
    if exp.has("cast.json") and exp.has("locations.json"):
        return exp.read_json("cast.json"), exp.read_json("locations.json")

    # Casting.
    cast_result = claude_tool(
        system=CAST_SYSTEM,
        user_content=f"Characters:\n{json.dumps(script['characters'], indent=2)}\n\nCast them.",
        tool_name="submit_casting",
        tool_schema=CAST_TOOL_SCHEMA,
        max_tokens=6000,
    )
    cast = cast_result["casting"]
    exp.write_json("cast.json", cast)

    # Locations.
    loc_result = claude_tool(
        system=LOCATIONS_SYSTEM,
        user_content=f"Scenes:\n{json.dumps(script['scenes'], indent=2)}\n\nDescribe unique locations.",
        tool_name="submit_locations",
        tool_schema=LOCATIONS_TOOL_SCHEMA,
        max_tokens=6000,
    )
    locations = loc_result["locations"]

    # Generate ONE moodboard per location via Nano Banana 2 (it has the
    # world knowledge to render real-world locations accurately).
    # All locations are independent — run in parallel.
    def _do_moodboard(loc: dict) -> None:
        slug = loc["slug"]
        out_path = exp.path(f"location_moodboards/{slug}/00.png")
        mb_prompt = (
            f"Cinematic empty-scene location reference photograph, no people. "
            f"{loc['description']}. "
            f"Palette: {', '.join(loc.get('color_palette', []))}. "
            f"Anamorphic, photorealistic, no text/logos."
        )
        if not out_path.exists():
            try:
                img = _generate_moodboard(mb_prompt, slug)
                out_path.write_bytes(img)
            except RuntimeError as e:
                _tprint(f"  ⚠ All moodboard attempts exhausted for {slug}: {e}")
        exp.log_prompt(
            target=f"location_moodboards/{slug}/00.png",
            model=NANO_BANANA_MODEL,
            prompt=mb_prompt,
            stage="moodboard",
        )
        loc["moodboard_paths"] = [str(p) for p in out_path.parent.glob("*.png")]

    _tprint(f"  Generating {len(locations)} moodboard(s) "
            f"({'parallel' if MAX_WORKERS > 1 else 'serial'}, "
            f"MAX_WORKERS={MAX_WORKERS})...")
    _parallel_run("moodboard", locations, _do_moodboard)

    exp.write_json("locations.json", locations)
    return cast, locations


# ============================================================================
# STAGE 3 — Look book (lock visual style)
# ============================================================================

LOOKBOOK_PROMPT = """You are the cinematographer locking the look book.
Given the script summary and locations, output the visual bible.

If a director or cinematographer is named in the input, treat them as
creative-direction input — translate their working sensibility into
CONCRETE CRAFT MARKERS:
  - typical lens package (focal lengths, anamorphic vs spherical)
  - lighting approach (sources, key/fill ratios, motivated practical?)
  - color palette and grade direction
  - framing tendencies (symmetry, headroom, negative space)
  - camera-movement vocabulary (locked-off, slow push-in, handheld, etc.)

Bake these markers into lens_package, lighting_style, grade_description,
and style_frame_prompt so they propagate. Cite their films in
reference_films with attribution. Do not instruct the downstream image
or video models to "imitate" the named artist — instead encode the
craft decisions you'd derive from studying their work.

If no director or DP is named, choose a neutral cinematic baseline
appropriate to the genre and era.

Output:
  - era, genre, tone (one line each)
  - lens_package (specific, period-appropriate)
  - lighting_style (one paragraph)
  - grade_description (one sentence)
  - reference_films (2-3 cited DPs/films, with attribution)
  - style_frame_prompt (a long prompt for an iconic film still)

Return only via the tool."""

LOOKBOOK_TOOL_SCHEMA = {
    "description": "Submit the locked look book.",
    "input_schema": {
        "type": "object",
        "properties": {
            "era": {"type": "string"},
            "genre": {"type": "string"},
            "tone": {"type": "string"},
            "lens_package": {"type": "string"},
            "lighting_style": {"type": "string"},
            "grade_description": {"type": "string"},
            "reference_films": {"type": "array", "items": {"type": "string"}},
            "style_frame_prompt": {"type": "string"},
        },
        "required": ["era", "genre", "tone", "lens_package", "lighting_style",
                     "grade_description", "style_frame_prompt"],
    },
}


def build_lookbook(exp: Experiment, script: dict, locations: list[dict]) -> dict:
    if exp.has("lookbook.json"):
        return exp.read_json("lookbook.json")

    payload: dict[str, Any] = {
        "title": script["title"],
        "scenes_summary": [
            {"id": s["id"], "location": s["location"], "mood": s.get("mood", "")}
            for s in script["scenes"][:8]
        ],
        "locations": [{"slug": l["slug"], "description": l["description"][:200]}
                      for l in locations[:6]],
    }
    # Optional creative-direction inputs. When set, Claude derives concrete
    # craft markers in the lookbook stage; the downstream image/video
    # prompts use those markers (not the names themselves).
    if DIRECTOR:
        payload["director"] = DIRECTOR
    if CINEMATOGRAPHER:
        payload["cinematographer"] = CINEMATOGRAPHER

    lookbook = claude_tool(
        system=LOOKBOOK_PROMPT,
        user_content=f"Film context:\n{json.dumps(payload, indent=2)}\n\nLock the look book.",
        tool_name="submit_lookbook",
        tool_schema=LOOKBOOK_TOOL_SCHEMA,
    )

    # Bake in our hard-locked style keywords + ffmpeg grade + credits.
    lookbook["style_keywords"] = LOOKBOOK_STYLE_KEYWORDS
    lookbook["ffmpeg_grade"] = LOOKBOOK_GRADE
    if DIRECTOR:
        lookbook["director"] = DIRECTOR
    if CINEMATOGRAPHER:
        lookbook["cinematographer"] = CINEMATOGRAPHER

    # Render the iconic style frame.
    try:
        sf = _generate_image_t2i(
            prompt=lookbook["style_frame_prompt"],
            size="1792x1024",
            quality="high",
            context="lookbook/style_frame",
        )
        exp.write_bytes("lookbook/style_frame.png", sf)
    except Exception as e:  # noqa: BLE001
        print(f"  Style frame failed: {e}")
    exp.log_prompt(
        target="lookbook/style_frame.png",
        model=GPT_IMAGE_MODEL,
        prompt=lookbook["style_frame_prompt"],
        stage="style_frame",
        size="1792x1024",
        quality="high",
    )

    exp.write_json("lookbook.json", lookbook)
    return lookbook


def style_preamble(lookbook: dict) -> str:
    """Condensed style block prepended to every image/video prompt."""
    return (
        f"VISUAL STYLE: {', '.join(lookbook.get('style_keywords', []))}.\n"
        f"LENS: {lookbook.get('lens_package', '')}.\n"
        f"GRADE: {lookbook.get('grade_description', '')}.\n\n"
    )


# ============================================================================
# STAGE 4 — Per-scene per-character reference images (identity locking)
# ============================================================================

# ============================================================================
# STAGE 4 — Per-scene per-character reference images
# ============================================================================
# Virtual actors only — no real celebrities, no web photo searches, no FLUX.
# Each reference image is generated from the character's physical description
# as written in the source book. GPT Image 2 (standard tier) is the primary
# model; nano_banana is the fallback.
#
# The prompt is intentionally compact (<900 chars) to stay within Runway's
# promptText limit and to keep generation fast and focused.

def _reference_prompt(character: dict, scene: dict, lookbook: dict) -> str:
    """Build a reference image prompt from the character's book description.

    Deliberately compact — Runway's text_to_image API rejects prompts
    longer than 1000 characters. Keeps under 900 to leave headroom.
    """
    style = ", ".join(lookbook.get("style_keywords", []))[:120]
    desc  = character.get("description", "")[:300]
    name  = character.get("name", "the character")
    loc   = scene.get("location", "")[:80]
    tod   = scene.get("time_of_day", "day")
    mood  = scene.get("mood", "")[:60]

    return (
        f"Photorealistic cinematic film still. {style}. "
        f"Anamorphic 16:9, shallow depth of field.\n\n"
        f"SUBJECT: {name} — {desc}\n"
        f"SCENE: {loc}, {tod}. Mood: {mood}.\n\n"
        f"NEGATIVE: no text, no logos, no watermarks, no UI."
    )


def build_references(exp: Experiment, script: dict, cast: list[dict],
                      locations: list[dict], lookbook: dict) -> dict:
    """Generate one reference image per character per scene.

    Uses the character's physical description from the source book rather
    than a real actor's name or photo. This avoids:
      - Celebrity likeness / rights issues
      - FLUX API reliability problems
      - DuckDuckGo rate-limits and zero-result searches

    Pipeline (single step, no composition+lock):
      1. gpt_image(quality="standard")  — 2K tier, ~$0.10/image
      2. nano_banana                    — fallback if GPT Image 2 fails
    """
    char_by_id   = {c["id"]: c for c in script["characters"]}
    loc_by_scene: dict[str, dict] = {}
    for loc in locations:
        for sid in loc.get("scene_ids", []):
            loc_by_scene[sid] = loc

    manifest: dict[str, dict[str, str]] = {}
    work_items = []
    for scene in script["scenes"]:
        scene_id = scene["id"]
        manifest.setdefault(scene_id, {})
        for cid in scene.get("characters", []):
            if cid in char_by_id:
                work_items.append((scene, cid))

    def _do_reference(item: tuple) -> tuple[str, str, str] | None:
        scene, cid = item
        scene_id  = scene["id"]
        character = char_by_id[cid]
        out_path  = exp.path(f"references/{cid}/{scene_id}.png")

        if out_path.exists():
            return scene_id, cid, str(out_path)

        ref_prompt = _reference_prompt(character, scene, lookbook)

        # Load moodboard for this scene if available
        scene_moodboard: bytes | None = None
        scene_loc = loc_by_scene.get(scene_id)
        if scene_loc:
            mb_paths = scene_loc.get("moodboard_paths", [])
            if mb_paths and Path(mb_paths[0]).exists():
                scene_moodboard = Path(mb_paths[0]).read_bytes()

        try:
            img = _generate_image_t2i(
                prompt=ref_prompt,
                size="1344x768",
                quality="standard",
                context=f"reference {cid}/{scene_id}",
            )
            out_path.write_bytes(img)
            exp.log_prompt(
                target=f"references/{cid}/{scene_id}.png",
                model=GPT_IMAGE_MODEL,
                prompt=ref_prompt,
                stage="reference",
                character=cid, scene=scene_id,
            )
            return scene_id, cid, str(out_path)
        except RuntimeError as e:
            _tprint(f"  ⚠ Reference failed for {cid}/{scene_id}: {e}")
            return None

    _tprint(f"  Generating {len(work_items)} reference image(s) "
            f"({'parallel' if MAX_WORKERS > 1 else 'serial'}, "
            f"MAX_WORKERS={MAX_WORKERS})...")
    for result in _parallel_run("reference", work_items, _do_reference):
        if result:
            scene_id, cid, path = result
            manifest[scene_id][cid] = path

    exp.write_json("references_manifest.json", manifest)
    return manifest



# ============================================================================
# STAGE 5 — Shot list per scene
# ============================================================================

SHOTLIST_SYSTEM = """You are the director and cinematographer. Break each
scene into 3-6 shots. For each shot specify:
  - shot_id, shot_size (ECU/CU/MCU/MS/MLS/LS/XLS), angle, camera_move,
    lens_mm, subject, action (terse, present tense), dialogue_excerpt
    (or empty), duration_seconds, composition_notes.

Duration guidance — HARD RULE:
  duration_seconds MUST be one of {4, 6, 8}. The video model's native
  single-call cap is 8 seconds; longer durations are not available.
  If a beat needs more than 8 seconds on screen, BREAK IT INTO
  MULTIPLE SHOTS — different size, different angle, motivated cut.
  This is how working features cover long beats anyway.

Pick shorter durations (4s) for inserts, reaction beats, and quick
coverage. Use 6s for standard medium shots and most dialogue. Reserve
8s for the strongest single moments — a long line, a held emotional
beat, a slow camera move that needs time to breathe. Modern cinema
averages ~3-5 second shots; don't default to 8s for everything.

Mix sizes and angles for visual variety. Motivate every camera move.

TRANSITIONS — `transition_out` field on each shot:
""" + transitions_prompt_guidance() + """

Return only via the tool."""

SHOTLIST_TOOL_SCHEMA = {
    "description": "Submit shot list for one scene.",
    "input_schema": {
        "type": "object",
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "shot_id": {"type": "string"},
                        "shot_size": {"type": "string"},
                        "angle": {"type": "string"},
                        "camera_move": {"type": "string"},
                        "lens_mm": {"type": "integer"},
                        "subject": {"type": "string"},
                        "action": {"type": "string"},
                        "dialogue_excerpt": {"type": "string"},
                        "duration_seconds": {
                            "type": "integer",
                            "enum": [4, 6, 8],
                            "description": "Must be 4, 6, or 8 seconds (Veo 3.1 cap).",
                        },
                        "composition_notes": {"type": "string"},
                        "transition_out": {
                            "type": "object",
                            "description": (
                                "How to transition from this shot to the NEXT shot. "
                                "Default cut. The transition on the LAST shot of "
                                "the LAST scene is ignored."
                            ),
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": transition_names(),
                                },
                                "duration": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 2.0,
                                    "description": "Seconds; ignored when type is cut.",
                                },
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["shot_id", "shot_size", "angle", "camera_move",
                                 "subject", "action", "duration_seconds", "composition_notes"],
                },
            },
        },
        "required": ["shots"],
    },
}


def shot_list_for_scene(scene: dict) -> list[dict]:
    return claude_tool(
        system=SHOTLIST_SYSTEM,
        user_content=f"Scene:\n{json.dumps(scene, indent=2)}\n\nBreak it into shots.",
        tool_name="submit_shotlist",
        tool_schema=SHOTLIST_TOOL_SCHEMA,
    )["shots"]


def build_storyboard(exp: Experiment, script: dict) -> dict:
    if exp.has("storyboard.json"):
        return exp.read_json("storyboard.json")
    storyboard: dict[str, list[dict]] = {}
    for scene in script["scenes"]:
        storyboard[scene["id"]] = shot_list_for_scene(scene)
    exp.write_json("storyboard.json", storyboard)

    # Compute and persist the routing plan immediately. This is the
    # single source of truth for which model handles which shot, used by
    # both the bible PDF and the video stage.
    plan = plan_shot_durations(storyboard)
    exp.write_json("shot_plan.json", plan)
    agg = plan.get("_aggregate", {})
    print(f"  → shot plan: {agg.get('total_seconds', 0)}s total, "
          f"~${agg.get('estimated_cost_usd', 0):.2f} estimated render cost")
    return storyboard


# ============================================================================
# STAGE 6 — Music per scene
# ============================================================================

def build_music(exp: Experiment, script: dict, storyboard: dict) -> None:
    """Generate one Stability music cue per scene.

    Duration is calculated from the storyboard so the cue spans the full
    scene (all shots) rather than a hardcoded 30 seconds. A small buffer
    is added so music can bleed naturally across the cut into the next
    scene without an abrupt silence. Stable Audio caps at 47 seconds;
    compile_final will loop any cue shorter than the scene if needed.

    ``storyboard`` maps scene_id → list of shot dicts with
    ``duration_seconds`` fields.
    """
    # Maximum Stable Audio can generate in one call.
    _STABLE_AUDIO_MAX_SECONDS = 47
    # Extra seconds to pad beyond the shot-sum so music bleeds across the
    # cut to the next shot / scene without running dry.
    _CROSSSHOT_BUFFER_SECONDS = 6

    def _scene_duration(scene_id: str) -> int:
        shots = storyboard.get(scene_id, [])
        shot_sum = sum(s.get("duration_seconds", SHOT_DURATION_SECONDS) for s in shots)
        # Pad for crossfades and cross-shot bleed; cap at Stable Audio max.
        return min(shot_sum + _CROSSSHOT_BUFFER_SECONDS, _STABLE_AUDIO_MAX_SECONDS)

    def _do_music(scene: dict) -> None:
        scene_id  = scene["id"]
        out_path  = exp.path(f"music/{scene_id}.wav")
        duration  = _scene_duration(scene_id)
        music_prompt = (
            f"{MUSIC_STYLE}. Scene mood: {scene.get('mood', '')}. "
            f"{scene.get('summary', '')[:150]}"
        )[:450]   # Stability caps at ~450 chars
        if not out_path.exists():
            try:
                audio = stable_audio(music_prompt, duration_seconds=duration)
                out_path.write_bytes(audio)
            except Exception as e:  # noqa: BLE001
                _tprint(f"  Music {scene_id} failed: {e}")
        exp.log_prompt(
            target=f"music/{scene_id}.wav",
            model="stable-audio-2.5",
            prompt=music_prompt,
            stage="music",
            scene=scene_id,
            duration_seconds=duration,
        )

    _tprint(f"  Generating {len(script['scenes'])} music cue(s) "
            f"({'parallel' if MAX_WORKERS > 1 else 'serial'})...")
    _parallel_run("music", script["scenes"], _do_music)


# ============================================================================
# STAGE 7 — First frames per shot
# ============================================================================

def first_frame_prompt(lookbook: dict, shot: dict, scene: dict,
                        chars_in_shot: list[dict], actors_in_shot: list[str],
                        location: dict | None) -> str:
    char_lines = "\n".join(
        f"  - {c['name']} (played by {a}): {c.get('description', '')}"
        for c, a in zip(chars_in_shot, actors_in_shot)
    ) or "  (no on-screen characters)"
    loc_text = location["description"][:300] if location else scene["location"]
    return (
        style_preamble(lookbook)
        + f"PHOTOREALISTIC CINEMATIC FILM STILL.\n\n"
        f"SHOT: {shot['shot_size']} ({shot['angle']} angle), ~{shot.get('lens_mm', 35)}mm lens.\n"
        f"CAMERA: starts {shot.get('camera_move', 'static')}.\n\n"
        f"LOCATION: {scene['location']}. {loc_text}\n"
        f"TIME: {scene.get('time_of_day', 'day')}\n\n"
        f"CHARACTERS:\n{char_lines}\n\n"
        f"ACTION: {shot['action']}\n"
        f"COMPOSITION: {shot['composition_notes']}\n"
        f"MOOD: {scene.get('mood', '')}\n\n"
        f"NEGATIVE: no text, no logos, no UI, no watermarks, no illustration."
    )


def build_first_frames(exp: Experiment, script: dict, cast: list[dict],
                        locations: list[dict], lookbook: dict, storyboard: dict) -> dict:
    char_by_id = {c["id"]: c for c in script["characters"]}
    actor_by_char = {row["character_id"]: row["actor"] for row in cast}
    loc_by_scene = {sid: l for l in locations for sid in l.get("scene_ids", [])}

    manifest: dict[str, dict[str, str]] = {}

    # Collect all (scene, shot) work items — each is independent.
    work_items = []
    for scene in script["scenes"]:
        scene_id = scene["id"]
        manifest.setdefault(scene_id, {})
        chars = [char_by_id[c] for c in scene.get("characters", []) if c in char_by_id]
        actors = [actor_by_char.get(c["id"], c["name"]) for c in chars]
        for shot in storyboard.get(scene_id, []):
            work_items.append((scene, shot, chars, actors))

    def _do_first_frame(item: tuple) -> tuple[str, str, str] | None:
        scene, shot, chars, actors = item
        scene_id = scene["id"]
        shot_id  = shot["shot_id"]
        out_path = exp.path(f"frames/{scene_id}/{shot_id}.png")
        if out_path.exists():
            return scene_id, shot_id, str(out_path)

        # Full prompt for gpt_image (no character limit).
        ff_prompt = first_frame_prompt(
            lookbook, shot, scene, chars, actors, loc_by_scene.get(scene_id)
        )

        # Compact prompt for nano_banana — Runway rejects promptText >1000
        # chars. The full ff_prompt can far exceed this (style_preamble +
        # location description + per-character descriptions).
        char_summary = "; ".join(c["name"] for c in chars) or "no characters"
        nano_prompt = (
            f"Photorealistic cinematic film still. "
            f"{', '.join(lookbook.get('style_keywords', []))[:80]}. "
            f"Anamorphic 16:9.\n"
            f"SHOT: {shot.get('shot_size', '')} {shot.get('angle', '')} "
            f"angle, {shot.get('lens_mm', 35)}mm.\n"
            f"LOCATION: {scene['location'][:80]}, "
            f"{scene.get('time_of_day', 'day')}.\n"
            f"CHARACTERS: {char_summary[:120]}\n"
            f"ACTION: {shot.get('action', '')[:150]}\n"
            f"MOOD: {scene.get('mood', '')[:60]}\n"
            f"NEGATIVE: no text, no logos, no watermarks."
        )[:950]

        # ── Step 1: Compose the frame ─────────────────────────────────
        composition: bytes | None = None
        composition_model = "auto"

        # Pull moodboard for this scene if available
        scene_moodboard: bytes | None = None
        scene_loc = loc_by_scene.get(scene_id)
        if scene_loc:
            mb_paths = scene_loc.get("moodboard_paths", [])
            if mb_paths and Path(mb_paths[0]).exists():
                scene_moodboard = Path(mb_paths[0]).read_bytes()

        try:
            composition = _generate_image_t2i(
                prompt=ff_prompt,
                size="1792x1024",
                quality="high",
                context=f"frame {scene_id}/{shot_id}",
            )
        except RuntimeError as e:
            _tprint(f"  ⚠ All image models failed for {scene_id}/{shot_id}: {e}")

        if composition is None:
            _tprint(f"  ⚠ All composition attempts failed for {scene_id}/{shot_id}")
            return None

        exp.log_prompt(
            target=f"frames/{scene_id}/{shot_id}.composition.png",
            model=composition_model,
            prompt=ff_prompt,
            stage="first_frame_composition",
            scene=scene_id, shot=shot_id,
        )

        # ── Step 2: Lock identity with character reference images ──────
        ref_imgs = []
        for c in chars:
            rp = exp.path(f"references/{c['id']}/{scene_id}.png")
            if rp.exists():
                ref_imgs.append(rp.read_bytes())

        # Include scene moodboard as the last ref slot for color/atmosphere
        # consistency. Total still <= 3 for Runway models (chars first).
        all_refs = ref_imgs[:2]
        if scene_moodboard and len(all_refs) < 3:
            all_refs.append(scene_moodboard)

        if all_refs:
            try:
                final = _generate_image_with_refs(
                    prompt=nano_prompt,       # compact prompt for Runway models
                    refs=all_refs,
                    size="1344x768",
                    quality="standard",
                    context=f"frame_lock {scene_id}/{shot_id}",
                )
            except RuntimeError as e:
                _tprint(f"    ⚠ Lock failed for {scene_id}/{shot_id}: {e}  (using composition)")
                final = composition
        else:
            final = composition

        out_path.write_bytes(final)
        return scene_id, shot_id, str(out_path)

    _tprint(f"  Generating {len(work_items)} first frame(s) "
            f"({'parallel' if MAX_WORKERS > 1 else 'serial'}, "
            f"MAX_WORKERS={MAX_WORKERS})...")
    for result in _parallel_run("first_frame", work_items, _do_first_frame):
        if result:
            scene_id, shot_id, path = result
            manifest[scene_id][shot_id] = path

    exp.write_json("frames_manifest.json", manifest)
    return manifest


# ============================================================================
# STAGE 8 — Image-to-video, N takes per shot
# ============================================================================

# Per-take performance variations. Edit these to change the editor's options.
TAKE_VARIATIONS = [
    "",
    "Slightly more restrained performance, less movement.",
    "Slightly more energy, more expressive face.",
]


def veo_prompt(lookbook: dict, shot: dict, scene: dict,
                chars_in_shot: list[dict], actors_in_shot: list[str],
                take_index: int) -> str:
    move = {
        "static": "locked-off camera",
        "pan": "smooth pan",
        "tilt": "smooth tilt",
        "track": "tracking shot",
        "dolly": "slow dolly push-in",
        "crane": "rising crane move",
        "handheld": "subtle handheld",
    }.get(shot.get("camera_move", "static"), "natural motion")

    char_block = ""
    if chars_in_shot:
        char_lines = [f"{c['name']} (played by {a})"
                      for c, a in zip(chars_in_shot, actors_in_shot)]
        char_block = f"CHARACTERS: {', '.join(char_lines)}.\n"

    dialogue_block = ""
    if shot.get("dialogue_excerpt"):
        speaker = chars_in_shot[0]["name"] if chars_in_shot else "the character"
        actor = actors_in_shot[0] if actors_in_shot else ""
        dialogue_block = (
            f"DIALOGUE (synchronized native audio in {actor}'s voice): "
            f"{speaker}: \"{shot['dialogue_excerpt']}\"\n"
        )

    take_var = TAKE_VARIATIONS[take_index] if take_index < len(TAKE_VARIATIONS) else ""

    return (
        f"STYLE: {', '.join(lookbook.get('style_keywords', []))}.\n"
        f"{move}.\n"
        f"{char_block}"
        f"ACTION: {shot['action']}\n"
        f"{dialogue_block}"
        f"AMBIENT: {scene.get('mood', '')} mood, {scene['location']}, "
        f"{scene.get('time_of_day', 'day')}.\n"
        f"AUDIO: Natural ambient sound and dialogue only. "
        f"NO background music. NO score. NO soundtrack. NO musical instruments.\n"
        f"{take_var}\n"
        f"24fps, anamorphic, fine grain."
    )


def _render_shot(
    route: dict,
    prompt: str,
    first_frame: bytes,
    reference_images: list[bytes],
    seed: int | None = None,
) -> bytes:
    """Render one take. Always a single Veo call now that the shot ceiling
    is 8 seconds — chaining/extension are not used."""
    return veo(
        prompt=prompt,
        first_frame=first_frame,
        reference_images=reference_images,
        model=route["model_id"],
        duration_seconds=route["segments"][0],
        seed=seed,
    )


def build_video(exp: Experiment, script: dict, cast: list[dict],
                 lookbook: dict, storyboard: dict) -> dict:
    char_by_id = {c["id"]: c for c in script["characters"]}
    actor_by_char = {row["character_id"]: row["actor"] for row in cast}
    scene_by_id = {s["id"]: s for s in script["scenes"]}

    # Load the routing plan written by build_storyboard. If missing
    # (e.g. an older experiment), recompute on the fly.
    if exp.has("shot_plan.json"):
        shot_plan = exp.read_json("shot_plan.json")
    else:
        shot_plan = plan_shot_durations(storyboard)
        exp.write_json("shot_plan.json", shot_plan)

    manifest: dict[str, dict[str, list[str]]] = {}

    # Collect all (scene_id, shot, take_idx) work items — every combination
    # is independent once first frames exist. Pre-populate the manifest
    # with cached takes so _parallel_run workers don't need to write it.
    work_items = []
    for scene_id, shots in storyboard.items():
        manifest.setdefault(scene_id, {})
        scene = scene_by_id[scene_id]
        chars = [char_by_id[c] for c in scene.get("characters", []) if c in char_by_id]
        actors = [actor_by_char.get(c["id"], c["name"]) for c in chars]

        for shot in shots:
            shot_id = shot["shot_id"]
            manifest[scene_id].setdefault(shot_id, [])
            ff = exp.path(f"frames/{scene_id}/{shot_id}.png")
            if not ff.exists():
                continue

            ref_imgs = [
                exp.path(f"references/{c['id']}/{scene_id}.png").read_bytes()
                for c in chars
                if exp.path(f"references/{c['id']}/{scene_id}.png").exists()
            ]
            route = shot_plan.get(scene_id, {}).get(shot_id) or route_shot(
                shot.get("duration_seconds", 8), tier=VEO_TIER
            )
            model_key = route["model_key"]
            total_dur  = sum(route["segments"])
            _tprint(f"  {scene_id}/{shot_id}: routing → {model_key} "
                    f"({total_dur}s, {len(route['segments'])} segment(s))")

            for take_idx in range(TAKES_PER_SHOT):
                take_path = exp.path(f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4")
                if take_path.exists():
                    manifest[scene_id][shot_id].append(str(take_path))
                    continue
                work_items.append((scene_id, shot, take_idx, ff, ref_imgs, route,
                                   scene, chars, actors))

    def _do_take(item: tuple) -> tuple[str, str, str] | None:
        scene_id, shot, take_idx, ff, ref_imgs, route, scene, chars, actors = item
        shot_id   = shot["shot_id"]
        take_path = exp.path(f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4")
        vp = veo_prompt(lookbook, shot, scene, chars, actors, take_idx)
        exp.log_prompt(
            target=f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4",
            model=route["model_id"],
            prompt=vp,
            stage="video",
            scene=scene_id, shot=shot_id, take=take_idx + 1,
            duration_seconds=route["segments"][0],
            estimated_cost=route["estimated_cost"],
        )
        try:
            video_bytes = _generate_video(
                prompt=vp,
                first_frame=ff.read_bytes(),
                ref_imgs=ref_imgs,
                moodboard=None,          # moodboard handled via ref_imgs in stage 4
                duration=route["segments"][0],
                seed=exp.seed + take_idx * 137,
                context=f"{scene_id}/{shot_id}/take_{take_idx + 1}",
            )
            take_path.write_bytes(video_bytes)
            _tprint(f"  ✓ take {scene_id}/{shot_id}/{take_idx + 1} "
                    f"({len(video_bytes)//1024}kB)")
            return scene_id, shot_id, str(take_path)
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, f"video/{scene_id}/{shot_id}/take_{take_idx+1}")
                return None
            _tprint(f"  Take {scene_id}/{shot_id}/{take_idx + 1} failed: {e}")
            return None

    _tprint(f"  Generating {len(work_items)} take(s) "
            f"({'parallel' if MAX_WORKERS > 1 else 'serial'}, "
            f"MAX_WORKERS={MAX_WORKERS})...")
    for result in _parallel_run("video_take", work_items, _do_take):
        if result:
            scene_id, shot_id, path = result
            manifest[scene_id][shot_id].append(path)

    # Sort takes within each shot into deterministic order.
    for scene_id in manifest:
        for shot_id in manifest[scene_id]:
            manifest[scene_id][shot_id].sort()

    exp.write_json("clips_manifest.json", manifest)
    return manifest


# ============================================================================
# STAGE 9 — Edit decisions (pick best take)
# ============================================================================

EDIT_SYSTEM = """You are a film editor making the first cut. Per shot,
pick the strongest take and (optionally) trim points. Priorities:
performance, framing, continuity, technical quality. Keep cuts tight.

You can also OVERRIDE the storyboard's transition for any shot by
emitting a transition_out field. Most of the time the planned
transition is correct; override only when the chosen take genuinely
calls for something different (e.g. the take's tail is dead air, so a
fade through black would feel forced — switch to a hard cut). Leaving
transition_out unset preserves whatever the storyboard specified.

Return only via the tool."""

EDIT_TOOL_SCHEMA = {
    "description": "Submit edit decision list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_id": {"type": "string"},
                        "shot_id": {"type": "string"},
                        "chosen_take": {"type": "integer"},
                        "in_seconds": {"type": "number"},
                        "out_seconds": {"type": "number"},
                        "rationale": {"type": "string"},
                        "transition_out": {
                            "type": "object",
                            "description": (
                                "Optional: override the storyboard's transition "
                                "for this shot. Omit to inherit from the storyboard."
                            ),
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": transition_names(),
                                },
                                "duration": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 2.0,
                                },
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["scene_id", "shot_id", "chosen_take"],
                },
            },
        },
        "required": ["decisions"],
    },
}


def _storyboard_transition_for(storyboard: dict, scene_id: str, shot_id: str) -> dict | None:
    """Look up a shot's planned transition_out from the storyboard. None if
    the shot wasn't found or had no transition specified."""
    for shot in storyboard.get(scene_id, []):
        if shot["shot_id"] == shot_id:
            return shot.get("transition_out")
    return None


def _merge_transition_into_decisions(decisions: list[dict], storyboard: dict) -> list[dict]:
    """For each EDL decision, fill in transition_out from the storyboard
    if the editor didn't already override it. Validates the result so
    downstream code never has to re-validate."""
    for d in decisions:
        if "transition_out" not in d or not d["transition_out"]:
            planned = _storyboard_transition_for(storyboard, d["scene_id"], d["shot_id"])
            if planned:
                d["transition_out"] = planned
        # Normalize whatever ended up in the slot.
        d["transition_out"] = validate_transition(d.get("transition_out"))
    return decisions


def build_edl(exp: Experiment, storyboard: dict, clips_manifest: dict) -> dict:
    if exp.has("edl.json"):
        return exp.read_json("edl.json")

    # Single-take fast path.
    max_takes = max((len(t) for s in clips_manifest.values() for t in s.values()),
                    default=0)
    if max_takes <= 1:
        decisions = [
            {"scene_id": sid, "shot_id": shid, "chosen_take": 1,
             "rationale": "Single take generated."}
            for sid, shots in clips_manifest.items()
            for shid in shots.keys()
        ]
        decisions = _merge_transition_into_decisions(decisions, storyboard)
        edl = {"decisions": decisions}
        exp.write_json("edl.json", edl)
        return edl

    # Vision-based selection: extract one frame per take, ask Claude to pick.
    all_decisions = []
    for scene_id, shots in storyboard.items():
        scene_takes = clips_manifest.get(scene_id, {})
        if not scene_takes:
            continue
        content: list[dict] = [
            {"type": "text", "text": f"SCENE {scene_id}\nPick the best take per shot.\n"},
        ]
        for shot in shots:
            shot_id = shot["shot_id"]
            takes = scene_takes.get(shot_id, [])
            if not takes:
                continue
            content.append({"type": "text", "text": (
                f"\n--- SHOT {shot_id} ---\nAction: {shot['action']}\n"
                f"Dialogue: {shot.get('dialogue_excerpt') or '(none)'}\nTakes:\n"
            )})
            for i, take_path in enumerate(takes):
                try:
                    frame = extract_video_frame(Path(take_path))
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64", "media_type": "image/png",
                            "data": base64.b64encode(frame).decode(),
                        },
                    })
                    content.append({"type": "text", "text": f"  ^ {shot_id} take {i + 1}"})
                except Exception as e:  # noqa: BLE001
                    content.append({"type": "text", "text": f"  (could not extract frame for take {i + 1}: {e})"})
        try:
            decisions = claude_tool(
                system=EDIT_SYSTEM,
                user_content=content,
                tool_name="submit_edl",
                tool_schema=EDIT_TOOL_SCHEMA,
                max_tokens=6000,
            )["decisions"]
            all_decisions.extend(decisions)
        except Exception as e:  # noqa: BLE001
            # Fallback: pick take 1 for everything in this scene.
            for shot in shots:
                if shot["shot_id"] in scene_takes:
                    all_decisions.append({
                        "scene_id": scene_id, "shot_id": shot["shot_id"],
                        "chosen_take": 1, "rationale": f"Fallback ({e})",
                    })

    all_decisions = _merge_transition_into_decisions(all_decisions, storyboard)
    edl = {"decisions": all_decisions}
    exp.write_json("edl.json", edl)
    return edl


# ============================================================================
# STAGE 10 — Final compile, color grade, sound mix
# ============================================================================

def _trim_clip(src: Path, dst: Path, in_s: float, out_s: float | None) -> Path:
    """Write a re-encoded copy of `src` trimmed to [in_s, out_s]. Re-encode
    rather than stream-copy because xfade/concat downstream require
    consistent SAR/PTS that stream-copy may not produce on Veo outputs.

    Skips work if `dst` already exists and is non-empty (resumability).
    """
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["-i", str(src)]
    if in_s and in_s > 0:
        args = ["-ss", str(in_s)] + args
    if out_s and out_s > (in_s or 0):
        args += ["-t", str(out_s - (in_s or 0))]
    args += [
        "-c:v", "libx264", "-preset", "medium", "-b:v", "8000k",
        "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(dst),
    ]
    ffmpeg(args)
    return dst


def _resolution_dims() -> tuple[int, int]:
    """(width, height) for compile output, derived from VEO_RESOLUTION."""
    res = os.getenv("VEO_RESOLUTION", "720p").lower()
    return (1920, 1080) if res == "1080p" else (1280, 720)


def compile_final(exp: Experiment, script: dict, storyboard: dict,
                   clips_manifest: dict, edl: dict, lookbook: dict) -> Path:
    # moviepy v1 uses moviepy.editor; v2 removed it (everything at top level
    # with renamed methods). We pin to <2.0.0 in pyproject.toml but guard
    # here so a mis-installed v2 gives a clear error rather than a traceback.
    try:
        from moviepy.editor import (
            AudioFileClip, CompositeAudioClip, VideoFileClip,
            concatenate_videoclips, afx,
        )
        _moviepy_v1 = True
    except ModuleNotFoundError:
        # v2 fallback — import the renamed equivalents.
        from moviepy import (  # type: ignore[no-redef]
            AudioFileClip, CompositeAudioClip, VideoFileClip,
            concatenate_videoclips,
        )
        import moviepy.audio.fx as afx  # type: ignore[no-redef]
        _moviepy_v1 = False
        import warnings
        warnings.warn(
            "moviepy v2 detected. Pin 'moviepy<2.0.0' in pyproject.toml and "
            "run 'uv sync' to avoid API incompatibilities.",
            stacklevel=2,
        )

    def _subclip(clip, t_start, t_end=None):
        """moviepy v1/v2 compatible subclip."""
        if _moviepy_v1:
            return clip.subclip(t_start, t_end) if t_end is not None else clip.subclip(t_start)
        return clip.subclipped(t_start, t_end) if t_end is not None else clip.subclipped(t_start)

    def _volume(clip, factor: float):
        """moviepy v1/v2 compatible volume adjustment."""
        if _moviepy_v1:
            return clip.fx(afx.volumex, factor)
        return clip.multiply_volume(factor)

    def _loop_to(clip, duration: float):
        """moviepy v1/v2 compatible audio looping to target duration."""
        if _moviepy_v1:
            return clip.fx(afx.audio_loop, duration=duration)
        return clip.loop(duration=duration)

    def _set_audio(video_clip, audio_clip):
        """moviepy v1/v2 compatible set_audio."""
        if _moviepy_v1:
            return video_clip.set_audio(audio_clip)
        return video_clip.with_audio(audio_clip)

    decision_by_shot = {(d["scene_id"], d["shot_id"]): d for d in edl["decisions"]}
    scene_order = [s["id"] for s in script["scenes"]]
    scene_by_id = {s["id"]: s for s in script["scenes"]}

    width, height = _resolution_dims()

    scene_clips = []
    for scene_id in scene_order:
        scene = scene_by_id[scene_id]
        shots = storyboard.get(scene_id, [])

        # Walk shots in storyboard order, gathering (decision, take_path)
        # pairs. We need decisions in storyboard order so transitions
        # line up with the right pair of clips.
        ordered_decisions: list[dict] = []
        ordered_paths: list[Path] = []
        for shot in shots:
            decision = decision_by_shot.get((scene_id, shot["shot_id"]))
            if not decision:
                continue
            takes = clips_manifest.get(scene_id, {}).get(shot["shot_id"], [])
            take_idx = max(0, decision["chosen_take"] - 1)
            if take_idx >= len(takes):
                take_idx = 0
            if not takes:
                continue
            cp = Path(takes[take_idx])
            if not cp.exists():
                continue
            ordered_decisions.append(decision)
            ordered_paths.append(cp)

        if not ordered_paths:
            continue

        # Two assembly paths:
        #
        # 1. All hard cuts → use moviepy's existing concatenate (fast,
        #    no re-encode of clips that don't need trimming).
        # 2. At least one non-cut transition → pre-trim each clip to a
        #    temp file, then render the scene through ffmpeg xfade.
        scene_transitions = transitions_for_scene(ordered_decisions)
        scene_video = None  # moviepy clip, set by one of the branches below

        if any_non_cut(scene_transitions):
            print(f"  {scene_id}: assembling with transitions "
                  f"{[t['type'] for t in scene_transitions]}")
            trimmed_paths: list[Path] = []
            for d, src in zip(ordered_decisions, ordered_paths):
                in_s = d.get("in_seconds") or 0.0
                out_s = d.get("out_seconds")
                trimmed_dst = exp.path(
                    f"_trimmed/{scene_id}/{d['shot_id']}_take{d['chosen_take']}.mp4"
                )
                trimmed_paths.append(_trim_clip(src, trimmed_dst, in_s, out_s))
            assembled = exp.path(f"_assembled/{scene_id}.mp4")
            render_clips_with_transitions(
                trimmed_paths, scene_transitions, assembled,
                fps=24, width=width, height=height,
            )
            scene_video = VideoFileClip(str(assembled))
        else:
            shot_clips = []
            for d, src in zip(ordered_decisions, ordered_paths):
                c = VideoFileClip(str(src))
                in_s = d.get("in_seconds") or 0.0
                out_s = d.get("out_seconds")
                if out_s and out_s > in_s:
                    c = _subclip(c, in_s, min(out_s, c.duration))
                elif in_s > 0:
                    c = _subclip(c, in_s)
                shot_clips.append(c)
            if not shot_clips:
                continue
            scene_video = concatenate_videoclips(shot_clips, method="compose")

        # Ambient bed. Optional — controlled by AMBIENT_SFX_ENABLED. When
        # off (the default), Veo's native audio + the music cue cover the
        # scene mix and we save ~$1/run. SFX now goes through Runway's
        # eleven_text_to_sound_v2 endpoint, billed in Runway credits.
        ambient_path = exp.path(f"sfx/{scene_id}/ambient.wav")
        if os.getenv("AMBIENT_SFX_ENABLED", "0").lower() in ("1", "true", "yes", "on"):
            sfx_prompt = (
                f"Continuous ambient sound, no music, no dialogue. "
                f"{scene['location']} at {scene.get('time_of_day', 'day')}. "
                f"Mood: {scene.get('mood', '')}."
            )
            if not ambient_path.exists():
                try:
                    amb = elevenlabs_sfx(sfx_prompt, duration_seconds=int(scene_video.duration))
                    ambient_path.write_bytes(amb)
                except Exception as e:  # noqa: BLE001
                    print(f"  Ambient {scene_id} failed: {e}")
            exp.log_prompt(
                target=f"sfx/{scene_id}/ambient.wav",
                model="eleven_text_to_sound_v2",
                prompt=sfx_prompt,
                stage="ambient_sfx",
                scene=scene_id,
                duration_seconds=int(scene_video.duration),
            )

        # Mix: Veo native (0 dB) + ambient (-16 dB) + music (-14 dB).
        layers = []
        if scene_video.audio is not None:
            layers.append(scene_video.audio)
        if ambient_path.exists():
            a = _volume(AudioFileClip(str(ambient_path)), 0.16)
            a = (_loop_to(a, scene_video.duration)
                 if a.duration < scene_video.duration
                 else _subclip(a, 0, scene_video.duration))
            layers.append(a)
        music_path = exp.path(f"music/{scene_id}.wav")
        if music_path.exists():
            m = _volume(AudioFileClip(str(music_path)), 0.20)
            if m.duration < scene_video.duration:
                # Music is shorter than the scene — loop it to fill.
                m = _loop_to(m, scene_video.duration)
            # If music is longer than the scene (because of the cross-shot
            # buffer added during generation), don't hard-cut it — let it
            # fade out over 2 seconds at the scene boundary so it bleeds
            # naturally across the cut into the next scene.
            elif m.duration > scene_video.duration:
                fade_end = min(m.duration, scene_video.duration + 2.0)
                m = _subclip(m, 0, fade_end)
            layers.append(m)
        if layers:
            scene_video = _set_audio(scene_video, CompositeAudioClip(layers))

        scene_clips.append(scene_video.crossfadein(0.5) if _moviepy_v1
                           else scene_video.with_effects([]))

    if not scene_clips:
        raise RuntimeError("No scene clips assembled.")

    final = concatenate_videoclips(scene_clips, method="compose")
    pre = exp.path("final_pregrade.mp4")
    final.write_videofile(
        str(pre),
        codec="libx264", audio_codec="aac", fps=24,
        bitrate="8000k", threads=4, preset="medium",
    )

    # Apply ffmpeg color grade in one pass.
    out = exp.path("final.mp4")
    grade = lookbook.get("ffmpeg_grade") or LOOKBOOK_GRADE
    if grade:
        try:
            ffmpeg([
                "-i", str(pre), "-vf", grade,
                "-c:v", "libx264", "-preset", "medium", "-b:v", "8000k",
                "-c:a", "copy",
                str(out),
            ])
            pre.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"  Color grade failed: {e}")
            pre.rename(out)
    else:
        pre.rename(out)

    return out


# ============================================================================
# Top-level pipeline runner
# ============================================================================

def run(exp: Experiment) -> Path:
    """Execute the full pipeline. Returns path to final.mp4."""
    print(f"[{exp.exp_id}] Stage 1: parse script")
    script = parse_script(exp)
    print(f"  → {len(script['scenes'])} scenes, {len(script['characters'])} characters")

    print(f"[{exp.exp_id}] Stage 1.5: format screenplay")
    script = format_screenplay(exp, script)
    n_elements = sum(len(s.get("elements") or []) for s in script["scenes"])
    print(f"  → {n_elements} screenplay elements across {len(script['scenes'])} scenes")

    print(f"[{exp.exp_id}] Stage 2: cast + locations")
    cast, locations = cast_and_locations(exp, script)
    print(f"  → {len(cast)} cast rows, {len(locations)} locations")

    print(f"[{exp.exp_id}] Stage 3: lookbook")
    lookbook = build_lookbook(exp, script, locations)
    print(f"  → grade: {lookbook.get('grade_description', '')[:80]}")

    print(f"[{exp.exp_id}] Stage 4: reference images")
    build_references(exp, script, cast, locations, lookbook)

    print(f"[{exp.exp_id}] Stage 5: shot list")
    storyboard = build_storyboard(exp, script)
    n_shots = sum(len(v) for v in storyboard.values())
    print(f"  → {n_shots} shots")

    print(f"[{exp.exp_id}] Stage 6: music")
    build_music(exp, script, storyboard)

    print(f"[{exp.exp_id}] Stage 7: first frames")
    build_first_frames(exp, script, cast, locations, lookbook, storyboard)

    print(f"[{exp.exp_id}] Stage 8: video ({TAKES_PER_SHOT} take/shot)")
    clips_manifest = build_video(exp, script, cast, lookbook, storyboard)

    print(f"[{exp.exp_id}] Stage 9: edit decisions")
    edl = build_edl(exp, storyboard, clips_manifest)

    print(f"[{exp.exp_id}] Stage 10: compile + grade + mix")
    final = compile_final(exp, script, storyboard, clips_manifest, edl, lookbook)
    print(f"  → {final}")
    return final


if __name__ == "__main__":
    exp = Experiment.new_or_resume()

    already_has = [
        art for art in ("script.json", "cast.json", "locations.json",
                        "lookbook.json", "storyboard.json", "final.mp4")
        if exp.has(art)
    ]
    if already_has:
        print(f"=== Resuming {exp.book_slug}/{exp.exp_id} "
              f"(seed={exp.seed}) ===")
        print(f"  Already complete: {', '.join(already_has)}")
        print(f"  Missing artifacts will be generated now.")
        print(f"  (Set FORCE_NEW=1 to start a fresh experiment instead.)")
    else:
        print(f"=== New experiment: {exp.book_slug}/{exp.exp_id} "
              f"(seed={exp.seed}) ===")

    final = run(exp)
    print(f"\nFinal film: {final}")

    # Auto-generate the production bible. Includes whatever has been
    # produced so far — if metric.json doesn't exist yet, the bible just
    # omits the critique section. After running evaluate.py, run
    # `python bible.py <exp_id>` to refresh the bible with critique data.
    try:
        from bible import build_bible
        bible_path = build_bible(exp)
        size_mb = bible_path.stat().st_size / 1_048_576
        print(f"Production bible: {bible_path}  ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"Bible generation failed (non-fatal): {e}")

    print(f"Next: `python evaluate.py {exp.exp_id}` to score it, "
          f"then `python bible.py {exp.exp_id}` to refresh the bible.")
