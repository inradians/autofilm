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
    EXPERIMENTS_DIR,
    BOOK_PDF_PATH,
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
    MAX_SHOTS_PER_SCENE,
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
    extract_last_video_frame,
    encode_image_for_claude,
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
    runway_tts,
    seedance,
    stable_audio,
    stable_image,
    veo,
    veo_final_model,
    DEFAULT_NARRATION_VOICE,
    RUNWAY_VOICE_IDS,
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
#
# Runway's default tier splits its concurrency budget by media category:
#   - video: 1 concurrent task per account
#   - image: 2 concurrent tasks per account
#   - audio: 1 concurrent task per account (sound_effect, text_to_speech)
#
# So we keep one knob per category. ``MAX_WORKERS`` is the overall cap
# and per-category fallback. Override individually for higher Runway
# tiers or when routing specific stages to non-Runway backends (LTX,
# BFL, Reve, Stability, Google all have their own limits).
MAX_WORKERS:       int = int(os.getenv("MAX_WORKERS",       "1"))
MAX_WORKERS_IMAGE: int = int(os.getenv("MAX_WORKERS_IMAGE", "2"))
MAX_WORKERS_VIDEO: int = int(os.getenv("MAX_WORKERS_VIDEO", "1"))
MAX_WORKERS_AUDIO: int = int(os.getenv("MAX_WORKERS_AUDIO", "1"))

# Map _parallel_run labels → category. Adding a new pipeline stage
# means adding it here so the runner picks the right concurrency cap.
_LABEL_CATEGORY: dict[str, str] = {
    "moodboard":   "image",
    "reference":   "image",
    "first_frame": "image",
    "style_frame": "image",
    "music":       "audio",
    "narration":   "audio",
    "video_take":  "video",
    "clip":        "video",
}


def _workers_for(label: str) -> int:
    """Pick concurrency cap for a stage label.

    Falls back to the overall MAX_WORKERS when the label has no entry
    in _LABEL_CATEGORY (so ad-hoc parallel stages still work) — that
    way an unmapped label is conservative, not over-eager.
    """
    cat = _LABEL_CATEGORY.get(label)
    if cat == "image": return MAX_WORKERS_IMAGE
    if cat == "video": return MAX_WORKERS_VIDEO
    if cat == "audio": return MAX_WORKERS_AUDIO
    return MAX_WORKERS

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
                    Determines the concurrency cap via _LABEL_CATEGORY
                    (image=2, video=1, audio=1 by default).
        work_items: One argument per worker_fn call.
        worker_fn:  Thread-safe callable; receives one item, returns a result.
        workers:    Pool size override. Defaults to the per-label cap.

    Returns:
        List of successful return values (order matches completion order,
        not input order). Failed items are logged and omitted.
    """
    n = workers if workers is not None else _workers_for(label)

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

PARSE_SYSTEM = """You are a script supervisor extracting structure from a
novel for screen adaptation.

For each chunk of pages, return:
  - book_title:  the actual title of the book (only required from the
                 chunk that contains the title page; otherwise leave
                 empty so later chunks don't overwrite it).
  - book_author: the author's name (same caveat — only the title-page
                 chunk needs to fill it in).
  - characters:  every character in this chunk with id, name, description.
  - scenes:      every scene with id, location, time_of_day, page_start,
                 page_end, characters present, summary, mood, and
                 (optionally) up to 3 short dialogue_excerpts — each at
                 most 12 words, capturing the dramatic shape of a key
                 line, NOT a verbatim transcription.

Return only via the tool."""

PARSE_TOOL_SCHEMA = {
    "description": "Submit characters and scene index for this chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "book_title": {
                "type": "string",
                "description": "Actual book title from the title page. "
                               "Empty string if this chunk doesn't contain it.",
            },
            "book_author": {
                "type": "string",
                "description": "Author's name. Empty string if not visible "
                               "in this chunk.",
            },
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
        cached = exp.read_json("script.json")
        # Stale-title repair: older versions of produce.py hardcoded
        # the title (e.g. "Jurassic Park") regardless of which book was
        # actually parsed. If we detect that the cached title doesn't
        # match the current BOOK_PDF_PATH the user is running against,
        # repair the title and source IN PLACE so downstream stages
        # (and reviewers) see the right metadata. We don't regenerate
        # the entire script — characters, scenes, and dialogue are
        # still valid — just the title/source fields.
        derived_from_filename = _humanize_filename_to_title(BOOK_PDF_PATH)
        cached_title = (cached.get("title") or "").strip()
        rewrite = False
        if (
            derived_from_filename
            and derived_from_filename != "Untitled"
            and cached_title
            and _title_seems_stale(cached_title, derived_from_filename)
        ):
            print(f"  ⚠ script.json title '{cached_title}' doesn't match "
                  f"current book file (expected ~ '{derived_from_filename}'); "
                  f"repairing in place.")
            cached["title"] = derived_from_filename
            # Preserve author if Claude extracted one — but rebuild the
            # combined source string from the corrected title.
            author = (cached.get("author") or "").strip()
            cached["source"] = (
                f"{derived_from_filename} by {author}" if author
                else derived_from_filename
            )
            rewrite = True
        # Stale-cast repair: older runs accumulated every character
        # mentioned anywhere in the book, even those not appearing in
        # the kept (MAX_SCENES-truncated) scenes. Sending all of them
        # to the casting tool blew past max_tokens and the response
        # arrived without a 'casting' key. Prune now so resumes
        # automatically heal.
        before = len(cached.get("characters", []))
        cached["characters"] = _prune_characters_to_scenes(
            cached.get("characters", []),
            cached.get("scenes", []),
        )
        if len(cached["characters"]) != before:
            print(f"  ⚠ script.json had {before} characters; pruned to "
                  f"{len(cached['characters'])} appearing in kept scenes.")
            rewrite = True
        if rewrite:
            exp.write_json("script.json", cached)
        return cached

    chunks = book_chunks(pages_per_chunk=25)
    characters: list[dict] = []
    scenes: list[dict] = []
    book_title  = ""
    book_author = ""

    for s, e, text in chunks:
        prior = json.dumps([{"id": c["id"], "name": c["name"]} for c in characters])
        result = claude_tool(
            system=PARSE_SYSTEM,
            user_content=f"BOOK pp.{s}-{e}:\n\n{text}\n\nPrior characters (reuse IDs):\n{prior}",
            tool_name="submit_extraction",
            tool_schema=PARSE_TOOL_SCHEMA,
            max_tokens=8000,
        )
        # Title/author are populated only by the chunk that contains the
        # title page — keep the first non-empty value we see.
        if not book_title and result.get("book_title"):
            book_title = result["book_title"].strip()
        if not book_author and result.get("book_author"):
            book_author = result["book_author"].strip()
        # Merge characters by id.
        by_id = {c["id"]: c for c in characters}
        for c in result.get("characters", []):
            if c["id"] not in by_id:
                by_id[c["id"]] = c
        characters = list(by_id.values())
        scenes.extend(result.get("scenes", []))

    if MAX_SCENES:
        scenes = scenes[:MAX_SCENES]

    # Prune characters to only those appearing in the kept scenes.
    # Without this, casting receives every character mentioned anywhere
    # in the book — for a long novel that can be hundreds of names that
    # never appear on screen because we only render the first N scenes.
    # Sending all of them to the casting tool blows past max_tokens and
    # the response is truncated, missing the required 'casting' key.
    before = len(characters)
    characters = _prune_characters_to_scenes(characters, scenes)
    if before != len(characters):
        print(f"  → pruned cast: {before} → {len(characters)} "
              f"(only characters appearing in kept scenes)")

    # Final fallback: if Claude couldn't extract a title from any chunk
    # (happens for PDFs with no title page, or scanned books with poor
    # OCR on the cover), humanize the filename so we never end up with
    # a stale hardcoded title from another book.
    if not book_title:
        book_title = _humanize_filename_to_title(BOOK_PDF_PATH)
    source = (
        f"{book_title} by {book_author}" if book_author else book_title
    )

    script = {
        "title":   book_title,
        "source":  source,
        "characters": characters,
        "scenes":  scenes,
    }
    exp.write_json("script.json", script)
    return script


def _prune_characters_to_scenes(characters: list[dict], scenes: list[dict]) -> list[dict]:
    """Trim a character list to only those appearing in the given scenes.

    Used in two places:
      1. After MAX_SCENES truncation in fresh parse_script runs, so the
         casting tool only sees relevant characters.
      2. On cached script.json load, to heal old runs that wrote a full
         all-chunks character list before this pruning existed.

    Returns the original list unchanged if no scene specifies its
    characters list (i.e. nothing to prune against — better to send
    everything than accidentally empty the cast).
    """
    kept_ids: set[str] = set()
    for sc in scenes:
        for cid in sc.get("characters", []):
            kept_ids.add(cid)
    if not kept_ids:
        return characters
    return [c for c in characters if c.get("id") in kept_ids]


def _humanize_filename_to_title(path: Path | str) -> str:
    """Best-effort title from a filename, used only when Claude can't
    extract a title from the book content.

    Examples:
      'JurassicPark-MichaelCrichton.pdf' -> 'Jurassic Park'
      'last_exit_to_brooklyn.pdf'        -> 'Last Exit To Brooklyn'
      'the_steel_drivin_man.pdf'         -> 'The Steel Drivin Man'
      'The-Steel-Drivin-Man.pdf'         -> 'The Steel Drivin Man'

    Same one-dash-only Title-Author heuristic as _book_slug — multi-
    dash filenames use the dash as a word separator and must not be
    truncated to a single leading word.
    """
    import re
    p = Path(path) if path else Path("")
    if not p.name:
        return "Untitled"
    stem = p.stem
    # Title-Author convention only when there's exactly one dash.
    if stem.count("-") == 1:
        stem = stem.split("-", 1)[0]
    # CamelCase / PascalCase → words.
    stem = re.sub(r"(?<!^)(?=[A-Z])", " ", stem)
    # Underscores / dashes / dots → spaces.
    stem = re.sub(r"[_.\-]+", " ", stem).strip()
    # Title-case each word.
    return " ".join(w.capitalize() for w in stem.split() if w) or "Untitled"


def _title_seems_stale(cached_title: str, expected_title: str) -> bool:
    """Decide whether a cached script.json title is a stale hardcode.

    'Stale' means the cached title contains no shared significant word
    with the title we'd derive from the current book filename. We can't
    require an exact match because Claude often produces a more polished
    version of the filename-derived title (e.g., 'The Steel-Driving
    Man' vs 'The Steel Drivin Man') — those are LEGITIMATELY different.

    Stale: cached='Jurassic Park', filename='The Steel Drivin Man'
           → no shared word → repair.
    Fine:  cached='The Steel-Driving Man', filename='The Steel Drivin Man'
           → 'steel'/'man' overlap → keep cached (Claude's version is
           better than the mechanical filename humanization).
    """
    import re

    def _significant_words(s: str) -> set[str]:
        # Strip stopwords and short tokens. 'the', 'a', 'of', 'by' etc.
        # carry no signal for matching.
        stop = {"the", "a", "an", "of", "and", "or", "by", "to", "in",
                "on", "at", "for", "with", "from"}
        words = re.findall(r"[a-z0-9]+", s.lower())
        return {w for w in words if len(w) >= 3 and w not in stop}

    cached_words   = _significant_words(cached_title)
    expected_words = _significant_words(expected_title)
    if not expected_words:
        return False  # we have nothing to compare against; trust cached
    # If the two share NO significant word, the cached title is from a
    # different book entirely.
    return not (cached_words & expected_words)


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

  - "narration" elements: voice-over narration (V.O.) that rides over
    the visual. Use sparingly — only when the source uses interior
    monologue, expository framing, or first-person reflection that
    cannot be shown visually. Each narration element MUST be at most
    35 words. The character field, if set, identifies whose V.O. it is
    (a character_id or "narrator" for an omniscient narrator). Narration
    is rendered as ElevenLabs TTS and layered over the scene's clips.

  - "transition" elements: rare; only where the scene needs a hard
    transition like CUT TO BLACK or SMASH CUT.

Aim for 4-10 elements per scene. Mix action and dialogue. Open with an
action line establishing the visual; close with the strongest beat of
the summary. Add narration only where it earns its place. Return only
via the tool."""

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
                            "enum": ["action", "dialogue", "narration", "transition"],
                        },
                        "text": {
                            "type": "string",
                            "description": "Action ≤25 words, dialogue ≤15 words, narration ≤35 words.",
                        },
                        "character": {
                            "type": "string",
                            "description": "character_id; required for dialogue. For narration: character_id or 'narrator'.",
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
#
# Each cascade publishes the LABEL of the model that ultimately succeeded
# via this thread-local. Callers read it AFTER a successful return to
# attribute the artifact for log_prompt() / cost tracking / UI badges. We
# use a thread-local rather than a tuple return value to avoid touching
# every caller across the codebase.

_last_chain_model = threading.local()

def _publish_chain_model(label: str) -> None:
    """Record which model just succeeded for the current thread.
    Callers read it via last_chain_model() to log_prompt with accurate
    model attribution."""
    _last_chain_model.value = label

def last_chain_model() -> str:
    """Return the model label most recently published by a successful
    cascade call ON THIS THREAD. Empty string if nothing yet."""
    return getattr(_last_chain_model, "value", "")

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
            _publish_chain_model(label)
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                # Mark Runway as unavailable and skip remaining Runway attempts
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if l not in ("seedance", "veo")]
            # Veo's safety filter sometimes returns the operation with a
            # null response — surfaced as "no videos returned" or
            # "'NoneType' object has no attribute ...". Squash those down
            # to a one-liner; the cascade still handles the recovery.
            msg = str(e)
            if ("no videos returned" in msg
                or "'NoneType' object has no attribute" in msg
                or "safety-filter block" in msg):
                msg = "veo safety filter or null response; cascading"
            _tprint(f"    ✗ [{label}] {context}: {msg}")
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

    # Stability AI — absolute last resort. Uses STABILITY_API_KEY (already
    # required for the music stage). Stable Image Core is the cheapest tier.
    if os.environ.get("STABILITY_API_KEY"):
        attempts.append((
            "stable_image",
            lambda: stable_image(prompt, aspect_ratio="16:9", tier="core"),
        ))

    last_exc: Exception | None = None
    for label, fn in attempts:
        try:
            _tprint(f"    [{label}] t2i {context}")
            result = fn()
            _tprint(f"    ✓ [{label}] {context} ({len(result)//1024}kB)")
            _publish_chain_model(label)
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if not l.startswith(("gpt_image", "nano_banana"))]
            # Trim noisy moderation-block tracebacks down to a one-liner;
            # cascade still proceeds normally.
            msg = str(e)
            if "moderation_blocked" in msg:
                msg = "moderation_blocked (provider safety filter); cascading"
            _tprint(f"    ✗ [{label}] {context}: {msg}")
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
            _publish_chain_model(label)
            return result
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, context)
                runway_ok = False
                attempts = [(l, f) for l, f in attempts
                            if not l.startswith(("gen4_image", "nano_banana"))]
            msg = str(e)
            if "moderation_blocked" in msg:
                msg = "moderation_blocked (provider safety filter); cascading"
            _tprint(f"    ✗ [{label}] {context}: {msg}")
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


CAST_SYSTEM = """You are a casting director for a virtual film adaptation
where every performer is a fully-fictional digital actor — no real,
named human actors are used.

For each character, write a virtual-actor card describing the kind of
human you'd cast IF you had unlimited fabrication. Use ARCHETYPE
language: physical type, age range, ethnicity (only if the character
is canonically of a specific background), build, signature features,
and an acting register (e.g. "subdued and watchful", "loose and
playful").

Do NOT name any real actor, celebrity, model, or public figure — living
or deceased. The performer field is the *archetype*, not a person. Pick
short evocative archetype labels like "weather-worn academic in his
fifties" or "wiry teen with a quick suspicious gaze". Provide a short
rationale tying the archetype to the character. No reuse of identical
archetypes across roles. Return only via the tool."""

CAST_TOOL_SCHEMA = {
    "description": "Submit virtual casting choices (no real actors).",
    "input_schema": {
        "type": "object",
        "properties": {
            "casting": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "character_id": {"type": "string"},
                        "actor": {
                            "type": "string",
                            "description": "Virtual archetype label (≤12 words). NEVER a real person's name.",
                        },
                        "rationale": {"type": "string"},
                        "alternative": {
                            "type": "string",
                            "description": "Backup virtual archetype.",
                        },
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
        max_tokens=8000,
    )
    cast = cast_result.get("casting")
    if not cast:
        # Common cause: too many characters in script['characters'] →
        # response truncated by max_tokens → tool call returns no
        # 'casting' key. Prevention is in parse_script (we now prune
        # characters to those appearing in MAX_SCENES kept scenes),
        # but if it still happens, surface a clear error rather than
        # KeyError'ing on the next line.
        raise RuntimeError(
            f"casting tool returned no 'casting' key (got: "
            f"{list(cast_result.keys())}). Most likely the response "
            f"was truncated — script has {len(script.get('characters', []))} "
            f"characters; consider lowering MAX_SCENES or pruning the cast."
        )
    exp.write_json("cast.json", cast)

    # Locations.
    loc_result = claude_tool(
        system=LOCATIONS_SYSTEM,
        user_content=f"Scenes:\n{json.dumps(script['scenes'], indent=2)}\n\nDescribe unique locations.",
        tool_name="submit_locations",
        tool_schema=LOCATIONS_TOOL_SCHEMA,
        max_tokens=6000,
    )
    locations = loc_result.get("locations")
    if not locations:
        raise RuntimeError(
            f"locations tool returned no 'locations' key (got: "
            f"{list(loc_result.keys())})."
        )

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

    _n_mb = _workers_for('moodboard')
    _tprint(f"  Generating {len(locations)} moodboard(s) "
            f"({'parallel' if _n_mb > 1 else 'serial'}, "
            f"workers={_n_mb})...")
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


def _load_user_moodboards(exp: Experiment) -> list[bytes]:
    """Read user-uploaded moodboard images for this experiment's book.

    These are the **general creative aesthetic / "vibe" references** for
    the whole movie — not per-scene or per-character. The lookbook stage
    uses them as style refs when generating the master style frame, so
    the entire film inherits the user's intended look.

    Saved by the UI server to experiments/<book_slug>/user_moodboards/.
    Returns image bytes in lexicographic filename order, or [] if the
    directory doesn't exist or is empty.

    Located per-book (not per-experiment) so refs survive across all
    iterations of the autoresearch loop without re-uploading.
    """
    mb_dir = EXPERIMENTS_DIR / exp.book_slug / "user_moodboards"
    if not mb_dir.is_dir():
        return []
    refs: list[bytes] = []
    for p in sorted(mb_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            refs.append(p.read_bytes())
        except Exception:
            continue
    return refs


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
    # If the user uploaded moodboard examples via the UI, they're saved
    # under experiments/<book_slug>/user_moodboards/ — use them as refs
    # so the style frame inherits the user's intended aesthetic.
    user_mb_refs: list[bytes] = _load_user_moodboards(exp)

    try:
        if user_mb_refs:
            # Use the FIRST user moodboard as the primary style ref.
            sf = _generate_image_with_refs(
                prompt=lookbook["style_frame_prompt"],
                refs=user_mb_refs[:3],   # up to 3 refs for the style frame
                size="1792x1024",
                quality="high",
                context="lookbook/style_frame",
            )
            print(f"  using {len(user_mb_refs[:3])} user moodboard(s) as style refs")
        else:
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

    _n_ref = _workers_for('reference')
    _tprint(f"  Generating {len(work_items)} reference image(s) "
            f"({'parallel' if _n_ref > 1 else 'serial'}, "
            f"workers={_n_ref})...")
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

DIALOGUE / NARRATION — MANDATORY when dialogue_excerpt is non-empty:
  - speaker_id: the character_id of whoever speaks the line. Pull it
    from the scene's `characters` list (or "narrator" for narration).
    Required. Without this, the video model picks the wrong on-screen
    character to lip-sync — visible mouth on the wrong face.
  - is_narration: true ONLY when the line is voice-over by an unseen
    narrator (NOT spoken by a visible character). Default false.
    When true, NO character should lip-sync this line on camera —
    the narrator is offscreen, riding over the visuals. Choose
    dialogue_excerpts of in-scene type whenever possible; reserve
    narration excerpts for shots where the V.O. anchors the visual.

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
                        "speaker_id": {
                            "type": "string",
                            "description": (
                                "character_id of whoever speaks "
                                "dialogue_excerpt. Use 'narrator' "
                                "for V.O. narration. Required when "
                                "dialogue_excerpt is non-empty."
                            ),
                        },
                        "is_narration": {
                            "type": "boolean",
                            "description": (
                                "True iff dialogue_excerpt is "
                                "voice-over narration by an unseen "
                                "narrator. When true, no on-screen "
                                "character should lip-sync the line."
                            ),
                        },
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
    # When MAX_SHOTS_PER_SCENE is set (smoke-test mode), nudge the
    # model toward exactly that many shots and truncate as a safety
    # net. Saves Claude tokens and ensures the cap holds even if the
    # model overshoots the hint.
    extra_hint = ""
    if MAX_SHOTS_PER_SCENE > 0:
        extra_hint = (
            f"\n\nIMPORTANT: produce EXACTLY {MAX_SHOTS_PER_SCENE} "
            f"shot(s) for this scene. Cover the most essential beat."
        )
    shots = claude_tool(
        system=SHOTLIST_SYSTEM,
        user_content=(
            f"Scene:\n{json.dumps(scene, indent=2)}"
            f"\n\nBreak it into shots.{extra_hint}"
        ),
        tool_name="submit_shotlist",
        tool_schema=SHOTLIST_TOOL_SCHEMA,
    )["shots"]
    if MAX_SHOTS_PER_SCENE > 0:
        shots = shots[:MAX_SHOTS_PER_SCENE]
    return shots


def build_storyboard(exp: Experiment, script: dict) -> dict:
    if exp.has("storyboard.json"):
        cached = exp.read_json("storyboard.json")
        # Auto-repair for storyboards from before speaker_id /
        # is_narration were added. Without those, veo_prompt picks
        # the wrong character to lip-sync (it defaults to the first
        # character listed in the scene). Try a best-effort heuristic:
        #   - 0 characters in the scene → it's V.O., set is_narration
        #   - 1 character             → that's the speaker, fill it in
        #   - 2+ characters           → ambiguous, leave blank + warn
        scene_chars = {sc["id"]: sc.get("characters", []) for sc in script["scenes"]}
        repaired = 0
        ambiguous = 0
        for sid, shots in cached.items():
            chars = scene_chars.get(sid, [])
            for shot in shots:
                if not (shot.get("dialogue_excerpt") or "").strip():
                    continue
                if shot.get("speaker_id") or "is_narration" in shot:
                    continue   # already populated by the new schema
                if not chars:
                    shot["speaker_id"]   = "narrator"
                    shot["is_narration"] = True
                    repaired += 1
                elif len(chars) == 1:
                    shot["speaker_id"]   = chars[0]
                    shot["is_narration"] = False
                    repaired += 1
                else:
                    # Multi-character scene with no speaker hint — this
                    # one needs Claude to disambiguate. Mark and warn.
                    ambiguous += 1
        if repaired:
            print(f"  ⚠ storyboard.json: auto-filled speaker_id for "
                  f"{repaired} shot(s) (1-character scenes / V.O.)")
            exp.write_json("storyboard.json", cached)
        if ambiguous:
            print(f"  ⚠ storyboard.json: {ambiguous} dialogue shot(s) in "
                  f"multi-character scenes have no speaker_id — Veo will "
                  f"fall back to the first listed character. Delete "
                  f"storyboard.json to regenerate with explicit speakers.")
        return cached
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

    _n_mu = _workers_for('music')
    _tprint(f"  Generating {len(script['scenes'])} music cue(s) "
            f"({'parallel' if _n_mu > 1 else 'serial'}, workers={_n_mu})...")
    _parallel_run("music", script["scenes"], _do_music)


# ============================================================================
# STAGE 6.5 — Narration / voice-over (ElevenLabs TTS via Runway)
# ============================================================================

def _scene_narration_text(scene: dict) -> tuple[str, str | None]:
    """Concatenate all `narration` elements in a scene into a single block.

    Returns (text, voice_id_or_None). Voice is taken from the first
    narration element's `character` field if it maps to a known preset;
    otherwise None (caller picks the default voice).
    """
    parts: list[str] = []
    voice_hint: str | None = None
    for el in scene.get("elements", []) or []:
        if el.get("type") != "narration":
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        # Prepend parenthetical tone hint as ElevenLabs audio tag if present
        # (e.g. "(quietly)" → "[whispers]"). Only the most common ones map.
        paren = (el.get("parenthetical") or "").strip().lower()
        tag_map = {
            "whisper":   "[whispers]",
            "quiet":     "[whispers]",
            "quietly":   "[whispers]",
            "warmly":    "[soft]",
            "amused":    "[chuckles]",
            "warning":   "[serious]",
            "urgent":    "[urgent]",
        }
        tag = tag_map.get(paren, "")
        parts.append(f"{tag} {text}".strip())
        if voice_hint is None and el.get("character"):
            voice_hint = el["character"]
    return " ".join(parts), voice_hint


def build_narration(exp: Experiment, script: dict, cast: list[dict]) -> None:
    """Generate per-scene narration audio (V.O.) via ElevenLabs TTS.

    Walks each scene's screenplay elements, collects all `narration`
    elements, and synthesizes a single MP3 per scene at
    `narration/{scene_id}.mp3`. Scenes without narration elements are
    skipped (no file written).

    The audio mix step (compile_final) layers this MP3 over the scene's
    clips when present.
    """
    # Map character_id → preferred voice slot (first character gets
    # rachel, second gets george, etc.). The "narrator" sentinel always
    # uses the default rachel voice.
    # Cast rows use 'character_id' per CAST_TOOL_SCHEMA, not 'id'. Be
    # defensive and accept either so this works for stale fixtures too.
    voice_pool = list(RUNWAY_VOICE_IDS.values())
    char_voice: dict[str, str] = {}
    for i, c in enumerate(cast):
        cid = c.get("character_id") or c.get("id")
        if not cid:
            continue
        char_voice[cid] = voice_pool[(i + 1) % len(voice_pool)]

    def _do_narration(scene: dict) -> None:
        scene_id = scene["id"]
        out_path = exp.path(f"narration/{scene_id}.mp3")
        if out_path.exists():
            return

        text, voice_hint = _scene_narration_text(scene)
        if not text:
            return  # No narration in this scene

        # Pick voice: explicit character_id → that character's voice,
        # "narrator" → default, else default.
        voice_id = (
            char_voice.get(voice_hint)
            if voice_hint and voice_hint != "narrator"
            else RUNWAY_VOICE_IDS[DEFAULT_NARRATION_VOICE]
        )

        try:
            audio = runway_tts(text=text, voice_id=voice_id)
            out_path.write_bytes(audio)
            _tprint(f"  ✓ narration/{scene_id} ({len(text)} chars, "
                    f"{len(audio)//1024}kB)")
        except Exception as e:  # noqa: BLE001
            _tprint(f"  ⚠ Narration {scene_id} failed: {e}")

        exp.log_prompt(
            target=f"narration/{scene_id}.mp3",
            model="eleven_multilingual_v2",
            prompt=text[:200],
            stage="narration",
            scene=scene_id,
            voice_id=voice_id,
        )

    n_with_narration = sum(
        1 for s in script["scenes"]
        if any(e.get("type") == "narration" for e in (s.get("elements") or []))
    )
    if n_with_narration == 0:
        _tprint("  No narration in script — skipping TTS stage.")
        return
    _n_na = _workers_for('narration')
    _tprint(f"  Generating narration for {n_with_narration} scene(s) "
            f"({'parallel' if _n_na > 1 else 'serial'}, workers={_n_na})...")
    _parallel_run("narration", script["scenes"], _do_narration)


# ============================================================================
# STAGE 7 — First frames per shot
# ============================================================================

def first_frame_prompt(lookbook: dict, shot: dict, scene: dict,
                        chars_in_shot: list[dict], actors_in_shot: list[str],
                        location: dict | None) -> str:
    # Virtual characters: blend the canonical book description with the
    # archetype label. No real-actor attribution.
    char_lines = "\n".join(
        f"  - {c['name']} — {a}. {c.get('description', '')}"
        for c, a in zip(chars_in_shot, actors_in_shot)
    ) or "  (no on-screen characters)"
    loc_text = location["description"][:300] if location else scene["location"]

    # Speaker emphasis: if the shot has in-scene dialogue, surface the
    # speaker so the still composition leads with that character —
    # video models tend to lip-sync whoever's most visually prominent
    # in the starting frame, so getting composition right here pays off
    # downstream. For narration shots, suppress speaker emphasis: the
    # narrator is unseen, no character should be presented as 'about
    # to speak'.
    speaker_block = ""
    excerpt = (shot.get("dialogue_excerpt") or "").strip()
    if excerpt and not shot.get("is_narration"):
        speaker_id = shot.get("speaker_id") or ""
        speaker_name = None
        for c in chars_in_shot:
            if c.get("id") == speaker_id or c.get("character_id") == speaker_id:
                speaker_name = c["name"]
                break
        if speaker_name:
            speaker_block = (
                f"SPEAKER FOCUS: {speaker_name} is the one delivering the "
                f"line in this shot — frame and light them as the dialogue "
                f"subject; other characters react or listen.\n"
            )
    elif excerpt and shot.get("is_narration"):
        # Narration: visually treat as an action/establishing beat. The
        # voice-over rides over visuals; nobody on screen is "speaking".
        speaker_block = (
            f"VOICE-OVER SCENE: the line in dialogue_excerpt is "
            f"voice-over by an unseen narrator — compose this as a "
            f"clean action / establishing image; no character should "
            f"be framed as if mid-speech.\n"
        )

    return (
        style_preamble(lookbook)
        + f"PHOTOREALISTIC CINEMATIC FILM STILL.\n\n"
        f"SHOT: {shot['shot_size']} ({shot['angle']} angle), ~{shot.get('lens_mm', 35)}mm lens.\n"
        f"CAMERA: starts {shot.get('camera_move', 'static')}.\n\n"
        f"LOCATION: {scene['location']}. {loc_text}\n"
        f"TIME: {scene.get('time_of_day', 'day')}\n\n"
        f"CHARACTERS:\n{char_lines}\n\n"
        f"ACTION: {shot['action']}\n"
        f"{speaker_block}"
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
            composition_model = last_chain_model() or "auto"
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
                final_model = last_chain_model() or "unknown"
            except RuntimeError as e:
                _tprint(f"    ⚠ Lock failed for {scene_id}/{shot_id}: {e}  (using composition)")
                final = composition
                final_model = composition_model
        else:
            final = composition
            final_model = composition_model

        out_path.write_bytes(final)
        # Log the on-disk artifact with the model that actually produced
        # it. This is the key the UI uses for the colored model border;
        # without it, first-frame thumbs render unstyled because the
        # bible's frames.by_scene_shot[k] resolves to the .png path but
        # prompts.json only had an entry under .composition.png.
        exp.log_prompt(
            target=f"frames/{scene_id}/{shot_id}.png",
            model=final_model,
            prompt=nano_prompt if all_refs else ff_prompt,
            stage="first_frame",
            scene=scene_id, shot=shot_id,
        )
        return scene_id, shot_id, str(out_path)

    _n_ff = _workers_for('first_frame')
    _tprint(f"  Generating {len(work_items)} first frame(s) "
            f"({'parallel' if _n_ff > 1 else 'serial'}, "
            f"workers={_n_ff})...")
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
        # Virtual-archetype phrasing — never says "played by <real actor>".
        # The "actor" field is now an archetype label like "weather-worn
        # academic in his fifties", so we surface it as a casting note
        # rather than as a real-person attribution.
        char_lines = [f"{c['name']} ({a})"
                      for c, a in zip(chars_in_shot, actors_in_shot)]
        char_block = f"CHARACTERS: {', '.join(char_lines)}.\n"

    # Dialogue / narration block. Two distinct cases the video model
    # has to handle correctly:
    #
    #   1. In-scene dialogue: a visible character speaks. We need to
    #      tell the model EXACTLY which character lip-syncs (using
    #      shot.speaker_id, NOT chars_in_shot[0] — that picked the
    #      wrong character whenever the scene had multiple people).
    #
    #   2. Voice-over narration: the line is spoken by an unseen
    #      narrator riding OVER the visuals. No on-screen character
    #      should lip-sync. The narration audio is added separately
    #      in build_narration() — the video stage must keep mouths
    #      shut on this shot or the cut will look like everyone is
    #      mouthing the narrator's words.
    dialogue_block = ""
    excerpt = (shot.get("dialogue_excerpt") or "").strip()
    if excerpt:
        is_narration = bool(shot.get("is_narration"))
        speaker_id = shot.get("speaker_id") or ""

        if is_narration or speaker_id == "narrator":
            # Narrator is OFF-SCREEN. Tell Veo this is V.O. and
            # instruct it explicitly NOT to lip-sync the line on
            # any visible character. The narration audio comes from
            # ElevenLabs in build_narration() and is mixed at compile
            # time — the video here should be clean visuals only.
            dialogue_block = (
                f"VOICE-OVER (narrator unseen — DO NOT lip-sync "
                f"this line on any visible character; characters "
                f"on screen continue their action silently): "
                f'"{excerpt}"\n'
            )
        else:
            # Resolve speaker by id from the shot's character list.
            # Falls back to first character only if no speaker_id was
            # set (legacy storyboards) — and emits a fallback note so
            # downstream debugging can spot the schema gap.
            speaker = None
            for c in chars_in_shot:
                if c.get("id") == speaker_id or c.get("character_id") == speaker_id:
                    speaker = c["name"]
                    break
            if speaker is None:
                speaker = chars_in_shot[0]["name"] if chars_in_shot else "the character"
            # Voice direction comes from the character's archetype, not
            # from any real actor's voice. We omit the explicit "in X's
            # voice" because the model would otherwise try to mimic a
            # real person.
            dialogue_block = (
                f"DIALOGUE (synchronized native audio): "
                f"ONLY {speaker} speaks and lip-syncs this line; "
                f"all other characters on screen remain silent and "
                f"do not move their mouths. {speaker}: \"{excerpt}\"\n"
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

        for shot_idx, shot in enumerate(shots):
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

            # Continuity: tag each shot 2+ in a scene with the
            # previous shot's id. _do_take will look up the previous
            # take's tail frame and pass it as a style ref so cross-
            # model renders stay visually coherent within a scene.
            prev_shot_id = shots[shot_idx - 1]["shot_id"] if shot_idx > 0 else None

            for take_idx in range(TAKES_PER_SHOT):
                take_path = exp.path(f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4")
                if take_path.exists():
                    manifest[scene_id][shot_id].append(str(take_path))
                    continue
                work_items.append((scene_id, shot, take_idx, ff, ref_imgs, route,
                                   scene, chars, actors, prev_shot_id))

    def _do_take(item: tuple) -> tuple[str, str, str] | None:
        (scene_id, shot, take_idx, ff, ref_imgs, route,
         scene, chars, actors, prev_shot_id) = item
        shot_id   = shot["shot_id"]
        take_path = exp.path(f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4")
        vp = veo_prompt(lookbook, shot, scene, chars, actors, take_idx)

        # Continuity refinement — STYLE/MOOD context, not content.
        # When MAX_WORKERS_VIDEO=1 (the Runway-friendly default), work
        # items are processed in submission order, so by the time we
        # reach shot N+1 the previous shot's first take is already on
        # disk. We grab its tail frame and prepend it to ref_imgs so
        # whichever video model handles this shot has the prior shot's
        # color, lighting, and grading as a style anchor. This is the
        # crucial trick that keeps the look stable when shot N goes
        # through Veo and shot N+1 falls back to LTX (or vice versa).
        cont_status = ""
        if prev_shot_id:
            prev_video = exp.path(f"clips/{scene_id}/{prev_shot_id}/take_1.mp4")
            if prev_video.exists():
                try:
                    last_frame = extract_last_video_frame(prev_video)
                    ref_imgs = [last_frame] + ref_imgs
                    cont_status = " (cont. from prev)"
                except Exception:                                  # noqa: BLE001
                    pass   # continuity is best-effort; fall through

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
                    f"({len(video_bytes)//1024}kB){cont_status}")
            return scene_id, shot_id, str(take_path)
        except Exception as e:  # noqa: BLE001
            if _is_daily_limit(e):
                _record_daily_limit(e, f"video/{scene_id}/{shot_id}/take_{take_idx+1}")
                return None
            _tprint(f"  Take {scene_id}/{shot_id}/{take_idx + 1} failed: {e}")
            return None

    _n_vt = _workers_for('video_take')
    _tprint(f"  Generating {len(work_items)} take(s) "
            f"({'parallel' if _n_vt > 1 else 'serial'}, "
            f"workers={_n_vt})...")
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
    # Resume: if the existing EDL covers every storyboard shot, return
    # it. Otherwise rebuild — a partial EDL from a prior crashed run
    # would silently drop the missing shots from the final cut.
    if exp.has("edl.json"):
        cached = exp.read_json("edl.json")
        cached_keys = {(d["scene_id"], d["shot_id"]) for d in cached.get("decisions", [])}
        storyboard_keys = {
            (sid, shot["shot_id"])
            for sid, shots in storyboard.items()
            for shot in shots
        }
        missing = storyboard_keys - cached_keys
        if not missing:
            return cached
        print(f"  ⚠ Cached edl.json missing {len(missing)} shot(s) "
              f"({sorted(missing)[:3]}{'...' if len(missing) > 3 else ''}); "
              f"rebuilding.")

    # Single-take fast path. Iterate STORYBOARD order so every shot
    # gets a decision — even if its take is missing from the manifest
    # for some reason. compile_final's fallback then handles the
    # "decision exists but no take" case explicitly rather than
    # silently dropping the shot from the cut.
    max_takes = max((len(t) for s in clips_manifest.values() for t in s.values()),
                    default=0)
    if max_takes <= 1:
        decisions = [
            {"scene_id": scene_id, "shot_id": shot["shot_id"],
             "chosen_take": 1, "rationale": "Single take generated."}
            for scene_id, shots in storyboard.items()
            for shot in shots
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
                    media_type, data = encode_image_for_claude(frame)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
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
    # moviepy 1.x and 2.x have different APIs and can't be told apart by
    # import alone (some 2.x installs ship a backwards-compat
    # ``moviepy.editor`` shim that re-exports the v2 classes). We import
    # whichever works, then dispatch each operation by INSPECTING the
    # actual method names on the resulting class — that way we don't
    # care which major version is installed.
    try:
        from moviepy.editor import (
            AudioFileClip, CompositeAudioClip, VideoFileClip,
            concatenate_videoclips,
        )
    except ModuleNotFoundError:
        from moviepy import (  # type: ignore[no-redef]
            AudioFileClip, CompositeAudioClip, VideoFileClip,
            concatenate_videoclips,
        )

    try:
        import moviepy.audio.fx as afx          # noqa: F401  (used in v1 fx() path)
    except Exception:                           # noqa: BLE001
        afx = None  # v2 may not expose it the same way

    _mp_version = getattr(__import__("moviepy"), "__version__", "?")
    print(f"  moviepy {_mp_version}: dispatching by method introspection")

    def _subclip(clip, t_start, t_end=None):
        """v1: clip.subclip / v2: clip.subclipped."""
        fn = getattr(clip, "subclipped", None) or getattr(clip, "subclip", None)
        if fn is None:
            raise AttributeError("clip has neither subclipped nor subclip")
        return fn(t_start, t_end) if t_end is not None else fn(t_start)

    def _volume(clip, factor: float):
        """Volume scale by factor.

        v2.x:    clip.with_volume_scaled(factor)
        v2 mid:  clip.multiply_volume(factor)
        v1.x:    clip.fx(afx.volumex, factor)

        We pick the first one that exists. ``with_volume_scaled`` is the
        canonical v2 form per moviepy.org docs.
        """
        for name in ("with_volume_scaled", "multiply_volume"):
            fn = getattr(clip, name, None)
            if fn is not None:
                return fn(factor)
        # Fall back to the v1 effect path.
        if afx is not None and hasattr(afx, "volumex"):
            return clip.fx(afx.volumex, factor)
        raise AttributeError(
            f"no compatible volume method on {type(clip).__name__} "
            f"(tried with_volume_scaled, multiply_volume, fx(volumex))"
        )

    def _loop_to(clip, duration: float):
        """Loop a clip out to ``duration`` seconds.

        v2 high-level helper:    clip.with_loop(duration=...)   (some builds)
        v2 effect-based form:    clip.with_effects([afx.AudioLoop(duration=...)])
        v2 mid-build:            clip.audio_loop(duration=...)
        v1 fx-style:             clip.fx(afx.audio_loop, duration=...)
        """
        for name in ("with_loop", "audio_loop", "loop"):
            fn = getattr(clip, name, None)
            if fn is not None:
                try:
                    return fn(duration=duration)
                except TypeError:
                    return fn(duration)
        # Effect-based fallback (v2 canonical).
        try:
            from moviepy.audio.fx.AudioLoop import AudioLoop  # type: ignore
            return clip.with_effects([AudioLoop(duration=duration)])
        except Exception:                                         # noqa: BLE001
            pass
        if afx is not None and hasattr(afx, "audio_loop"):
            return clip.fx(afx.audio_loop, duration=duration)
        raise AttributeError(
            f"no compatible loop method on {type(clip).__name__}"
        )

    def _set_audio(video_clip, audio_clip):
        """v1: video.set_audio / v2: video.with_audio."""
        fn = getattr(video_clip, "with_audio", None) \
             or getattr(video_clip, "set_audio", None)
        if fn is None:
            raise AttributeError(
                "VideoFileClip has neither with_audio nor set_audio"
            )
        return fn(audio_clip)

    def _crossfadein(clip, duration: float):
        """Apply a crossfade-in transition.

        v1: clip.crossfadein(duration)
        v2: clip.with_effects([vfx.CrossFadeIn(duration=...)])
        If neither path works (some 2.x intermediate builds), return
        the clip unchanged — losing only the first-scene fade-in is
        much better than crashing the whole compile.
        """
        fn = getattr(clip, "crossfadein", None)
        if fn is not None:
            try:
                return fn(duration)
            except Exception:                                 # noqa: BLE001
                pass
        try:
            from moviepy.video.fx.CrossFadeIn import CrossFadeIn  # type: ignore
            return clip.with_effects([CrossFadeIn(duration=duration)])
        except Exception:                                     # noqa: BLE001
            return clip

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
        skipped: list[str] = []   # for the per-scene summary line
        for shot in shots:
            shot_id = shot["shot_id"]
            decision = decision_by_shot.get((scene_id, shot_id))
            if not decision:
                # Shot is in storyboard but EDL has no decision for it.
                # Most common cause: build_edl's vision-based picker
                # returned a partial response, or the manifest didn't
                # include this shot when the EDL was first written.
                # Fall back to take 1 if a take file actually exists,
                # so the shot isn't silently dropped from the cut.
                takes = clips_manifest.get(scene_id, {}).get(shot_id, [])
                if takes and Path(takes[0]).exists():
                    decision = {
                        "scene_id":     scene_id,
                        "shot_id":      shot_id,
                        "chosen_take":  1,
                        "rationale":    "Fallback (no EDL decision found at compile time)",
                    }
                    print(f"  ⚠ {scene_id}/{shot_id}: no EDL decision, "
                          f"falling back to take 1")
                else:
                    skipped.append(f"{shot_id} (no decision, no takes)")
                    continue
            takes = clips_manifest.get(scene_id, {}).get(shot_id, [])
            if not takes:
                skipped.append(f"{shot_id} (no takes in manifest)")
                continue
            take_idx = max(0, decision["chosen_take"] - 1)
            if take_idx >= len(takes):
                take_idx = 0
            cp = Path(takes[take_idx])
            if not cp.exists():
                skipped.append(f"{shot_id} (take file missing: {cp.name})")
                continue
            ordered_decisions.append(decision)
            ordered_paths.append(cp)

        # Per-scene summary: makes 'missing clips in final' diagnosable.
        included = len(ordered_paths)
        total    = len(shots)
        if skipped:
            print(f"  {scene_id}: {included}/{total} shots included; "
                  f"skipped — {', '.join(skipped)}")
        else:
            print(f"  {scene_id}: {included}/{total} shots included")

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
            # Guard against audio/video duration drift in the assembled
            # MP4. moviepy reads `duration` from container metadata which
            # reflects the LONGER of the two streams — when audio runs a
            # hair past video output, asking moviepy to read every frame
            # produces 'using last valid frame' warnings (a 1-3s frozen
            # tail). Trim moviepy's view to the actual VIDEO STREAM
            # duration. ONLY trim if the gap is small (<=5s); a larger
            # gap means something is wrong upstream — silently trimming
            # in that case would just compound the loss.
            try:
                from transitions import _ffprobe_video_stream_duration
                v_only = _ffprobe_video_stream_duration(assembled)
                drift = (scene_video.duration - v_only) if v_only else 0
                if v_only and 0 < drift <= 5.0:
                    scene_video = _subclip(scene_video, 0, v_only)
                elif drift > 5.0:
                    print(f"  ⚠ {scene_id}: video stream {v_only:.1f}s "
                          f"vs container {scene_video.duration:.1f}s "
                          f"(>5s gap — NOT auto-trimming; check input clips)")
            except Exception:                                         # noqa: BLE001
                pass
        else:
            shot_clips = []
            for d, src in zip(ordered_decisions, ordered_paths):
                c = VideoFileClip(str(src))
                # Same drift fix as the assembled-scene path: a take
                # whose container duration is longer than its actual
                # video stream (e.g., 6.5s audio over 6.0s video) makes
                # moviepy concatenate read frames past the real video
                # end and pad with the last valid frame — visible as a
                # frozen tail on every shot in cuts-only scenes. Trim
                # to video-stream duration when there's a real gap.
                try:
                    from transitions import _ffprobe_video_stream_duration
                    v_only = _ffprobe_video_stream_duration(Path(src))
                    if v_only and 0 < c.duration - v_only <= 5.0:
                        c = _subclip(c, 0, v_only)
                except Exception:                                  # noqa: BLE001
                    pass
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

        scene_clips.append(_crossfadein(scene_video, 0.5))

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

    print(f"[{exp.exp_id}] Stage 6.5: narration")
    build_narration(exp, script, cast)

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

    # Auto-generate the machine-readable production bible JSON. This is
    # what run_loop.py reads to plan the next iteration's carryover.
    # Always written — even before the critic runs — so partial data is
    # available if the run is inspected mid-loop.
    try:
        from production_bible import build_production_bible_json
        pb_path = build_production_bible_json(exp)
        print(f"Production bible JSON: {pb_path}")
    except Exception as e:  # noqa: BLE001
        print(f"production_bible.json failed (non-fatal): {e}")

    # Auto-generate the production bible PDF. Includes whatever has been
    # produced so far — if metric.json doesn't exist yet, the bible just
    # omits the critique section. After running evaluate.py, run
    # `python bible.py <exp_id>` to refresh the bible with critique data.
    try:
        from bible import build_bible
        bible_path = build_bible(exp)
        size_mb = bible_path.stat().st_size / 1_048_576
        print(f"Production bible PDF: {bible_path}  ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"Bible PDF generation failed (non-fatal): {e}")

    print(f"\nNext: `python evaluate.py {exp.exp_id}` to score it,")
    print(f"  or `python run_loop.py --resume --iterations 3` to keep iterating.")
