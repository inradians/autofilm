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
from pathlib import Path
from typing import Any

from prepare import (
    Experiment,
    GPT_IMAGE_MODEL,
    MAX_PLANNED_SHOT_SECONDS,
    MAX_SCENES,
    NANO_BANANA_MODEL,
    SHOT_DURATION_SECONDS,
    TAKES_PER_SHOT,
    VEO_MODEL_LITE,
    VEO_TIER,
    VIDEO_MODELS,
    book_chunks,
    claude_tool,
    elevenlabs_sfx,
    extract_video_frame,
    ffmpeg,
    gpt_image,
    nano_banana,
    plan_shot_durations,
    route_shot,
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
    for loc in locations:
        slug = loc["slug"]
        out_path = exp.path(f"location_moodboards/{slug}/00.png")
        prompt = (
            f"Cinematic empty-scene location reference photograph, no people. "
            f"{loc['description']}. "
            f"Palette: {', '.join(loc.get('color_palette', []))}. "
            f"Anamorphic, photorealistic, no text/logos."
        )
        if not out_path.exists():
            try:
                img = nano_banana(prompt)
                out_path.write_bytes(img)
            except Exception as e:  # noqa: BLE001
                print(f"  Moodboard failed for {slug}: {e}")
        exp.log_prompt(
            target=f"location_moodboards/{slug}/00.png",
            model=NANO_BANANA_MODEL,
            prompt=prompt,
            stage="moodboard",
        )
        loc["moodboard_paths"] = [str(p) for p in out_path.parent.glob("*.png")]

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
        sf = gpt_image(lookbook["style_frame_prompt"], size="1792x1024", quality="high")
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

def reference_image_prompt(lookbook: dict, character: dict, scene: dict, actor: str) -> str:
    return (
        style_preamble(lookbook)
        + f"Cinematic film still, photorealistic, anamorphic 16:9, "
        f"shallow depth of field. Medium shot, eye-level.\n\n"
        f"SUBJECT: {actor} portraying {character['name']}.\n"
        f"CHARACTER: {character.get('description', '')}\n"
        f"LOCATION: {scene['location']}, {scene.get('time_of_day', 'day')}.\n"
        f"MOOD: {scene.get('mood', 'neutral')}.\n\n"
        f"NEGATIVE: no text, no logos, no UI, no watermarks."
    )


def build_references(exp: Experiment, script: dict, cast: list[dict],
                      locations: list[dict], lookbook: dict) -> dict:
    actor_by_char = {row["character_id"]: row for row in cast}
    char_by_id = {c["id"]: c for c in script["characters"]}
    loc_by_scene: dict[str, dict] = {}
    for loc in locations:
        for sid in loc.get("scene_ids", []):
            loc_by_scene[sid] = loc

    actor_photos_root = exp.path("references/actor_photos")
    manifest: dict[str, dict[str, str]] = {}

    for scene in script["scenes"]:
        scene_id = scene["id"]
        manifest.setdefault(scene_id, {})
        loc = loc_by_scene.get(scene_id)
        moodboards = []
        if loc:
            moodboards = [Path(p).read_bytes() for p in loc.get("moodboard_paths", [])
                          if Path(p).exists()][:1]

        for cid in scene.get("characters", []):
            character = char_by_id.get(cid)
            cast_row = actor_by_char.get(cid)
            if not character or not cast_row:
                continue
            out_path = exp.path(f"references/{cid}/{scene_id}.png")
            if out_path.exists():
                manifest[scene_id][cid] = str(out_path)
                continue

            actor = cast_row["actor"]
            prompt = reference_image_prompt(lookbook, character, scene, actor)

            try:
                # Step 1: GPT Image 2 composes the scene.
                composition = gpt_image(prompt, size="1792x1024", quality="high")
                exp.log_prompt(
                    target=f"references/{cid}/{scene_id}.composition.png",
                    model=GPT_IMAGE_MODEL,
                    prompt=prompt,
                    stage="reference_composition",
                    character=cid,
                    scene=scene_id,
                )

                # Step 2: Nano Banana 2 fuses composition + actor photos
                # (if user dropped any in actor_photos/) + location moodboard.
                actor_slug = actor.lower().replace(" ", "_")
                photos_dir = actor_photos_root / actor_slug
                actor_imgs: list[bytes] = []
                if photos_dir.exists():
                    for p in sorted(list(photos_dir.glob("*.png")) + list(photos_dir.glob("*.jpg")))[:3]:
                        actor_imgs.append(p.read_bytes())

                refs = [composition] + actor_imgs + moodboards
                if len(refs) > 1:
                    lock_prompt = (
                        prompt + "\n\nUse FIRST image as composition, then character "
                        "identity refs, then location moodboard. Preserve face exactly."
                    )
                    locked = nano_banana(lock_prompt, reference_images=refs)
                    exp.log_prompt(
                        target=f"references/{cid}/{scene_id}.png",
                        model=NANO_BANANA_MODEL,
                        prompt=lock_prompt,
                        stage="reference_lock",
                        character=cid,
                        scene=scene_id,
                        n_reference_images=len(refs),
                    )
                else:
                    locked = composition

                out_path.write_bytes(locked)
                manifest[scene_id][cid] = str(out_path)
            except Exception as e:  # noqa: BLE001
                print(f"  Reference {cid}/{scene_id} failed: {e}")

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

def build_music(exp: Experiment, script: dict) -> None:
    for scene in script["scenes"]:
        scene_id = scene["id"]
        out_path = exp.path(f"music/{scene_id}.wav")
        # Per-scene mood overlay onto the locked MUSIC_STYLE.
        prompt = f"{MUSIC_STYLE}. Scene mood: {scene.get('mood', '')}. {scene.get('summary', '')[:200]}"
        if not out_path.exists():
            try:
                audio = stable_audio(prompt, duration_seconds=30)
                out_path.write_bytes(audio)
            except Exception as e:  # noqa: BLE001
                print(f"  Music {scene_id} failed: {e}")
        exp.log_prompt(
            target=f"music/{scene_id}.wav",
            model="stable-audio-2.5",
            prompt=prompt,
            stage="music",
            scene=scene_id,
            duration_seconds=30,
        )


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
    for scene in script["scenes"]:
        scene_id = scene["id"]
        manifest.setdefault(scene_id, {})
        chars = [char_by_id[c] for c in scene.get("characters", []) if c in char_by_id]
        actors = [actor_by_char.get(c["id"], c["name"]) for c in chars]

        for shot in storyboard.get(scene_id, []):
            shot_id = shot["shot_id"]
            out_path = exp.path(f"frames/{scene_id}/{shot_id}.png")
            if out_path.exists():
                manifest[scene_id][shot_id] = str(out_path)
                continue

            prompt = first_frame_prompt(lookbook, shot, scene, chars, actors,
                                         loc_by_scene.get(scene_id))
            try:
                composition = gpt_image(prompt, size="1792x1024", quality="high")
                exp.log_prompt(
                    target=f"frames/{scene_id}/{shot_id}.composition.png",
                    model=GPT_IMAGE_MODEL,
                    prompt=prompt,
                    stage="first_frame_composition",
                    scene=scene_id,
                    shot=shot_id,
                )
                # Lock with character refs from stage 4.
                ref_imgs = []
                for c in chars:
                    rp = exp.path(f"references/{c['id']}/{scene_id}.png")
                    if rp.exists():
                        ref_imgs.append(rp.read_bytes())
                if ref_imgs:
                    lock_prompt = (
                        prompt + "\n\nUse FIRST image as composition, then character "
                        "identity refs. Preserve faces exactly."
                    )
                    final = nano_banana(lock_prompt, reference_images=[composition] + ref_imgs)
                    exp.log_prompt(
                        target=f"frames/{scene_id}/{shot_id}.png",
                        model=NANO_BANANA_MODEL,
                        prompt=lock_prompt,
                        stage="first_frame_lock",
                        scene=scene_id,
                        shot=shot_id,
                        n_reference_images=1 + len(ref_imgs),
                    )
                else:
                    final = composition
                out_path.write_bytes(final)
                manifest[scene_id][shot_id] = str(out_path)
            except Exception as e:  # noqa: BLE001
                print(f"  First frame {scene_id}/{shot_id} failed: {e}")

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

            ref_imgs = []
            for c in chars:
                rp = exp.path(f"references/{c['id']}/{scene_id}.png")
                if rp.exists():
                    ref_imgs.append(rp.read_bytes())

            # Look up routing for this shot.
            route = shot_plan.get(scene_id, {}).get(shot_id)
            if not route:
                # Fallback: route on the fly.
                route = route_shot(shot.get("duration_seconds", 8), tier=VEO_TIER)

            model_key = route["model_key"]
            segments = route["segments"]
            total_duration = sum(segments)
            print(f"  {scene_id}/{shot_id}: routing → {model_key} "
                  f"({total_duration}s, {len(segments)} segment{'s' if len(segments) > 1 else ''})")

            for take_idx in range(TAKES_PER_SHOT):
                take_path = exp.path(f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4")
                prompt = veo_prompt(lookbook, shot, scene, chars, actors, take_idx)
                exp.log_prompt(
                    target=f"clips/{scene_id}/{shot_id}/take_{take_idx + 1}.mp4",
                    model=route["model_id"],
                    prompt=prompt,
                    stage="video",
                    scene=scene_id,
                    shot=shot_id,
                    take=take_idx + 1,
                    duration_seconds=route["segments"][0],
                    estimated_cost=route["estimated_cost"],
                )
                if take_path.exists():
                    manifest[scene_id][shot_id].append(str(take_path))
                    continue
                try:
                    video_bytes = _render_shot(
                        route, prompt, ff.read_bytes(), ref_imgs,
                        seed=1000 + take_idx * 137,
                    )
                    take_path.write_bytes(video_bytes)
                    manifest[scene_id][shot_id].append(str(take_path))
                except Exception as e:  # noqa: BLE001
                    print(f"  Take {scene_id}/{shot_id}/{take_idx + 1} failed: {e}")

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
    from moviepy.editor import (
        AudioFileClip, CompositeAudioClip, VideoFileClip,
        concatenate_videoclips, afx,
    )

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
                    c = c.subclip(in_s, min(out_s, c.duration))
                elif in_s > 0:
                    c = c.subclip(in_s)
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
            a = AudioFileClip(str(ambient_path)).fx(afx.volumex, 0.16)
            a = (a.fx(afx.audio_loop, duration=scene_video.duration)
                 if a.duration < scene_video.duration
                 else a.subclip(0, scene_video.duration))
            layers.append(a)
        music_path = exp.path(f"music/{scene_id}.wav")
        if music_path.exists():
            m = AudioFileClip(str(music_path)).fx(afx.volumex, 0.20)
            m = (m.fx(afx.audio_loop, duration=scene_video.duration)
                 if m.duration < scene_video.duration
                 else m.subclip(0, scene_video.duration))
            layers.append(m)
        if layers:
            scene_video = scene_video.set_audio(CompositeAudioClip(layers))

        scene_clips.append(scene_video.crossfadein(0.5))

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
    build_music(exp, script)

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
    exp = Experiment.new()
    print(f"=== {exp.exp_id} ===")
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
