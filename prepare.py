"""prepare.py — fixed scaffolding for the book-to-film pipeline.

This file is NEVER modified by the agent. It contains:
  - All API clients with retry policy
  - Constants (model IDs, file paths, the SOTA stack as of April 2026)
  - Book parsing utilities
  - The fixed execution shell that runs the produce.py pipeline
  - The film_loss evaluation function (the single scalar metric)
  - State/artifact management

The agent edits `produce.py` to change creative parameters (prompts, look
book, shot lists, take strategy, edit decisions). Everything pipe-shaped
lives here; everything taste-shaped lives in produce.py.

After producing a film the agent reads `film_loss` from output/metric.json
and decides what to change in produce.py for the next experiment.

Layout under output/exp_NNN/:
  produce.py           # snapshot of produce.py used for this experiment
  script.json          # parsed screenplay
  cast.json            # actor casting
  locations.json       # location moodboards (paths)
  lookbook.json        # locked visual bible
  storyboard.json      # shot list
  frames/{scene}/{shot}.png
  clips/{scene}/{shot}/take_N.mp4
  edl.json             # edit decision list
  music/{scene}.wav
  sfx/{scene}/ambient.wav
  final.mp4            # the deliverable
  critique.md          # prose critique
  metric.json          # film_loss + per-axis scores  ← THE METRIC
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import anthropic
import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env from project root if present.
try:
    from dotenv import load_dotenv
    _DOTENV_PATH = Path(__file__).resolve().parent / ".env"
    if _DOTENV_PATH.exists():
        load_dotenv(_DOTENV_PATH)
except ImportError:
    pass

# Optional imports loaded lazily so partial environments still work.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore


# ============================================================================
# CONSTANTS — SOTA stack as of April 2026
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Source book — Crichton's Jurassic Park
BOOK_PDF_PATH = Path(
    os.getenv("BOOK_PDF_PATH", "/mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf")
)

# Cap on how many scenes to render per experiment. Keeps each run on a
# fixed budget, like autoresearch's 5-min training cap.
MAX_SCENES = int(os.getenv("MAX_SCENES", "3"))

# --- Models (the canonical, SOTA-as-of-April-2026 lineup) ---
CLAUDE_MODEL = "claude-opus-4-7"
GPT_IMAGE_MODEL = "gpt-image-2"
NANO_BANANA_MODEL = "gemini-3.1-flash-image-preview"
VEO_MODEL_LITE = "veo-3.1-lite-generate-preview"
VEO_MODEL_FAST = "veo-3.1-fast-generate-preview"
VEO_MODEL_STANDARD = "veo-3.1-generate-preview"
GEMINI_PRO_MODEL = "gemini-3-pro"

# --- Video model registry ---
# Single source of truth for shot-list planning. Veo 3.1's native single-call
# cap is 8 seconds; we use that as the hard ceiling for shot duration. Three
# tiers trade cost for quality at the same 8s cap. Numbers verified against
# vendor docs April 2026.
VIDEO_MODELS = {
    "veo3.1_lite": {
        "id":               VEO_MODEL_LITE,
        "vendor":           "google",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     True,
        "cost_per_sec":     0.07,
        "use_for":          "previs",
    },
    "veo3.1_fast": {
        "id":               VEO_MODEL_FAST,
        "vendor":           "google",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     True,
        "cost_per_sec":     0.15,
        "use_for":          "iteration",
    },
    "veo3.1_standard": {
        "id":               VEO_MODEL_STANDARD,
        "vendor":           "google",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "4K",  # preview adds 4K at same length cap
        "native_audio":     True,
        "cost_per_sec":     0.40,
        "use_for":          "hero",
    },
}

# --- Render budget. Capped tightly so iteration stays cheap. ---
ASPECT_RATIO = "16:9"
SHOT_DURATION_SECONDS = int(os.getenv("SHOT_DURATION_SECONDS", "8"))
VEO_TIER = os.getenv("VEO_TIER", "fast").lower()  # fast | standard
VEO_RESOLUTION = os.getenv("VEO_RESOLUTION", "720p")
TAKES_PER_SHOT = int(os.getenv("TAKES_PER_SHOT", "1"))  # 1 for cheap iteration


def veo_final_model() -> str:
    return VEO_MODEL_STANDARD if VEO_TIER == "standard" else VEO_MODEL_FAST


# ============================================================================
# API CLIENT FACTORIES (cached singletons)
# ============================================================================
@lru_cache(maxsize=1)
def claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=_require_key("ANTHROPIC_API_KEY"),
        timeout=httpx.Timeout(180.0, connect=10.0),
        max_retries=3,
    )


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    return OpenAI(
        api_key=_require_key("OPENAI_API_KEY"),
        timeout=httpx.Timeout(180.0, connect=10.0),
        max_retries=3,
    )


@lru_cache(maxsize=1)
def gemini_client():
    """Drives Nano Banana 2 (image), Veo 3.1 (video), Gemini 3 Pro (critique)."""
    from google import genai  # type: ignore
    return genai.Client(api_key=_require_key("GOOGLE_AI_API_KEY"))


@lru_cache(maxsize=1)
def elevenlabs_client():
    from elevenlabs.client import ElevenLabs  # type: ignore
    return ElevenLabs(api_key=_require_key("ELEVENLABS_API_KEY"))


def _require_key(name: str) -> str:
    val = os.getenv(name, "")
    if not val or val.endswith("...") or val == "":
        raise RuntimeError(
            f"Missing API key: {name}. Copy .env.example to .env and set it."
        )
    return val


# Decorator for exponential-backoff retry on flaky model APIs.
api_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)


# ============================================================================
# BOOK PARSING (deterministic, not modified by agent)
# ============================================================================
def extract_book_pages() -> dict[int, str]:
    """Return {page_number: text} for the full book."""
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed; pip install pdfplumber")
    pages: dict[int, str] = {}
    with pdfplumber.open(BOOK_PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages[i] = page.extract_text() or ""
    return pages


def book_chunks(pages_per_chunk: int = 25) -> list[tuple[int, int, str]]:
    """Return [(start_page, end_page, text), ...]."""
    pages = extract_book_pages()
    total = max(pages.keys())
    out: list[tuple[int, int, str]] = []
    for s in range(1, total + 1, pages_per_chunk):
        e = min(s + pages_per_chunk - 1, total)
        text = "\n\n".join(f"[Page {i}]\n{pages.get(i, '')}" for i in range(s, e + 1))
        out.append((s, e, text))
    return out


# ============================================================================
# EXPERIMENT STATE
# ============================================================================
@dataclass
class Experiment:
    """One experiment = one full pipeline run with a particular produce.py."""
    exp_id: str
    root: Path

    @classmethod
    def new(cls) -> "Experiment":
        existing = sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
        n = len(existing) + 1
        exp_id = f"exp_{n:03d}"
        root = EXPERIMENTS_DIR / exp_id
        root.mkdir()
        # Snapshot produce.py for reproducibility.
        produce_src = PROJECT_ROOT / "produce.py"
        if produce_src.exists():
            (root / "produce.py").write_text(produce_src.read_text())
        return cls(exp_id=exp_id, root=root)

    @classmethod
    def load(cls, exp_id: str) -> "Experiment":
        return cls(exp_id=exp_id, root=EXPERIMENTS_DIR / exp_id)

    # --- Artifact I/O ---
    def write_json(self, name: str, data: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return path

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text())

    def has(self, name: str) -> bool:
        return (self.root / name).exists()

    def write_bytes(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def path(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_prompt(self, target: str, model: str, prompt: str, **meta: Any) -> None:
        """Record the prompt used to generate one artifact.

        Writes (or upserts) into prompts.json. Bible's prompts section
        reads this and groups by model. Idempotent: re-logging the same
        target replaces the prior entry — useful for cached re-runs.
        """
        log_path = self.root / "prompts.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
        else:
            log = {}
        entry: dict = {"model": model, "prompt": prompt}
        if meta:
            entry["meta"] = meta
        log[target] = entry
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))


# ============================================================================
# RUNTIME UTILITIES (used by produce.py)
# ============================================================================

@api_retry
def gpt_image(prompt: str, size: str = "1792x1024", quality: str = "high") -> bytes:
    """GPT Image 2 — best instruction-following image gen."""
    resp = openai_client().images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    return base64.b64decode(resp.data[0].b64_json)  # type: ignore[union-attr]


@api_retry
def nano_banana(prompt: str, reference_images: list[bytes] | None = None) -> bytes:
    """Nano Banana 2 — identity locking, multi-image fusion (up to 14 elements)."""
    from google.genai import types  # type: ignore
    parts: list = [prompt]
    for img in (reference_images or [])[:8]:
        parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))
    resp = gemini_client().models.generate_content(
        model=NANO_BANANA_MODEL,
        contents=parts,
    )
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None):
            return part.inline_data.data
    raise RuntimeError("Nano Banana 2: no image returned")


@api_retry
def veo(
    prompt: str,
    first_frame: bytes,
    reference_images: list[bytes] | None = None,
    model: str | None = None,
    duration_seconds: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
) -> bytes:
    """Veo 3.1 image-to-video with native synchronized audio."""
    from google.genai import types  # type: ignore
    client = gemini_client()
    start_image = types.Image(image_bytes=first_frame, mime_type="image/png")
    refs = [
        types.VideoGenerationReferenceImage(
            image=types.Image(image_bytes=r, mime_type="image/png"),
            reference_type="asset",
        )
        for r in (reference_images or [])[:3]
    ]
    cfg: dict = {
        "aspect_ratio": ASPECT_RATIO,
        "resolution": resolution or VEO_RESOLUTION,
        "duration_seconds": duration_seconds or SHOT_DURATION_SECONDS,
    }
    if refs:
        cfg["reference_images"] = refs
    if seed is not None:
        cfg["seed"] = seed
    op = client.models.generate_videos(
        model=model or veo_final_model(),
        prompt=prompt,
        image=start_image,
        config=types.GenerateVideosConfig(**cfg),
    )
    deadline = time.time() + 600
    while not op.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo operation {op.name} did not complete")
        time.sleep(8)
        op = client.operations.get(op)
    video = op.result.generated_videos[0].video
    try:
        b = client.files.download(file=video)
        if isinstance(b, bytes):
            return b
    except Exception:
        pass
    if hasattr(video, "video_bytes") and video.video_bytes:
        return video.video_bytes
    if hasattr(video, "uri") and video.uri:
        return httpx.get(video.uri, timeout=120).content
    raise RuntimeError("Veo: could not extract video bytes")


@api_retry
def claude_tool(system: str, user_content: Any, tool_name: str, tool_schema: dict,
                max_tokens: int = 4000) -> dict:
    """Call Claude Opus 4.7 with a forced tool call — returns the tool input dict."""
    resp = claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        tools=[{"name": tool_name, "description": tool_schema.get("description", ""),
                "input_schema": tool_schema["input_schema"]}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user_content}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input  # type: ignore[return-value]
    raise RuntimeError(f"Claude tool '{tool_name}' returned no tool_use block")


@api_retry
def stable_audio(prompt: str, duration_seconds: int = 30) -> bytes:
    """Stability Stable Audio 2.5 — instrumental cinematic cues."""
    api_key = _require_key("STABILITY_API_KEY")
    resp = httpx.post(
        "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "audio/wav"},
        files={"none": ""},
        data={"prompt": prompt, "duration": min(duration_seconds, 47), "output_format": "wav"},
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.content


@api_retry
def elevenlabs_sfx(prompt: str, duration_seconds: int = 10) -> bytes:
    """ElevenLabs Sound Effects — ambient beds and Foley."""
    el = elevenlabs_client()
    audio_iter = el.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=min(max(duration_seconds, 1), 22),
    )
    return b"".join(audio_iter)


def ffmpeg(args: list[str]) -> None:
    """Run ffmpeg, raising on error."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def extract_video_frame(video_path: Path, at_seconds: float = 1.0) -> bytes:
    """Pull a single PNG frame from a video for vision-model review."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(at_seconds),
            "-i", str(video_path),
            "-frames:v", "1", "-f", "image2", "-vcodec", "png",
            "pipe:1",
        ],
        capture_output=True, check=True,
    )
    return result.stdout


# ============================================================================
# SHOT ROUTING — pick the right video model + duration for each shot
# ============================================================================
# Given a desired shot duration in seconds, this routing function returns:
#   - which model key from VIDEO_MODELS to use
#   - whether the shot needs to be chained (extension calls)
#   - the per-segment durations so a caller can iterate
#
# Routing logic (verified April 2026 model limits):
#   - ≤ 8s        → Veo 3.1 single call (best at this length, native audio)
#   - 9-15s       → Kling 3.0 single call (15s native cap, no chaining)
#   - 16-60s      → Kling 3.0 chained 5s extensions (oner shots only)
#   - > 60s       → forbidden; coherence collapses across all models

# Hard cap on shot duration. Veo 3.1's native single-call max is 8 seconds
# at 24fps, 1080p. Anything longer would require extension chaining, which
# accumulates character drift and color shift, so we don't go there — we
# break long beats into multiple shots in the storyboard instead.
MAX_PLANNED_SHOT_SECONDS = 8


def route_shot(
    desired_seconds: float,
    tier: str = "fast",
) -> dict:
    """Pick which Veo tier renders a shot.

    Args:
        desired_seconds: how long the shot should be on screen (capped at 8).
        tier: "fast" | "standard" | "previs". Standard is Veo Quality at
              ~$0.40/sec for hero shots; previs is Veo Lite at ~$0.07/sec
              for cheap blocking validation.

    Returns a routing dict:
        {
          "model_key": str,           # key into VIDEO_MODELS
          "model_id":  str,           # the actual model identifier
          "segments":  [int],         # always single-element since cap is 8s
          "estimated_cost": float,    # USD
          "rationale": str,           # for the bible / debugging
        }
    """
    desired_seconds = max(1, min(desired_seconds, MAX_PLANNED_SHOT_SECONDS))

    if tier == "previs":
        model_key = "veo3.1_lite"
    elif tier == "standard":
        model_key = "veo3.1_standard"
    else:
        model_key = "veo3.1_fast"
    m = VIDEO_MODELS[model_key]
    d = _snap_to_options(desired_seconds, m["duration_options"])
    return {
        "model_key": model_key,
        "model_id":  m["id"],
        "segments":  [d],
        "estimated_cost": round(d * m["cost_per_sec"], 2),
        "rationale": f"{model_key.replace('_', ' ')} single call at {d}s",
    }


def _snap_to_options(seconds: float, options: list[int]) -> int:
    """Snap desired_seconds to the smallest allowed option that's >= it,
    so a planned 7s lands as Veo's 8s (easier to trim than under-shoot)."""
    seconds = float(seconds)
    options_sorted = sorted(options)
    for o in options_sorted:
        if seconds <= o:
            return o
    return options_sorted[-1]


def plan_shot_durations(storyboard: dict) -> dict:
    """Walk a storyboard and produce a routing plan for every shot.

    Returns a dict keyed by (scene_id, shot_id) with the routing info
    plus aggregate cost. Useful as a single source of truth for the
    bible's shot-list section and for produce.py's video stage.
    """
    plan: dict[str, Any] = {}
    total_cost = 0.0
    total_seconds = 0
    by_model: dict[str, int] = {}
    for scene_id, shots in storyboard.items():
        plan.setdefault(scene_id, {})
        for shot in shots:
            d = int(shot.get("duration_seconds", 8))
            route = route_shot(d, tier=VEO_TIER)
            plan[scene_id][shot["shot_id"]] = route
            total_cost += route["estimated_cost"]
            total_seconds += sum(route["segments"])
            by_model[route["model_key"]] = by_model.get(route["model_key"], 0) + 1
    plan["_aggregate"] = {
        "total_seconds": total_seconds,
        "estimated_cost_usd": round(total_cost, 2),
        "shots_by_model": by_model,
    }
    return plan


# ============================================================================
# THE METRIC: film_loss — single scalar, lower is better
# ============================================================================
# Six axes scored 0-1 each (0=perfect, 1=broken). Weighted sum = film_loss.
# Weights chosen so that "professional baseline" is around 0.30-0.40 and
# truly broken films are above 0.70.

LOSS_WEIGHTS = {
    "cinematography": 0.20,  # composition, framing, lensing, camera movement
    "color":          0.15,  # grade consistency, palette adherence
    "sound":          0.15,  # dialogue clarity, music fit, ambient layering
    "acting":         0.20,  # performance, lip-sync, line readings
    "continuity":     0.15,  # visual + character + spatial continuity
    "fidelity":       0.15,  # faithfulness to source novel
}


CRITIC_SYSTEM = """You are a senior film critic with a background in
cinematography and post-production. You will watch a generated film
adaptation and score it on six axes, each 0.0 to 1.0 where 0.0 is
flawless professional work and 1.0 is unwatchably bad.

Be calibrated and honest. A reasonable AI-generated short would score
~0.30-0.45 on most axes today. Don't grade on a curve — if a shot has
visible character drift or a line that doesn't lip-sync, say so.

For each axis you must:
  - Give a numerical score (0.0 to 1.0, two decimals).
  - Cite at least one specific timestamp or shot ID when criticizing.
  - Explain the reasoning in 2-4 sentences of prose.

Then produce a list of structured CHANGES — actionable edits the
producing agent should make to its produce.py for the next experiment.
Each change names the function/parameter to modify and the suggested
direction. Be specific: "increase TAKES_PER_SHOT from 1 to 2 for shots
with dialogue" not "improve acting".

THE SIX AXES:

1. CINEMATOGRAPHY (composition, framing, lensing, camera movement)
   - Are shots motivated and varied?
   - Does coverage tell the story?
   - Are camera moves purposeful?

2. COLOR (grade consistency, palette adherence)
   - Does the film look like one film, not a montage?
   - Does the grade match the look book intent?
   - Are skin tones natural where they should be?

3. SOUND (dialogue clarity, music fit, ambient layering)
   - Is dialogue intelligible?
   - Does music support, not overwhelm?
   - Is ambient bed plausible for the location?

4. ACTING (performance, lip-sync, line readings)
   - Do faces emote on the right beats?
   - Does mouth movement match dialogue?
   - Do line readings have appropriate energy/restraint?

5. CONTINUITY (visual + character + spatial continuity)
   - Does each character look the same across shots?
   - Do edits respect screen direction?
   - Is wardrobe/lighting consistent within a scene?

6. FIDELITY (faithfulness to source novel)
   - Are the right beats from the book present?
   - Are characters recognizable as their book counterparts?
   - Has the adaptation preserved the tone?

Return ONLY via the tool."""


CRITIC_TOOL_SCHEMA = {
    "description": "Submit per-axis scores, prose critique, and structured fixes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    "cinematography": {"type": "number", "minimum": 0, "maximum": 1},
                    "color":          {"type": "number", "minimum": 0, "maximum": 1},
                    "sound":          {"type": "number", "minimum": 0, "maximum": 1},
                    "acting":         {"type": "number", "minimum": 0, "maximum": 1},
                    "continuity":     {"type": "number", "minimum": 0, "maximum": 1},
                    "fidelity":       {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["cinematography", "color", "sound", "acting",
                             "continuity", "fidelity"],
            },
            "prose_critique_markdown": {"type": "string"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "axis": {"enum": ["cinematography", "color", "sound", "acting",
                                          "continuity", "fidelity"]},
                        "priority": {"enum": ["low", "medium", "high"]},
                        "target": {
                            "type": "string",
                            "description": "Function or parameter in produce.py to modify, e.g. 'shot_list_for_scene', 'LOOKBOOK_GRADE', 'TAKES_PER_SHOT'",
                        },
                        "current_behavior": {"type": "string"},
                        "suggested_change": {"type": "string"},
                        "expected_impact": {"type": "string"},
                    },
                    "required": ["axis", "priority", "target", "suggested_change"],
                },
                "minItems": 3,
            },
        },
        "required": ["scores", "prose_critique_markdown", "changes"],
    },
}


def evaluate_film(exp: Experiment) -> dict:
    """Run the critic over the finished film + write metric.json.

    Combines:
      - Gemini 3 Pro: native long-video review with timestamp citations
      - Claude Opus 4.7: independent second-opinion on representative stills
      - CLIP: numerical character-identity drift across shots

    Final film_loss is the weighted sum of the six axis scores from the
    averaged Gemini + Claude rubrics. CLIP drift is reported separately
    but does NOT modify film_loss directly — the human-style critics
    factor it into 'continuity' implicitly.
    """
    final_path = exp.path("final.mp4")
    if not final_path.exists():
        raise RuntimeError(f"final.mp4 missing for {exp.exp_id} — produce.py failed")

    script = exp.read_json("script.json")

    # --- Reviewer A: Gemini 3 Pro on the actual video ---
    gemini_result = _critic_gemini_video(final_path, script)

    # --- Reviewer B: Claude on representative stills ---
    frames = sorted(exp.path("frames").rglob("*.png"))
    step = max(1, len(frames) // 16)
    sampled_frames = frames[::step][:16]
    claude_result = _critic_claude_stills(script, sampled_frames)

    # --- Average the two reviewers' scores per axis ---
    avg_scores = {}
    for axis in LOSS_WEIGHTS:
        avg_scores[axis] = (
            gemini_result["scores"][axis] + claude_result["scores"][axis]
        ) / 2.0

    film_loss = sum(LOSS_WEIGHTS[a] * avg_scores[a] for a in LOSS_WEIGHTS)

    # --- CLIP drift (informational) ---
    clip_scores = _clip_drift(exp)

    # --- Write critique markdown ---
    md = "# Film Critique\n\n"
    md += f"## Final film_loss = {film_loss:.4f}\n\n"
    md += "Per-axis (lower is better):\n\n"
    for axis, score in avg_scores.items():
        md += f"- **{axis}**: {score:.3f}  (weight {LOSS_WEIGHTS[axis]:.2f})\n"
    md += "\n## Reviewer A — Gemini 3 Pro (video)\n\n"
    md += gemini_result["prose_critique_markdown"] + "\n\n"
    md += "## Reviewer B — Claude Opus 4.7 (stills)\n\n"
    md += claude_result["prose_critique_markdown"]
    if clip_scores:
        md += "\n\n## CLIP character-identity drift (informational)\n\n"
        md += "Cosine similarity, lower = more drift. <0.70 means actor lost.\n\n"
        for k, v in sorted(clip_scores.items()):
            md += f"- **{k}**: {v:.3f}\n"
    exp.write_bytes("critique.md", md.encode())

    # --- Merge changes from both reviewers, prioritized by impact ---
    all_changes = []
    for c in gemini_result.get("changes", []):
        all_changes.append({**c, "reviewer": "gemini"})
    for c in claude_result.get("changes", []):
        all_changes.append({**c, "reviewer": "claude"})
    # Sort: high-priority first, then medium, then low.
    pri = {"high": 0, "medium": 1, "low": 2}
    all_changes.sort(key=lambda c: pri.get(c.get("priority", "low"), 3))

    metric = {
        "film_loss": film_loss,
        "scores": avg_scores,
        "weights": LOSS_WEIGHTS,
        "reviewers": {
            "gemini_scores": gemini_result["scores"],
            "claude_scores": claude_result["scores"],
        },
        "clip_drift": clip_scores,
        "changes": all_changes,
    }
    exp.write_json("metric.json", metric)
    return metric


def _critic_gemini_video(final_path: Path, script: dict) -> dict:
    """Gemini 3 Pro reviews the actual video with timestamp citations."""
    from google.genai import types  # type: ignore
    client = gemini_client()
    uploaded = client.files.upload(file=str(final_path))
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded video")

    user_prompt = (
        "Watch the attached film end-to-end. Score it on the six axes "
        "described in your system prompt and return everything via the tool.\n\n"
        f"SOURCE SCRIPT (truncated to 15K chars):\n"
        f"{json.dumps(script)[:15000]}"
    )

    resp = client.models.generate_content(
        model=GEMINI_PRO_MODEL,
        contents=[uploaded, user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=CRITIC_SYSTEM,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name="submit_review",
                description=CRITIC_TOOL_SCHEMA["description"],
                parameters=CRITIC_TOOL_SCHEMA["input_schema"],
            )])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY"),
            ),
        ),
    )

    for part in resp.candidates[0].content.parts:
        fc = getattr(part, "function_call", None)
        if fc and fc.name == "submit_review":
            return dict(fc.args)
    raise RuntimeError("Gemini critic: no function_call returned")


def _critic_claude_stills(script: dict, frames: list[Path]) -> dict:
    """Claude reviews representative stills as a second opinion."""
    content: list[dict] = [
        {"type": "text", "text": (
            "Score this generated film on the six axes via the tool.\n\n"
            f"SOURCE SCRIPT (truncated):\n{json.dumps(script)[:15000]}\n\n"
            f"REPRESENTATIVE STILLS:\n"
        )},
    ]
    for p in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(p.read_bytes()).decode(),
            },
        })
        content.append({"type": "text", "text": f"^ {p.parent.name}/{p.stem}"})

    return claude_tool(
        system=CRITIC_SYSTEM,
        user_content=content,
        tool_name="submit_review",
        tool_schema=CRITIC_TOOL_SCHEMA,
        max_tokens=8000,
    )


def _clip_drift(exp: Experiment) -> dict[str, float]:
    """Cosine similarity between each character's canonical reference and
    every frame they appear in. Lower = more drift."""
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        return {}

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()

    def embed(path: Path):
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return model.encode_image(img).squeeze(0)

    refs_dir = exp.path("references")
    scores: dict[str, list[float]] = {}
    for char_dir in refs_dir.iterdir():
        if not char_dir.is_dir() or char_dir.name == "actor_photos":
            continue
        ref_imgs = list(char_dir.glob("*.png"))
        if not ref_imgs:
            continue
        canonical = embed(ref_imgs[0])
        canonical = canonical / canonical.norm()

        frames_dir = exp.path("frames")
        if not frames_dir.exists():
            continue
        for scene_dir in frames_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            for frame in scene_dir.glob("*.png"):
                f_emb = embed(frame)
                f_emb = f_emb / f_emb.norm()
                sim = float((canonical @ f_emb).item())
                scores.setdefault(char_dir.name, []).append(sim)
    return {k: sum(v) / len(v) for k, v in scores.items() if v}
