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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
import threading

import anthropic
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Runway proxies image, video, and SFX generation through one API.
# See https://docs.dev.runwayml.com/ and runway-skills/ in this repo.
from runwayml import RunwayML

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

# Source book — set BOOK_PDF_PATH env var to point at any PDF. The
# pipeline derives a per-book slug from the filename so multiple books
# can coexist under experiments/. No book-specific default is hardcoded;
# the UI server and CLI runner are both responsible for providing this.
BOOK_PDF_PATH = Path(os.getenv("BOOK_PDF_PATH", ""))

# Cap on how many scenes to render per experiment. Keeps each run on a
# fixed budget, like autoresearch's 5-min training cap.
MAX_SCENES = int(os.getenv("MAX_SCENES", "3"))

# --- Models (SOTA stack as of May 2026, accessed via Runway + Anthropic + Stability) ---
# Anthropic: Claude Opus 4.7 (text/critic-stills, direct API).
# Google AI: Gemini 3 Pro (long-video critic only — Runway has no video-review LLM).
# Runway: image, video, and SFX generation. See https://docs.dev.runwayml.com/guides/models/
# Stability: Stable Audio 2.5 (music score — Runway has no music model).
CLAUDE_MODEL = "claude-opus-4-7"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

# Runway image model IDs. The Runway endpoint is /v1/text_to_image with the
# `model` field selecting which generator. All three accept `referenceImages`
# (gen4_image_turbo *requires* it). Pricing is in credits at $0.01/credit.
GPT_IMAGE_MODEL    = "gpt_image_2"          # 1-41 credits/image (high@1K=20, high@4K=41)
NANO_BANANA_MODEL  = "gemini_image3_pro"    # 20 credits @ 1K/2K, 40 credits @ 4K
GEN4_IMAGE_MODEL   = "gen4_image"           # 5 credits @ 720p, 8 credits @ 1080p — native ref-image support
GEN4_IMAGE_TURBO   = "gen4_image_turbo"     # 2 credits, references REQUIRED
GEMINI_FLASH_MODEL = "gemini_2.5_flash"     # 5 credits, any resolution

# ── Google direct backend ────────────────────────────────────────────────────
# When VIDEO_BACKEND=google or IMAGE_BACKEND=google in produce.py, the
# pipeline calls these instead of the Runway-proxied equivalents. No daily
# task limits — billed directly against your GOOGLE_AI_API_KEY quota.
GOOGLE_VEO_MODEL          = os.getenv("GOOGLE_VIDEO_MODEL", "veo-3.1-generate-preview")
GOOGLE_IMAGE_MODEL        = "imagen-3.0-generate-002"
# Gemini 2.5 Flash Image (aka Nano Banana) — text-to-image with reference
# image support via the Gemini API directly. Same model that Runway proxies
# as gemini_image3_pro, but accessed without going through Runway.
GOOGLE_NANO_BANANA_MODEL  = os.getenv("GOOGLE_NANO_BANANA_MODEL", "gemini-2.5-flash-image")

# Reve API — api.reve.com
# State-of-the-art image generation with create, remix (multi-ref), and edit.
# Remix accepts up to 6 reference images — excellent for character consistency.
REVE_API_BASE = "https://api.reve.com"

# LTX 2.3 — Lightricks API (api.ltx.video)
# ltx-2-3-pro  → best quality, all endpoints, ~$0.05-0.12/s @ 720p
# ltx-2-3-fast → rapid iteration, text-to-video + image-to-video only
LTX_API_BASE    = "https://api.ltx.video"
# ltx-2-3-pro:  best quality, durations 6/8/10s, all endpoints
# ltx-2-3-fast: faster/cheaper, durations 6/8/10/12/14/16/18/20s
LTX_PRO_MODEL   = "ltx-2-3-pro"
LTX_FAST_MODEL  = "ltx-2-3-fast"
LTX_VIDEO_MODEL = os.getenv("LTX_VIDEO_MODEL", LTX_PRO_MODEL)

# Valid durations per LTX model (must be an exact match)
_LTX_DURATIONS: dict[str, list[int]] = {
    LTX_PRO_MODEL:  [6, 8, 10],
    LTX_FAST_MODEL: [6, 8, 10, 12, 14, 16, 18, 20],
    "ltx-2-pro":    [5, 9, 13, 17, 21],
    "ltx-2-fast":   [5, 9, 13, 17, 21],
}
# Valid resolutions for LTX API — minimum is 1080p, no 720p support
_LTX_RESOLUTIONS: dict[str, str] = {
    "1080p": "1920x1080",
    "1440p": "2560x1440",
    "4k":    "3840x2160",
}


@lru_cache(maxsize=1)
def _genai_client():
    """Lazy Google GenAI client using the existing GOOGLE_AI_API_KEY."""
    try:
        from google import genai as _genai  # already in deps for the critic
    except ImportError as exc:
        raise ImportError(
            "google-genai not installed. Run: uv add google-genai"
        ) from exc
    return _genai.Client(api_key=_require_key("GOOGLE_AI_API_KEY"))


def google_imagen(prompt: str, aspect_ratio: str = "16:9") -> bytes:
    """Generate an image via Google Imagen 3 (AI Studio API).

    Requires GOOGLE_AI_API_KEY. Supported aspect ratios:
    '1:1', '3:4', '4:3', '9:16', '16:9'.
    Returns PNG bytes.
    """
    from google.genai import types as _gtypes
    resp = _genai_client().models.generate_images(
        model=GOOGLE_IMAGE_MODEL,
        prompt=prompt[:1000],
        config=_gtypes.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
        ),
    )
    if not resp.generated_images:
        raise RuntimeError("Google Imagen 3: no images returned")
    return resp.generated_images[0].image.image_bytes


def reve_image(
    prompt: str,
    reference_images: list[bytes] | None = None,
    aspect_ratio: str = "16:9",
    version: str = "latest",
) -> bytes:
    """Generate an image via the Reve API (api.reve.com).

    Routes to the best endpoint based on inputs:
      - reference_images provided → POST /v1/image/remix
        Accepts up to 6 base64 reference images. Excellent for character
        and moodboard consistency — the model attends to all refs.
      - no references → POST /v1/image/create
        Pure text-to-image with automatic prompt enhancement.

    Both endpoints are synchronous and return PNG bytes directly when
    Accept: image/png is set — no polling required.

    Supported aspect ratios: '16:9', '9:16', '3:2', '2:3', '4:3', '3:4', '1:1'.
    Requires REVE_API_KEY from api.reve.com/console.
    """
    import base64 as _b64
    api_key = _require_key("REVE_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "image/png",   # direct bytes, no base64 decode needed
        "Content-Type":  "application/json",
    }

    if reference_images:
        # Remix: text + up to 6 reference images → new image
        refs_b64 = [_b64.b64encode(r).decode() for r in reference_images[:6]]
        body: dict = {
            "prompt":           prompt[:2560],
            "reference_images": refs_b64,
            "aspect_ratio":     aspect_ratio,
            "version":          version,
        }
        endpoint = "/v1/image/remix"
    else:
        # Create: text → image
        body = {
            "prompt":       prompt[:2560],
            "aspect_ratio": aspect_ratio,
            "version":      version,
        }
        endpoint = "/v1/image/create"

    resp = httpx.post(
        f"{REVE_API_BASE}{endpoint}",
        headers=headers,
        json=body,
        timeout=120.0,
    )
    if not resp.is_success:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(f"Reve {endpoint} {resp.status_code}: {detail}")

    # With Accept: image/png the response body is raw PNG bytes
    return resp.content


def google_nano_banana(
    prompt: str,
    reference_images: list[bytes] | None = None,
) -> bytes:
    """Generate an image via Google Gemini 2.5 Flash Image (Nano Banana).

    Direct Gemini API call — independent of Runway's daily task limits.
    Uses the same underlying model that Runway proxies as gemini_image3_pro,
    but billed against GOOGLE_AI_API_KEY.

    Multi-reference support: pass up to ~14 reference image byte strings.
    Excellent at character consistency and multi-image fusion.

    Returns PNG bytes from the first inline_data part in the response.
    """
    from io import BytesIO
    from google.genai import types as _gtypes
    try:
        from PIL import Image as _PIL
    except ImportError as exc:
        raise RuntimeError("Pillow required for google_nano_banana") from exc

    client = _genai_client()

    # Build content list: prompt first, then any reference images as PIL Images
    contents: list = [prompt]
    if reference_images:
        for ref in reference_images[:14]:
            contents.append(_PIL.open(BytesIO(ref)))

    response = client.models.generate_content(
        model=GOOGLE_NANO_BANANA_MODEL,
        contents=contents,
    )

    # Walk parts looking for the first image
    for part in (response.parts or []):
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return part.inline_data.data
        # Some SDK versions surface as `as_image()` helper instead
        try:
            img = part.as_image()
            if img is not None:
                buf = BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            pass

    # Bubble up text response if no image returned (often a refusal)
    text_blocks = [p.text for p in (response.parts or []) if getattr(p, "text", None)]
    detail = " | ".join(text_blocks)[:300] if text_blocks else "no image in response"
    raise RuntimeError(f"google_nano_banana ({GOOGLE_NANO_BANANA_MODEL}): {detail}")


def google_veo(
    prompt: str,
    first_frame: bytes | None = None,
    duration_seconds: int = 8,
    resolution: str = "720p",
    reference_images: list[bytes] | None = None,
) -> bytes:
    """Generate a video via Google Veo 3.1 (AI Studio API).

    Valid durations: 4, 6, or 8 seconds.
    Constraints from Google docs:
      - Must be 8 s when resolution is 1080p or 4k
      - Must be 8 s when reference_images are provided

    Resolution strings: "720p" (default), "1080p", "4k".
    Returns MP4 bytes.
    """
    import tempfile
    from google.genai import types as _gtypes

    _VEO_VALID_DURATIONS = [4, 6, 8]
    _VEO_RESOLUTIONS = {"720p": "720p", "1080p": "1080p", "4k": "4k"}

    res = _VEO_RESOLUTIONS.get(resolution, "720p")

    # Snap to nearest valid duration
    dur = min(_VEO_VALID_DURATIONS, key=lambda d: abs(d - int(duration_seconds)))

    # Google requires 8 s when using 1080p, 4k, or reference images
    if res in ("1080p", "4k") or reference_images:
        dur = 8

    config_kwargs: dict = {
        "number_of_videos": 1,
        "duration_seconds": dur,
        "aspect_ratio":     "16:9",
        "resolution":       res,
        # NOTE: enhance_prompt is not supported by veo-3.1-generate-preview
        # (Google returns 400 'enhancePrompt isn't supported by this model').
        # We leave it unset so the model uses its default behavior.
    }
    if reference_images:
        config_kwargs["reference_images"] = [
            _gtypes.VideoGenerationReferenceImage(
                image=_gtypes.Image(image_bytes=r, mime_type="image/png"),
                reference_type="asset",
            )
            for r in reference_images[:3]
        ]

    config = _gtypes.GenerateVideosConfig(**config_kwargs)
    client = _genai_client()
    kwargs: dict = {"model": GOOGLE_VEO_MODEL, "prompt": prompt, "config": config}

    if first_frame:
        kwargs["image"] = _gtypes.Image(
            image_bytes=first_frame, mime_type="image/png"
        )

    operation = client.models.generate_videos(**kwargs)

    for _ in range(120):
        if operation.done:
            break
        time.sleep(5)
        operation = client.operations.get(operation)

    if not operation.done:
        raise RuntimeError(
            f"Google Veo ({GOOGLE_VEO_MODEL}) timed out after 10 minutes"
        )

    # When Veo's safety filter blocks a prompt, the operation's response
    # is None (not an empty .generated_videos list). Surface a proper
    # message so the cascade fallback log isn't filled with cryptic
    # AttributeError tracebacks.
    response = getattr(operation, "response", None)
    if response is None:
        raise RuntimeError(
            "Google Veo: response was None (likely safety-filter block "
            "or upstream rejection). Falling through to the next backend."
        )

    vids = getattr(response, "generated_videos", None)
    if not vids:
        raise RuntimeError("Google Veo: no videos returned")

    video = vids[0].video
    client.files.download(file=video)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    video.save(tmp_path)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data

def ltx_video(
    prompt: str,
    first_frame: bytes | None = None,
    duration_seconds: int = 8,
    resolution: str = "720p",
    seed: int | None = None,
    model: str | None = None,
    generate_audio: bool = True,
    camera_motion: str | None = None,
) -> bytes:
    """Generate video via LTX 2.3 (Lightricks API, api.ltx.video).

    Models:
        ltx-2-3-pro   quality, durations 6/8/10 s, all endpoints
        ltx-2-3-fast  speed,   durations 6-20 s in 2 s steps, cheaper

    generate_audio=True (default) produces native synchronized audio —
    dialogue, ambient, and SFX. The pipeline's veo_prompt suppresses
    music so Stability still owns the score.

    Requires LTX_API_KEY from console.ltx.video. Returns MP4 bytes.
    """
    api_key   = _require_key("LTX_API_KEY")
    auth      = {"Authorization": f"Bearer {api_key}"}
    use_model = model or LTX_VIDEO_MODEL

    # Snap to nearest valid duration for this model
    valid_durs = _LTX_DURATIONS.get(use_model, [6, 8, 10])
    duration   = min(valid_durs, key=lambda d: abs(d - duration_seconds))
    res_str    = _LTX_RESOLUTIONS.get(resolution, "1920x1080")  # default 1080p

    body: dict = {
        "prompt":         prompt[:5000],
        "model":          use_model,
        "duration":       duration,
        "resolution":     res_str,
        "generate_audio": generate_audio,
    }
    if seed is not None:
        body["seed"] = seed
    if camera_motion:
        body["camera_motion"] = camera_motion

    if first_frame:
        import base64 as _b64
        b64 = _b64.b64encode(first_frame).decode()
        body["image_uri"] = f"data:image/png;base64,{b64}"
        endpoint = "image-to-video"
    else:
        endpoint = "text-to-video"

    submit = httpx.post(
        f"{LTX_API_BASE}/v2/{endpoint}",
        headers={**auth, "Content-Type": "application/json"},
        json=body,
        timeout=30.0,
    )
    if not submit.is_success:
        try:
            detail = submit.json()
        except Exception:
            detail = submit.text[:400]
        raise RuntimeError(f"LTX submit {submit.status_code}: {detail}")

    job_id: str = submit.json()["id"]

    for _ in range(120):   # poll every 5 s, up to 10 min
        time.sleep(5)
        poll = httpx.get(
            f"{LTX_API_BASE}/v2/{endpoint}/{job_id}",
            headers=auth, timeout=30.0,
        )
        poll.raise_for_status()
        data = poll.json()
        if data["status"] == "completed":
            return httpx.get(data["result"]["video_url"], timeout=120.0).content
        if data["status"] == "failed":
            err = data.get("error", {})
            raise RuntimeError(f"LTX job failed: {err.get('message', err)}")

    raise RuntimeError(f"LTX job {job_id} timed out after 10 minutes")


# Runway video model IDs. Endpoint: /v1/image_to_video (or /v1/text_to_video,
# /v1/video_to_video for Aleph). Veo 3.1 is proxied through Runway so the
# numbers match what direct Google AI charged us pre-migration. The new
# entries (gen4.5, seedance2, gen4_aleph) are Runway-native and unlock
# stronger identity-lock and longer single-call durations.
VEO_MODEL_LITE     = "veo3.1_fast"          # Runway has no Lite tier; fast is the cheapest Veo
VEO_MODEL_FAST     = "veo3.1_fast"          # 10-15 credits/sec depending on audio
VEO_MODEL_STANDARD = "veo3.1"               # 20-40 credits/sec depending on audio
GEN45_MODEL        = "gen4.5"               # 12 credits/sec — Runway flagship, image-to-video w/ refs
GEN4_TURBO_MODEL   = "gen4_turbo"           # 5 credits/sec — image-to-video only, fast iteration
GEN4_ALEPH_MODEL   = "gen4_aleph"           # 15 credits/sec — video-to-video, transformation
SEEDANCE2_MODEL    = "seedance2"            # 36 credits/sec — premium, supports up to 15s

# Runway SFX model.
RUNWAY_SFX_MODEL   = "eleven_text_to_sound_v2"

# --- Video model registry ---
# Single source of truth for shot-list planning. Three Veo tiers preserve the
# old previs/fast/standard contract; gen4.5 / seedance2 are added as creative
# alternatives the agent can switch into. cost_per_sec is in USD and assumes
# audio is enabled (the Veo prompt block produces dialogue).
#
# Note on duration cap: Veo's native single-call max is 8s. Seedance2 lifts
# that to 15s, but the storyboarding logic still defaults to ≤8s shots so
# old produce.py doesn't have to know about the new ceiling. Switch a shot
# to seedance2 explicitly via `route_shot(.., tier="seedance2")` to use it.
VIDEO_MODELS = {
    "veo3.1_lite": {
        "id":               VEO_MODEL_FAST,    # alias to fast — Runway has no Lite
        "vendor":           "runway",
        "endpoint":         "image_to_video",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     True,
        "cost_per_sec":     0.15,
        "use_for":          "previs",
        "supports_refs":    False,
    },
    "veo3.1_fast": {
        "id":               VEO_MODEL_FAST,
        "vendor":           "runway",
        "endpoint":         "image_to_video",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     True,
        "cost_per_sec":     0.15,
        "use_for":          "iteration",
        "supports_refs":    False,
    },
    "veo3.1_standard": {
        "id":               VEO_MODEL_STANDARD,
        "vendor":           "runway",
        "endpoint":         "image_to_video",
        "max_seconds":      8,
        "duration_options": [4, 6, 8],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     True,
        "cost_per_sec":     0.40,
        "use_for":          "hero",
        "supports_refs":    False,
    },
    "gen4.5": {
        "id":               GEN45_MODEL,
        "vendor":           "runway",
        "endpoint":         "image_to_video",
        "max_seconds":      10,
        "duration_options": [5, 10],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     False,           # Runway adds audio via TTS/SFX layer
        "cost_per_sec":     0.12,
        "use_for":          "identity_lock",  # native referenceImages support
        "supports_refs":    True,
    },
    "seedance2": {
        "id":               SEEDANCE2_MODEL,
        "vendor":           "runway",
        "endpoint":         "image_to_video",
        "max_seconds":      15,
        "duration_options": [5, 10, 15],
        "fps":              24,
        "max_resolution":   "1080p",
        "native_audio":     False,
        "cost_per_sec":     0.36,
        "use_for":          "long_oner",      # the only model that can do >8s in one call
        "supports_refs":    True,
    },
}


# --- Render budget. Capped tightly so iteration stays cheap. ---
ASPECT_RATIO = "16:9"
SHOT_DURATION_SECONDS = int(os.getenv("SHOT_DURATION_SECONDS", "8"))
VEO_TIER = os.getenv("VEO_TIER", "fast").lower()  # previs | fast | standard | gen4.5 | seedance2
VEO_RESOLUTION = os.getenv("VEO_RESOLUTION", "720p")
TAKES_PER_SHOT = int(os.getenv("TAKES_PER_SHOT", "1"))  # 1 for cheap iteration


def veo_final_model() -> str:
    """Default model id for the current VEO_TIER. Used when produce.py
    asks for the 'main' video model rather than picking one per shot."""
    if VEO_TIER == "standard":
        return VEO_MODEL_STANDARD
    if VEO_TIER == "gen4.5":
        return GEN45_MODEL
    if VEO_TIER == "seedance2":
        return SEEDANCE2_MODEL
    return VEO_MODEL_FAST


# Aspect-ratio strings for Runway. Per docs.dev.runwayml.com/assets/inputs,
# each video model supports only a specific set of pixel ratios, and most
# *don't* support 1920:1080 — it's a Veo-only ratio. Image-to-video Gen-4.5
# tops out at 1280:720 in landscape; image generators have their own list.
# Map (model, aspect, resolution) → the closest supported pixel ratio.
def _runway_ratio(
    aspect: str = ASPECT_RATIO,
    resolution: str = VEO_RESOLUTION,
    model: str | None = None,
) -> str:
    """Return the Runway pixel-ratio string for a given aspect/resolution/model.

    Falls back to model-agnostic defaults if `model` is None or unknown.
    Veo accepts 1920:1080; most others don't, so we clamp 1080p requests
    on non-Veo models to the closest 720p-class ratio they accept.
    """
    veo_ish = model in (VEO_MODEL_FAST, VEO_MODEL_STANDARD, "veo3", "veo3.1", "veo3.1_fast")
    if aspect == "16:9":
        if resolution == "1080p" and veo_ish:
            return "1920:1080"
        return "1280:720"
    if aspect == "9:16":
        if resolution == "1080p" and veo_ish:
            return "1080:1920"
        return "720:1280"
    if aspect == "1:1":
        return "960:960" if model in ("gen4.5", GEN45_MODEL, "gen4_turbo", GEN4_TURBO_MODEL) else "1024:1024"
    return "1280:720"


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
def runway_client() -> RunwayML:
    """Single Runway client. Drives all image, video, and SFX generation.

    Reads `RUNWAYML_API_SECRET` from the environment (the SDK's default).
    The skill helpers in runway-skills/ use the same env var name.
    """
    # The SDK reads RUNWAYML_API_SECRET automatically; we still resolve it
    # here so we get our friendly missing-key error message.
    _require_key("RUNWAYML_API_SECRET")
    return RunwayML()


@lru_cache(maxsize=1)
def gemini_client():
    """Critic-only Gemini 3 Pro client (long-video review in evaluate_film).

    All image/video generation lives in Runway now; this client only exists
    because Runway has no equivalent video-review LLM endpoint. If you don't
    care about Reviewer A and only want Claude's stills review, you can
    leave GOOGLE_AI_API_KEY blank and `evaluate_film` will degrade to a
    one-reviewer score.
    """
    from google import genai  # type: ignore
    return genai.Client(api_key=_require_key("GOOGLE_AI_API_KEY"))


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
def _book_slug(book_path: str | Path | None = None) -> str:
    """Derive a filesystem-safe slug from a book PDF path. Used to group
    experiments by source book under experiments/{slug}/exp_NNN/.

    Examples:
        "/uploads/JurassicPark-MichaelCrichton.pdf" -> "jurassic_park"
        "/uploads/Last Exit to Brooklyn.pdf"        -> "last_exit_to_brooklyn"
        "/uploads/the_great_gatsby.pdf"             -> "the_great_gatsby"
        ""                                          -> "unknown_book"

    Author suffixes after the first dash are dropped (a common
    "Title-Author" filename convention). CamelCase is split into snake.
    Anything non-alphanumeric collapses to a single underscore.
    """
    import re
    if book_path is None:
        book_path = os.environ.get("BOOK_PDF_PATH", "")
    p = Path(book_path)
    if not p.name:
        return "unknown_book"
    stem = p.stem
    # Drop author suffix when filename uses "Title-Author" convention.
    if "-" in stem:
        stem = stem.split("-", 1)[0]
    # CamelCase / PascalCase → snake: insert _ before any non-leading caps.
    stem = re.sub(r"(?<!^)(?=[A-Z])", "_", stem)
    # Collapse anything non-alphanumeric to underscores; lowercase; trim.
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug or "unknown_book"


def iter_all_experiments() -> list[Path]:
    """Return every experiment directory under EXPERIMENTS_DIR, sorted
    by mtime (oldest first).

    Handles both layouts simultaneously:
      - new:  experiments/{book_slug}/exp_NNN/
      - old:  experiments/exp_NNN/         (pre-per-book refactor)

    Skips internal dirs that start with an underscore (e.g.
    ``experiments/_smoke_tests/``).
    """
    out: list[Path] = []
    if not EXPERIMENTS_DIR.exists():
        return out
    for entry in EXPERIMENTS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if entry.name.startswith("exp_"):
            # Old flat layout — direct experiment dir.
            out.append(entry)
        else:
            # New per-book layout — entry is a book slug; walk its experiments.
            for sub in entry.iterdir():
                if sub.is_dir() and sub.name.startswith("exp_"):
                    out.append(sub)
    return sorted(out, key=lambda p: p.stat().st_mtime)


@dataclass
class Experiment:
    """One experiment = one full pipeline run with a particular produce.py.

    Filesystem layout (new):
        experiments/
          jurassic_park/
            exp_001/
              produce.py        ← snapshot
              book.txt          ← book slug ("jurassic_park")
              script.json
              cast.json
              ...
              final.mp4
              bible.pdf
            exp_002/
            ...
          last_exit_to_brooklyn/
            exp_001/
            ...
          _smoke_tests/         ← Runway smoke test outputs (not real experiments)
            20260508_214500/
            ...
    """
    exp_id: str
    root: Path
    # Per-experiment lock protecting prompts.json from concurrent writes.
    # Excluded from __init__, __repr__, and __eq__ so it's transparent.
    _log_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @classmethod
    def new(cls, book_slug: str | None = None) -> "Experiment":
        """Create a new experiment dir under experiments/{book_slug}/.

        If ``book_slug`` is None, derives one from the BOOK_PDF_PATH
        env var. Numbering is per-book — the first Jurassic Park run
        and the first Last Exit run are both ``exp_001`` under their
        respective book subdirs.
        """
        if book_slug is None:
            book_slug = _book_slug()
        book_dir = EXPERIMENTS_DIR / book_slug
        book_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p for p in book_dir.iterdir()
            if p.is_dir() and p.name.startswith("exp_")
        )
        n = len(existing) + 1
        exp_id = f"exp_{n:03d}"
        root = book_dir / exp_id
        root.mkdir()
        # Snapshot produce.py for reproducibility.
        produce_src = PROJECT_ROOT / "produce.py"
        if produce_src.exists():
            (root / "produce.py").write_text(produce_src.read_text())
        # Stamp the book slug so downstream tools (bible, evaluate) can
        # surface it without re-deriving from BOOK_PDF_PATH (which may
        # have changed by the time we look).
        (root / "book.txt").write_text(book_slug)
        # Assign a random seed. Stored so every resume of this experiment
        # uses the same seed, making generation deterministic across runs.
        import random as _random
        seed = int(os.environ.get("SEED", "")) if os.environ.get("SEED", "").isdigit() \
               else _random.randint(100_000, 999_999)
        (root / "seed.txt").write_text(str(seed))
        # If the UI server staged a "pending" run config (set via the
        # AUTOFILM_PENDING_RUN_CONFIG env var), copy it into this exp dir
        # so the UI dropdown can later load these settings as defaults
        # when re-running. The pending file is per-book, so this works
        # for both fresh runs and resumes; for new_iteration() the parent
        # exp will already have its own run_config.json.
        pending_str = os.environ.get("AUTOFILM_PENDING_RUN_CONFIG", "")
        if pending_str:
            pending = Path(pending_str)
            if pending.is_file():
                try:
                    (root / "run_config.json").write_text(pending.read_text())
                except Exception:
                    pass   # non-fatal
        return cls(exp_id=exp_id, root=root)

    @classmethod
    def new_or_resume(cls, book_slug: str | None = None) -> "Experiment":
        """Return the experiment to run — resuming the latest incomplete one
        or creating a fresh one when appropriate.

        Resume logic:
          - If the latest experiment for this book has no ``final.mp4``, it
            is considered incomplete and is resumed. All already-generated
            artifacts (frames, clips, audio, references, moodboards) are
            kept intact; only missing files are generated.
          - If the latest experiment is complete (has ``final.mp4``), a new
            experiment is created so the completed run isn't overwritten.
          - ``FORCE_NEW=1`` always creates a new experiment, ignoring any
            incomplete prior run. Useful when you've changed ``produce.py``
            significantly and want a clean slate.
          - ``SEED=<int>`` pins the seed for the new experiment. On resume
            the stored seed is always used, regardless of this variable.

        The experiment's seed is stored in ``seed.txt`` at creation and read
        back on every resume so video takes are reproducible across runs.
        """
        force_new = os.environ.get("FORCE_NEW", "").lower() in ("1", "true", "yes")
        book_pdf_set = bool(os.environ.get("BOOK_PDF_PATH", "").strip())

        # ── Resolve which book this run targets ─────────────────────────
        # When neither a slug nor BOOK_PDF_PATH is given (e.g. user re-runs
        # `python run_loop.py ...` after a crash), resume the most recently
        # modified incomplete experiment across ALL books. Without this,
        # the slug derives to "unknown_book" and a fresh empty experiment
        # gets created instead — wasting credits and losing the partial
        # progress.
        #
        # We only do this when BOOK_PDF_PATH is unset, so a user pointing
        # at a NEW book (BOOK_PDF_PATH=other.pdf) isn't hijacked into
        # resuming an unrelated old experiment.
        if book_slug is None and not force_new and not book_pdf_set:
            incomplete = [
                p for p in iter_all_experiments()
                if not (p / "final.mp4").exists()
            ]
            if incomplete:
                latest = max(incomplete, key=lambda p: p.stat().st_mtime)
                book_slug = latest.parent.name
                print(f"  Auto-resuming incomplete experiment "
                      f"{book_slug}/{latest.name} "
                      f"(set BOOK_PDF_PATH to start a fresh book run).")

        if book_slug is None:
            book_slug = _book_slug()

        if not force_new:
            book_dir = EXPERIMENTS_DIR / book_slug
            if book_dir.exists():
                candidates = sorted(
                    p for p in book_dir.iterdir()
                    if p.is_dir() and p.name.startswith("exp_")
                )
                if candidates:
                    latest = candidates[-1]
                    if not (latest / "final.mp4").exists():
                        exp = cls(exp_id=latest.name, root=latest)
                        # Ensure seed.txt exists (older experiments may lack it).
                        if not (latest / "seed.txt").exists():
                            import random as _random
                            (latest / "seed.txt").write_text(
                                str(_random.randint(100_000, 999_999))
                            )
                        return exp

        return cls.new(book_slug)

    @classmethod
    def load(cls, exp_id: str) -> "Experiment":
        """Locate an experiment by id.

        Accepts either:
          - ``"exp_001"`` — searches all book subdirs (and the old flat
            layout) for the most recent match
          - ``"jurassic_park/exp_001"`` — direct path under EXPERIMENTS_DIR

        Raises FileNotFoundError if no matching dir exists.
        """
        if "/" in exp_id:
            slug, eid = exp_id.split("/", 1)
            root = EXPERIMENTS_DIR / slug / eid
            if not root.is_dir():
                raise FileNotFoundError(f"No experiment at {root}")
            return cls(exp_id=eid, root=root)
        # Bare exp_id — scan all book subdirs (newest match wins).
        candidates: list[Path] = []
        for book_dir in EXPERIMENTS_DIR.iterdir() if EXPERIMENTS_DIR.exists() else []:
            if not book_dir.is_dir() or book_dir.name.startswith("_"):
                continue
            cand = book_dir / exp_id
            if cand.is_dir():
                candidates.append(cand)
        # Old flat layout fallback.
        flat = EXPERIMENTS_DIR / exp_id
        if flat.is_dir():
            candidates.append(flat)
        if not candidates:
            raise FileNotFoundError(
                f"No experiment named '{exp_id}' under {EXPERIMENTS_DIR}. "
                f"Pass a fully-qualified id like 'jurassic_park/{exp_id}' or "
                f"check that the experiment hasn't been moved."
            )
        # If there's more than one (same exp_id in multiple books), pick
        # the most recent by mtime.
        winner = max(candidates, key=lambda p: p.stat().st_mtime)
        return cls(exp_id=exp_id, root=winner)

    @classmethod
    def latest(cls) -> "Experiment":
        """Return the most recently modified experiment across all books.

        Used by ``evaluate.py latest`` and the agent loop to find the
        experiment that just finished.
        """
        all_exps = iter_all_experiments()
        if not all_exps:
            raise FileNotFoundError(f"No experiments found under {EXPERIMENTS_DIR}")
        latest = max(all_exps, key=lambda p: p.stat().st_mtime)
        return cls(exp_id=latest.name, root=latest)

    @classmethod
    def new_iteration(
        cls,
        prev_exp: "Experiment",
        carryover: "dict[str, Any] | None" = None,
    ) -> "Experiment":
        """Create a new experiment that selectively inherits from prev_exp.

        Used by run_loop.py to chain experiments: each iteration starts
        fresh BUT copies forward all artifacts the critic didn't flag for
        regeneration. produce.py's stage-level ``if exp.has(...)`` checks
        skip those carried-forward files, so only the invalidated artifacts
        are regenerated.

        ``carryover`` schema (all fields optional, default = inherit
        everything that exists):
          {
            "regen_script":      bool,   # parse + screenplay format
            "regen_cast":        bool,   # casting + locations + moodboards
            "regen_lookbook":    bool,   # style frame + grade + style keywords
            "regen_references":  list[[scene_id, char_id]] | "all",
            "regen_storyboard":  bool,
            "regen_music":       list[scene_id] | "all",
            "regen_narration":   list[scene_id] | "all",
            "regen_frames":      list[[scene_id, shot_id]] | "all",
            "regen_clips":       list[[scene_id, shot_id]] | "all",
            "regen_edl":         bool,
          }

        Cascade rules: ``regen_lookbook`` forces references / frames /
        clips to "all" (the look fundamentally changed). ``regen_script``
        forces full pipeline regen.
        """
        import shutil
        carryover = dict(carryover or {})

        # Cascade: high-level invalidations force lower-level ones.
        if carryover.get("regen_script"):
            for k in ("regen_cast", "regen_lookbook", "regen_storyboard"):
                carryover[k] = True
            carryover["regen_references"] = "all"
            carryover["regen_music"]      = "all"
            carryover["regen_narration"]  = "all"
            carryover["regen_frames"]     = "all"
            carryover["regen_clips"]      = "all"
            carryover["regen_edl"]        = True
        if carryover.get("regen_lookbook"):
            carryover["regen_references"] = "all"
            carryover["regen_frames"]     = "all"
            carryover["regen_clips"]      = "all"

        # Create new exp dir under the same book.
        book_slug = prev_exp.book_slug
        book_dir = EXPERIMENTS_DIR / book_slug
        existing = sorted(
            p for p in book_dir.iterdir()
            if p.is_dir() and p.name.startswith("exp_")
        )
        n = len(existing) + 1
        exp_id = f"exp_{n:03d}"
        root = book_dir / exp_id
        root.mkdir()

        produce_src = PROJECT_ROOT / "produce.py"
        if produce_src.exists():
            (root / "produce.py").write_text(produce_src.read_text())
        (root / "book.txt").write_text(book_slug)
        # Inherit seed for reproducibility — same characters, same camera
        # luck, only invalidated stages get fresh API calls.
        (root / "seed.txt").write_text(str(prev_exp.seed))
        (root / "parent_exp.txt").write_text(prev_exp.exp_id)
        (root / "carryover.json").write_text(
            json.dumps(carryover, indent=2, ensure_ascii=False)
        )
        # Inherit the parent's run_config.json so the UI dropdown shows
        # consistent defaults across the whole iteration chain.
        prev_rc = prev_exp.root / "run_config.json"
        if prev_rc.exists():
            try:
                (root / "run_config.json").write_text(prev_rc.read_text())
            except Exception:
                pass

        # Top-level files we never copy forward (rebuilt by next exp).
        skip_files: set[str] = {
            "produce.py", "book.txt", "seed.txt", "parent_exp.txt",
            "carryover.json", "run_config.json",
            "final.mp4", "final_pregrade.mp4",
            "metric.json", "critique.md", "bible.pdf",
            "production_bible.json", "prompts.json",
        }

        def _copy_one(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Copy individual top-level JSON files unless flagged for regen.
        flag_for_file = {
            "script.json":           "regen_script",
            "cast.json":             "regen_cast",
            "locations.json":        "regen_cast",
            "lookbook.json":         "regen_lookbook",
            "storyboard.json":       "regen_storyboard",
            "edl.json":              "regen_edl",
            "frames_manifest.json":  "regen_frames",  # any per-shot regen invalidates the manifest
        }
        for src in prev_exp.root.iterdir():
            if not src.is_file() or src.name in skip_files:
                continue
            flag = flag_for_file.get(src.name)
            if flag and carryover.get(flag):
                continue
            _copy_one(src, root / src.name)

        def _copy_filtered_dir(
            dir_name: str,
            regen_spec: Any,
            key_pattern,
        ) -> None:
            """Copy dir_name from prev_exp to new exp, skipping any file
            whose key (via key_pattern) is in regen_spec. If regen_spec is
            'all', skip the whole dir (nothing carried forward)."""
            src_dir = prev_exp.root / dir_name
            if not src_dir.exists():
                return
            if regen_spec == "all":
                return
            regen_set: set[tuple] = (
                {tuple(x) for x in regen_spec}
                if isinstance(regen_spec, list)
                else set()
            )
            for src in src_dir.rglob("*"):
                if not src.is_file():
                    continue
                # Path relative to *the dir* for key lookup (e.g. for
                # frames/scene_001/shot_001.png, key_rel.parts is
                # ('scene_001', 'shot_001.png')). Path relative to *the
                # exp root* for the actual copy destination.
                key_rel  = src.relative_to(src_dir)
                copy_rel = src.relative_to(prev_exp.root)
                key = key_pattern(key_rel)
                if key and key in regen_set:
                    continue
                _copy_one(src, root / copy_rel)

        # Lookbook subdir (style_frame.png) — all-or-nothing.
        if not carryover.get("regen_lookbook"):
            lb_src = prev_exp.root / "lookbook"
            if lb_src.exists():
                shutil.copytree(lb_src, root / "lookbook", dirs_exist_ok=True)

        # Location moodboards — bound to cast/locations stage.
        if not carryover.get("regen_cast"):
            mb_src = prev_exp.root / "location_moodboards"
            if mb_src.exists():
                shutil.copytree(mb_src, root / "location_moodboards", dirs_exist_ok=True)

        # References: layout = references/{char_id}/{scene_id}.png
        # key_rel = char_id/{scene_id}.png; carryover key = (scene_id, char_id)
        _copy_filtered_dir(
            "references",
            carryover.get("regen_references"),
            lambda rel: (rel.parts[1].split(".")[0], rel.parts[0])
                        if len(rel.parts) >= 2 else None,
        )

        # Music: layout = music/{scene_id}.wav
        # key_rel = {scene_id}.wav
        _copy_filtered_dir(
            "music",
            carryover.get("regen_music"),
            lambda rel: (rel.stem,)
                        if rel.suffix in {".wav", ".mp3"} else None,
        )

        # Narration: layout = narration/{scene_id}.mp3
        _copy_filtered_dir(
            "narration",
            carryover.get("regen_narration"),
            lambda rel: (rel.stem,)
                        if rel.suffix in {".mp3", ".wav"} else None,
        )

        # Frames: layout = frames/{scene_id}/{shot_id}.png
        # key_rel = scene_id/{shot_id}.png; carryover key = (scene_id, shot_id)
        _copy_filtered_dir(
            "frames",
            carryover.get("regen_frames"),
            lambda rel: (rel.parts[0], rel.parts[1].split(".")[0])
                        if len(rel.parts) >= 2 else None,
        )

        # Clips: layout = clips/{scene_id}/{shot_id}/take_N.mp4
        # key_rel = scene_id/shot_id/take_N.mp4; key = (scene_id, shot_id)
        _copy_filtered_dir(
            "clips",
            carryover.get("regen_clips"),
            lambda rel: (rel.parts[0], rel.parts[1])
                        if len(rel.parts) >= 2 else None,
        )

        # SFX (ambient): bound to clip-level regen. If clips is 'all', skip.
        if carryover.get("regen_clips") != "all":
            sfx_src = prev_exp.root / "sfx"
            if sfx_src.exists():
                shutil.copytree(sfx_src, root / "sfx", dirs_exist_ok=True)

        return cls(exp_id=exp_id, root=root)

    @property
    def parent_exp_id(self) -> "str | None":
        """The parent experiment id if this exp inherited from one, else None.

        Set by ``new_iteration()`` when chaining experiments.
        """
        f = self.root / "parent_exp.txt"
        return f.read_text().strip() if f.exists() else None

    @property
    def book_slug(self) -> str:
        """Read the book slug stamped at experiment creation. Returns
        'unknown_book' for old flat-layout experiments that pre-date this
        field."""
        f = self.root / "book.txt"
        return f.read_text().strip() if f.exists() else "unknown_book"

    @property
    def seed(self) -> int:
        """The integer seed for this experiment's randomized API calls.

        Stored in ``seed.txt`` at creation and constant for the lifetime
        of the experiment. All video-generation seeds derive from this
        (``exp.seed + take_idx * 137``) so re-running with the same
        experiment produces the same takes for slots that weren't cached.
        Returns 1000 as a safe fallback for very old experiments that
        pre-date seed tracking.
        """
        f = self.root / "seed.txt"
        try:
            return int(f.read_text().strip())
        except Exception:
            return 1000

    @property
    def is_complete(self) -> bool:
        """True if final.mp4 exists — used by new_or_resume() to decide
        whether to resume or start fresh."""
        return (self.root / "final.mp4").exists()

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
        Thread-safe: protected by a per-experiment lock so parallel stages
        don't corrupt the JSON file.
        """
        with self._log_lock:
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

# Runway data URI size caps, from docs.dev.runwayml.com/assets/inputs.
# These are the *encoded* sizes; base64 inflates by ~33% so the binary
# limit is ~75% of the encoded limit.
_RUNWAY_DATA_URI_BYTES = {
    "image": int(5 * 1024 * 1024 * 0.74),   # ~3.7 MB raw → 5 MB encoded
    "video": int(16 * 1024 * 1024 * 0.74),  # ~11.8 MB raw → 16 MB encoded
    "audio": int(32 * 1024 * 1024 * 0.74),  # ~23.7 MB raw → 32 MB encoded
}


def _kind_from_mime(mime: str) -> str:
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    return "image"


def _ephemeral_upload(blob: bytes, mime: str) -> str:
    """Upload bytes to Runway's ephemeral object storage and return the
    `runway://` URI. Used when bytes exceed the data URI cap (5 MB image,
    16 MB video, 32 MB audio). The URI is valid for 24 hours.

    See https://docs.dev.runwayml.com/assets/uploads/ for the protocol —
    POST /v1/uploads → presigned uploadUrl + fields + runwayUri, then
    multipart-POST the file to uploadUrl.
    """
    api_key = _require_key("RUNWAYML_API_SECRET")
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
           "image/webp": "webp", "video/mp4": "mp4", "video/quicktime": "mov",
           "audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}.get(mime, "bin")
    filename = f"autofilm-{int(time.time() * 1000)}.{ext}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": "2024-11-06",
        "Content-Type": "application/json",
    }
    init = httpx.post(
        "https://api.dev.runwayml.com/v1/uploads",
        headers=headers,
        json={"filename": filename, "type": "ephemeral"},
        timeout=30.0,
    )
    init.raise_for_status()
    data = init.json()
    upload_url = data["uploadUrl"]
    fields = data.get("fields", {})
    runway_uri = data["runwayUri"]
    # Multipart POST to the presigned URL. Don't reuse the auth headers.
    put = httpx.post(
        upload_url,
        data=fields,
        files={"file": (filename, blob, mime)},
        timeout=300.0,
    )
    put.raise_for_status()
    return runway_uri


def _runway_uri(blob: bytes, mime: str = "image/png") -> str:
    """Encode bytes for a Runway request body.

    Uses a base64 data URI when the payload fits within Runway's documented
    size cap for that media type; falls back to an ephemeral upload (which
    returns a `runway://` URI valid for 24h) when it doesn't.

    This is the helper to use anywhere produce.py used to hand raw bytes
    into a `prompt_image`, `reference_images[].uri`, or `video_uri` field.
    """
    kind = _kind_from_mime(mime)
    if len(blob) <= _RUNWAY_DATA_URI_BYTES[kind]:
        return f"data:{mime};base64,{base64.b64encode(blob).decode()}"
    return _ephemeral_upload(blob, mime)


# Back-compat shim: `_data_uri` was the original name in this file before
# the upload-fallback was added. Some helpers below still reference it.
def _data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    return _runway_uri(image_bytes, mime)


@api_retry
def runway_image(
    prompt: str,
    reference_images: list[bytes] | None = None,
    reference_tags: list[str] | None = None,
    model: str = GEN4_IMAGE_MODEL,
    ratio: str | None = None,
) -> bytes:
    """Generate an image via the Runway /v1/text_to_image endpoint.

    Generic helper that backs `gpt_image()` and `nano_banana()` below; can
    also be called directly to use `gen4_image` / `gen4_image_turbo`, which
    have first-class reference-image support and are Runway's strongest
    identity-lock models.

    Args:
        prompt: text description of the desired image.
        reference_images: PNG/JPEG bytes for up to 3 reference images.
        reference_tags: optional tags (3-16 lowercase chars) for each
            reference; you can address them in the prompt as `@tag`. If
            unset, tags `ref1`, `ref2`, `ref3` are auto-assigned.
        model: Runway image model id. See the constants above.
        ratio: pixel ratio string. Defaults to a sensible per-model value.
    """
    refs_payload: list[dict] = []
    for i, img in enumerate((reference_images or [])[:3]):
        tag = (reference_tags[i] if reference_tags and i < len(reference_tags) else f"ref{i+1}")
        refs_payload.append({"tag": tag, "uri": _data_uri(img)})

    if model == GEN4_IMAGE_TURBO and not refs_payload:
        raise RuntimeError(
            "gen4_image_turbo requires at least one reference image. "
            "Pass reference_images=[..] or use gen4_image / gemini_image3_pro."
        )

    if ratio is None:
        ratio = _ratio_for_image_model(model)

    kwargs: dict[str, Any] = {
        "model": model,
        "prompt_text": prompt,
        "ratio": ratio,
    }
    if refs_payload:
        kwargs["reference_images"] = refs_payload

    task = runway_client().text_to_image.create(**kwargs).wait_for_task_output()
    if not task.output:
        raise RuntimeError(f"Runway {model}: no output URL returned")
    return httpx.get(task.output[0], timeout=120.0).content


# Per-model valid pixel ratios for image generation. Each Runway image
# model exposes a different set; submitting an unsupported value fails
# validation server-side. Sourced from the BadRequestError responses
# the API returns for invalid ratios. Update if you add a new model.
#
#   gpt_image_2:        1K/2K/4K tiers — no 720p/1080p shapes
#   gemini_image3_pro:  has its own family (1344:768 for 16:9), accepts
#                       1024:1024 / 2048:2048 / 4096:4096 for square
#   gemini_2.5_flash:   same family as gemini_image3_pro
#   gen4_image (turbo): accepts 720p/1080p ratios; uses _runway_ratio()
#
# Per (aspect, model) → pixel ratio. We default to the "1K-class" tier
# for cost reasons; gpt_image() overrides per quality= argument.
_IMAGE_MODEL_RATIOS: dict[str, dict[str, str]] = {
    # gpt_image_2 — using 2K wide as the safe default (exact 16:9 ratio)
    "gpt_image_2": {
        "16:9": "2560:1440",
        "1:1":  "2560:2560",
        "9:16": "1440:2560",
    },
    # gemini_image3_pro (Nano Banana) — 1K-class
    "gemini_image3_pro": {
        "16:9": "1344:768",
        "1:1":  "1024:1024",
        "9:16": "768:1344",
    },
    # gemini_2.5_flash (Nano Banana, fastest) — same family
    "gemini_2.5_flash": {
        "16:9": "1344:768",
        "1:1":  "1024:1024",
        "9:16": "768:1344",
    },
}


def _ratio_for_image_model(model: str) -> str:
    """Pick a valid pixel ratio for an image-generation model.

    Looks up the model in the per-model ratio table; falls back to the
    generic _runway_ratio() (which returns 1280:720 etc.) for models
    that aren't constrained to special shapes (gen4_image*).
    """
    if model in _IMAGE_MODEL_RATIOS:
        return _IMAGE_MODEL_RATIOS[model].get(ASPECT_RATIO, _runway_ratio())
    return _runway_ratio()


def _aspect_from_size(size: str) -> str:
    """Parse old OpenAI-style size strings ('1792x1024', '1024x1024') into
    one of '16:9' / '1:1' / '9:16'. Used by gpt_image() for backward
    compatibility with produce.py callers."""
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except Exception:
        return "16:9"
    if w > h * 1.3:
        return "16:9"
    if h > w * 1.3:
        return "9:16"
    return "1:1"


def gpt_image(prompt: str, size: str = "1792x1024", quality: str = "high") -> bytes:
    """GPT Image 2 — best instruction-following image gen, via Runway.

    Backwards-compatible signature for produce.py. The OpenAI ``size``
    string is mapped to one of '16:9' / '1:1' / '9:16'; the ``quality``
    argument selects the gpt_image_2 tier (1K → ~$0.01, 2K → ~$0.10,
    4K → ~$0.40 per image).
    """
    aspect = _aspect_from_size(size)
    # gpt_image_2's tier-specific ratios. These are the "exact" 16:9 / 1:1 /
    # 9:16 options at each tier — see the API's BadRequestError response
    # for the full list.
    tier_ratios: dict[str, dict[str, str]] = {
        "low": {       # 1K tier
            "16:9": "1920:1088",
            "1:1":  "1920:1920",
            "9:16": "1088:1920",
        },
        "standard": {  # 2K tier
            "16:9": "2560:1440",
            "1:1":  "2560:2560",
            "9:16": "1440:2560",
        },
        "high": {      # 4K tier
            "16:9": "3840:2160",
            "1:1":  "2880:2880",
            "9:16": "2160:3840",
        },
        "auto": {      # let Runway pick
            "16:9": "auto",
            "1:1":  "auto",
            "9:16": "auto",
        },
    }
    ratio = tier_ratios.get(quality, tier_ratios["high"]).get(aspect, "auto")
    return runway_image(prompt, model=GPT_IMAGE_MODEL, ratio=ratio)


def openai_image(
    prompt: str,
    size: str = "1536x1024",
    quality: str = "medium",
) -> bytes:
    """GPT Image via OpenAI API directly (not through Runway).

    Model: gpt-image-1 (the available model on OpenAI's API without
    special org verification). Uses a separate OPENAI_API_KEY and
    billing account — independent of Runway's daily task limits.

    Sizes:  '1024x1024' (square), '1536x1024' (landscape), '1024x1536' (portrait)
    Quality: 'low' | 'medium' | 'high' | 'auto'

    Returns PNG bytes (decoded from the base64 JSON response).
    """
    import base64 as _b64
    api_key = _require_key("OPENAI_API_KEY")
    resp = httpx.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":   "gpt-image-1",
            "prompt":  prompt[:32000],  # model supports long prompts
            "n":       1,
            "size":    size,
            "quality": quality,
        },
        timeout=180.0,
    )
    if not resp.is_success:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(f"OpenAI image {resp.status_code}: {detail}")
    return _b64.b64decode(resp.json()["data"][0]["b64_json"])


def nano_banana(prompt: str, reference_images: list[bytes] | None = None) -> bytes:
    """Nano Banana — Gemini Image 3 Pro via Runway.

    Identity locking and multi-image fusion. Up to 3 reference images,
    matching Runway's referenceImages cap (the old direct API allowed up
    to 14 elements; this is a small regression but the gen4_image refs
    system more than compensates for character continuity).
    """
    return runway_image(prompt, reference_images=reference_images, model=NANO_BANANA_MODEL)


def gen4_image(
    prompt: str,
    reference_images: list[bytes],
    turbo: bool = False,
    ratio: str | None = None,
) -> bytes:
    """Runway Gen4 — purpose-built identity-lock image model.

    Both gen4_image and gen4_image_turbo are Runway's strongest reference-
    image models, designed specifically for preserving character / object
    identity across shots. They differ in cost vs flexibility:
      - gen4_image       (turbo=False): 5–8 credits, more flexible quality
      - gen4_image_turbo (turbo=True):  2 credits, faster, refs REQUIRED

    Both REQUIRE at least one reference image. For text-to-image without
    refs, use gpt_image / nano_banana / flux_image instead.

    Up to 3 reference images, tagged ref1/ref2/ref3 (referenceable as @ref1
    etc. in the prompt).
    """
    if not reference_images:
        raise RuntimeError(
            "gen4_image requires at least one reference image. "
            "Use gpt_image / nano_banana / flux_image for text-only."
        )
    model = GEN4_IMAGE_TURBO if turbo else GEN4_IMAGE_MODEL
    return runway_image(
        prompt,
        reference_images=reference_images,
        model=model,
        ratio=ratio,
    )


@api_retry
def seedance(
    prompt: str,
    first_frame: bytes,
    reference_images: list[bytes] | None = None,
    duration_seconds: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
) -> bytes:
    """Image-to-video via SeedDance 2 (ByteDance model on Runway infrastructure).

    Distinct from veo() — SeedDance is a ByteDance diffusion model, not
    Google Veo. It natively accepts reference images for identity-consistent
    generation across shots. 36 credits/sec, up to 15 seconds per clip.

    Reference images are passed with tags ref1/ref2/ref3 so the model can
    attend to character appearance, location moodboard, etc.
    """
    duration = duration_seconds or SHOT_DURATION_SECONDS
    ratio    = _runway_ratio(resolution=resolution or VEO_RESOLUTION,
                             model=SEEDANCE2_MODEL)
    model_meta = next(
        (m for m in VIDEO_MODELS.values() if m["id"] == SEEDANCE2_MODEL), None
    )
    if model_meta:
        duration = _snap_to_options(duration, model_meta["duration_options"])

    kwargs: dict[str, Any] = {
        "model":        SEEDANCE2_MODEL,
        "prompt_text":  prompt,
        "prompt_image": _data_uri(first_frame),
        "ratio":        ratio,
        "duration":     duration,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if reference_images:
        kwargs["reference_images"] = [
            {"tag": f"ref{i+1}", "uri": _data_uri(r)}
            for i, r in enumerate(reference_images[:3])
        ]

    task = runway_client().image_to_video.create(**kwargs).wait_for_task_output()
    if not task.output:
        raise RuntimeError("SeedDance: no output URL returned")
    return httpx.get(task.output[0], timeout=300.0).content


def veo(
    prompt: str,
    first_frame: bytes,
    reference_images: list[bytes] | None = None,
    model: str | None = None,
    duration_seconds: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
) -> bytes:
    """Image-to-video generation via Runway.

    Default model is veo3.1_fast (matches the old direct-Google path
    1:1 in cost and capability). Pass `model=GEN45_MODEL` or
    `model=SEEDANCE2_MODEL` to use Runway-native alternatives that
    accept reference images directly (Veo on Runway does not).
    """
    target_model = model or veo_final_model()
    duration = duration_seconds or SHOT_DURATION_SECONDS
    ratio = _runway_ratio(resolution=resolution or VEO_RESOLUTION, model=target_model)

    # Look up duration constraints for this model.
    model_meta = next((m for m in VIDEO_MODELS.values() if m["id"] == target_model), None)
    if model_meta:
        duration = _snap_to_options(duration, model_meta["duration_options"])

    kwargs: dict[str, Any] = {
        "model": target_model,
        "prompt_text": prompt,
        "prompt_image": _data_uri(first_frame),
        "ratio": ratio,
        "duration": duration,
    }
    if seed is not None:
        kwargs["seed"] = seed

    # gen4.5 / seedance2 accept native reference_images; Veo on Runway
    # does not, so for Veo we silently drop refs and rely on the first
    # frame for identity (which is what the old pipeline did anyway).
    if reference_images and model_meta and model_meta.get("supports_refs"):
        kwargs["reference_images"] = [
            {"tag": f"ref{i+1}", "uri": _data_uri(r)}
            for i, r in enumerate(reference_images[:3])
        ]

    task = runway_client().image_to_video.create(**kwargs).wait_for_task_output()
    if not task.output:
        raise RuntimeError(f"Runway {target_model}: no output URL returned")
    return httpx.get(task.output[0], timeout=300.0).content


@api_retry
def aleph_video_to_video(
    prompt: str,
    input_video: bytes,
    reference_image: bytes | None = None,
) -> bytes:
    """Gen-4 Aleph — video-to-video transformation via Runway.

    NEW capability the old pipeline didn't have. Useful as a per-shot
    grading or restyling pass: feed in a rendered Veo clip, ask Aleph to
    re-grade it warmer, change the lighting key, or transform the
    location seasonally. Costs 15 credits/sec ($0.15/sec) on top of the
    original generation, so use sparingly — most of the time the
    LOOKBOOK_GRADE ffmpeg chain is the right (free) tool.
    """
    kwargs: dict[str, Any] = {
        "model": GEN4_ALEPH_MODEL,
        "prompt_text": prompt,
        "video_uri": _data_uri(input_video, mime="video/mp4"),
    }
    if reference_image is not None:
        kwargs["reference_images"] = [{"tag": "style", "uri": _data_uri(reference_image)}]
    task = runway_client().video_to_video.create(**kwargs).wait_for_task_output()
    if not task.output:
        raise RuntimeError("Aleph: no output URL returned")
    return httpx.get(task.output[0], timeout=300.0).content


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


def claude_text(user: str, system: str = "", model: str = "claude-haiku-4-5-20251001",
                max_tokens: int = 1024) -> str:
    """Simple single-turn Claude call that returns the text response.

    Use for cheap rewriting, classification, or short generation tasks
    where a tool call is unnecessary overhead. Defaults to Haiku for cost.
    """
    messages: list[dict] = [{"role": "user", "content": user}]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    resp = claude().messages.create(**kwargs)
    return "".join(
        block.text for block in resp.content if hasattr(block, "text")
    ).strip()


@api_retry
def stable_audio(prompt: str, duration_seconds: int = 30) -> bytes:
    """Stability Stable Audio 2.5 — instrumental cinematic cues.

    Kept on the direct Stability API; Runway has no music model.
    """
    api_key  = _require_key("STABILITY_API_KEY")
    # Cap prompt to 450 chars — Stability has an undocumented limit and
    # returns an opaque 400 when it's exceeded.
    prompt   = prompt[:450]
    duration = float(min(max(duration_seconds, 1), 47))

    # Build proper multipart/form-data without a dummy "none" field —
    # the old files={"none": ""} trick causes 400 on some API versions.
    # NOTE: Stability rejects specific audio subtypes in the Accept
    # header — it requires "audio/*" or "application/json". The actual
    # output format is set via the output_format form field below.
    resp = httpx.post(
        "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "audio/*"},
        files={
            "prompt":        (None, prompt),
            "duration":      (None, str(duration)),
            "output_format": (None, "wav"),
        },
        timeout=180.0,
    )
    if not resp.is_success:
        # Surface the actual rejection reason before raising.
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(
            f"Stability audio {resp.status_code}: {detail}"
        )
    return resp.content


# Stability image model tiers — picked by quality vs cost.
# Stable Image Ultra is the best (SD3.5 Large, 8 credits / image)
# Stable Image Core is the cheapest (3 credits / image)
# SD3.5 Large is a middle ground (6.5 credits / image)
STABILITY_IMAGE_BASE = "https://api.stability.ai/v2beta/stable-image"


@api_retry
def stable_image(
    prompt: str,
    aspect_ratio: str = "16:9",
    tier: str = "core",
    output_format: str = "png",
) -> bytes:
    """Stability AI text-to-image — last-resort fallback in t2i chain.

    Stability Video is no longer available via API (deprecated 2025), so
    this function is image-only.

    Tiers (cost / quality):
      - "ultra" → /generate/ultra        SD3.5 Large, 8 credits, top quality
      - "sd3"   → /generate/sd3          SD3.5 Large, 6.5 credits
      - "core"  → /generate/core         3 credits, fastest (default)

    Aspect ratios: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21.
    Returns image bytes (default PNG).
    """
    api_key = _require_key("STABILITY_API_KEY")
    endpoint_map = {
        "ultra": f"{STABILITY_IMAGE_BASE}/generate/ultra",
        "sd3":   f"{STABILITY_IMAGE_BASE}/generate/sd3",
        "core":  f"{STABILITY_IMAGE_BASE}/generate/core",
    }
    url = endpoint_map.get(tier, endpoint_map["core"])

    # Cap prompt to 9500 chars (Stability limit ~10K, leave a margin).
    prompt = prompt[:9500]

    resp = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            # Stability requires "image/*" or "application/json" — NOT a
            # specific subtype like "image/png". The output format is
            # controlled by the output_format form field below.
            "Accept":        "image/*",
        },
        files={
            "prompt":        (None, prompt),
            "aspect_ratio":  (None, aspect_ratio),
            "output_format": (None, output_format),
        },
        timeout=180.0,
    )
    if not resp.is_success:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(f"Stability image {resp.status_code}: {detail}")
    return resp.content


# Runway TTS via ElevenLabs Multilingual v2 — for narration / voice-over.
# Available since Sept 2025; same auth as other Runway endpoints.
RUNWAY_TTS_MODEL = "eleven_multilingual_v2"

# Runway TTS preset names. Runway's text_to_speech API accepts a fixed
# enum of preset IDs — sending an ElevenLabs voice hash (or anything
# else) returns a 400 with the full list. The presets below are picked
# from Runway's catalog to give the pipeline 5 distinct narrator vibes.
# Override per-call by passing voice_id explicitly. To see the full
# list of currently-supported presets, send any wrong value to /v1/
# text_to_speech and Runway returns the enum in the error body.
RUNWAY_VOICE_IDS: dict[str, str] = {
    "rachel":    "Rachel",   # warm female narrator
    "george":    "Bernard",  # mature male narrator
    "antoni":    "Mark",     # clear male narrator
    "bella":     "Maya",     # soft female
    "sam":       "Tom",      # neutral male
}
DEFAULT_NARRATION_VOICE = "rachel"


# FLUX model strings.
# flux-2-pro-preview — latest FLUX.2, multi-reference editing (up to 8 refs)
# flux-pro-1.1       — fast text-to-image, no reference support
# flux-pro-1.1-ultra — highest quality text-to-image
# flux-dev            — cheaper, open weights
FLUX2_PRO_MODEL  = "flux-2-pro-preview"   # multi-reference, image editing
FLUX_PRO_MODEL   = "flux-pro-1.1"
FLUX_ULTRA_MODEL = "flux-pro-1.1-ultra"
FLUX_DEV_MODEL   = "flux-dev"
BFL_API_BASE     = "https://api.bfl.ai"   # global endpoint (was api.bfl.ml)


def flux_image(
    prompt: str,
    reference_images: list[bytes] | None = None,
    width: int = 1344,
    height: int = 768,
    model: str | None = None,
    safety_tolerance: int = 2,
) -> bytes:
    """FLUX image generation via the Black Forest Labs API (api.bfl.ai).

    Routes to the best model based on inputs:
      - reference_images provided → flux-2-pro-preview (FLUX.2)
        Accepts up to 8 reference images via input_image, input_image_2, …
        Excellent for character consistency and multi-ref composition.
      - no references → flux-pro-1.1 (fast, text-to-image only)

    Async workflow:
      1. POST /v1/{model} → {id, polling_url}
      2. Poll polling_url until status == "Ready"
      3. Download result["sample"] (signed URL, valid 10 min)

    Requires BFL_API_KEY. ~$0.05/image (Pro) or ~$0.025/image (Dev).
    """
    import base64 as _b64
    api_key = _require_key("BFL_API_KEY")
    headers = {"x-key": api_key, "Content-Type": "application/json"}

    # Choose model: FLUX.2 when refs are provided, Pro for text-only
    use_model = model or (FLUX2_PRO_MODEL if reference_images else FLUX_PRO_MODEL)

    body: dict = {
        "prompt":           prompt,
        "width":            width,
        "height":           height,
        "safety_tolerance": safety_tolerance,
        "output_format":    "jpeg",
    }

    # Pass reference images as input_image, input_image_2, ... input_image_8
    if reference_images:
        for i, ref in enumerate(reference_images[:8]):
            key = "input_image" if i == 0 else f"input_image_{i + 1}"
            body[key] = _b64.b64encode(ref).decode()

    submit = httpx.post(
        f"{BFL_API_BASE}/v1/{use_model}",
        headers=headers,
        json=body,
        timeout=30.0,
    )
    if not submit.is_success:
        try:
            detail = submit.json()
        except Exception:
            detail = submit.text[:400]
        raise RuntimeError(f"FLUX submit {submit.status_code}: {detail}")

    result_data   = submit.json()
    polling_url: str = result_data.get("polling_url") or (
        f"{BFL_API_BASE}/v1/get_result?id={result_data['id']}"
    )

    # Poll until ready (FLUX.2 typically 15-60 s, Pro ~10-30 s)
    for _ in range(240):
        time.sleep(0.5)
        poll = httpx.get(polling_url, headers=headers, timeout=30.0)
        poll.raise_for_status()
        data   = poll.json()
        status = data.get("status", "")
        if status == "Ready":
            image_url: str = data["result"]["sample"]
            return httpx.get(image_url, timeout=60.0).content
        if status in ("Error", "Failed", "Request Moderated", "Content Moderated"):
            raise RuntimeError(
                f"FLUX ({use_model}) failed: status={status!r} "
                f"id={result_data.get('id')}"
            )

    raise RuntimeError(
        f"FLUX ({use_model}) timed out. id={result_data.get('id')}"
    )


@api_retry
def elevenlabs_sfx(prompt: str, duration_seconds: int = 10) -> bytes:
    """ElevenLabs sound effects via Runway's /v1/sound_effect endpoint.

    The Runway SDK's ``sound_effect.create()`` does NOT accept any
    duration parameter — the model picks the length from the prompt
    (typically 0.5-22 seconds). To honor the caller's requested
    ``duration_seconds``, we trim or loop the returned audio with
    ffmpeg after the fact. This keeps the function's contract intact
    for produce.py callers that pass a specific scene length.
    """
    duration = max(1, min(int(duration_seconds), 22))
    task = runway_client().sound_effect.create(
        model=RUNWAY_SFX_MODEL,
        prompt_text=prompt,
    ).wait_for_task_output()
    if not task.output:
        raise RuntimeError("Runway SFX: no output URL returned")
    raw = httpx.get(task.output[0], timeout=120.0).content
    return _fit_audio_to_duration(raw, duration)


def _fit_audio_to_duration(audio_bytes: bytes, target_seconds: float) -> bytes:
    """Trim audio if longer than ``target_seconds``, loop if shorter.

    Returns 16-bit-PCM 44.1kHz WAV bytes for predictable downstream
    handling by the audio mix stage. If the source decoder happens to
    fail (rare; would require malformed input from Runway), the
    original bytes are returned unchanged so the caller can decide
    what to do.
    """
    if target_seconds <= 0:
        return audio_bytes
    import tempfile
    with tempfile.TemporaryDirectory(prefix="autofilm_sfx_") as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "in.bin"
        out_path = tmpdir / "out.wav"
        in_path.write_bytes(audio_bytes)
        try:
            # -stream_loop -1 makes input loop indefinitely; -t caps the
            # output at target_seconds. This handles both cases (input
            # longer than target → trim; input shorter → loop) in one
            # invocation.
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-stream_loop", "-1",
                    "-i", str(in_path),
                    "-t", f"{target_seconds:.2f}",
                    "-c:a", "pcm_s16le", "-ar", "44100",
                    str(out_path),
                ],
                check=True,
            )
            return out_path.read_bytes()
        except subprocess.CalledProcessError:
            return audio_bytes


@api_retry
def runway_tts(text: str, voice_id: str = "Rachel") -> bytes:
    """Text-to-speech narration via Runway (or direct ElevenLabs fallback).

    Runway's /v1/text_to_speech accepts a fixed enum of preset IDs (e.g.
    "Maya", "Bernard", "Rachel"). Sending anything else returns 400 with
    the full enum in the error body.

    Used by produce.py for voice-over narration layered over each scene.
    1 credit per 50 chars (~$0.01 via Runway, similar via ElevenLabs).

    Falls back to direct ElevenLabs API when:
      - the Runway SDK doesn't expose .text_to_speech (older version), or
      - the Runway call fails for transport reasons (not validation).
    """
    # ── Runway preset path ──────────────────────────────────────────────
    try:
        client = runway_client()
        if hasattr(client, "text_to_speech"):
            task = client.text_to_speech.create(
                model="eleven_multilingual_v2",
                prompt_text=text,
                # Nested dict keys go through verbatim — the SDK only
                # auto-converts top-level kwargs from snake_case to
                # camelCase. The API expects `presetId`, not `preset_id`.
                voice={"type": "runway-preset", "presetId": voice_id},
            ).wait_for_task_output()
            if task.output:
                return httpx.get(task.output[0], timeout=120.0).content
            raise RuntimeError("Runway TTS: no output URL returned")
    except Exception as e:                                  # noqa: BLE001
        # If we got a validation error (wrong preset), don't paper over
        # it with a fallback — the caller has a bug.
        if "presetId" in str(e) or "Invalid option" in str(e):
            raise
        # Otherwise (transport / SDK shape / 5xx), try ElevenLabs direct.
        last_runway_error = e

    # ── ElevenLabs direct fallback (requires ELEVENLABS_API_KEY) ────────
    el_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not el_key:
        raise RuntimeError(
            f"Runway TTS failed and no ELEVENLABS_API_KEY for fallback: "
            f"{last_runway_error}"
        )
    # ElevenLabs takes voice IDs (hashes), not preset names; map common
    # presets that Runway exposes back to known ElevenLabs voice hashes
    # so the fallback path still produces sensible-sounding narration.
    el_voice_map = {
        "Rachel":   "21m00Tcm4TlvDq8ikWAM",
        "Bernard":  "JBFqnCBsd6RMkjVDRZzb",
        "Mark":     "ErXwobaYiN019PkySvjV",
        "Maya":     "EXAVITQu4vr4xnSDxMaL",
        "Tom":      "yoZ06aMxZJJ28mfd3POQ",
    }
    el_voice = el_voice_map.get(voice_id, el_voice_map["Rachel"])
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{el_voice}",
        headers={
            "xi-api-key":    el_key,
            "Content-Type":  "application/json",
            "Accept":        "audio/mpeg",
        },
        json={
            "text":     text[:5000],
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability":         0.5,
                "similarity_boost":  0.75,
                "style":             0.2,
                "use_speaker_boost": True,
            },
        },
        timeout=120.0,
    )
    if not resp.is_success:
        try:    detail = resp.json()
        except Exception:    detail = resp.text[:400]
        raise RuntimeError(f"ElevenLabs TTS {resp.status_code}: {detail}")
    return resp.content


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
    """Pick which video model renders a shot.

    Args:
        desired_seconds: how long the shot should be on screen.
        tier: which model family to use.
            - "previs"   → veo3.1_fast at $0.15/sec (Runway has no Lite)
            - "fast"     → veo3.1_fast at $0.15/sec, native dialogue audio
            - "standard" → veo3.1 at $0.40/sec, native dialogue audio
            - "gen4.5"   → Runway Gen-4.5 at $0.12/sec, native ref-image lock
            - "seedance2"→ Seedance 2 at $0.36/sec, up to 15s in one call

    Returns a routing dict:
        {
          "model_key": str,           # key into VIDEO_MODELS
          "model_id":  str,           # the actual model identifier
          "segments":  [int],         # always single-element since we don't chain
          "estimated_cost": float,    # USD
          "rationale": str,           # for the bible / debugging
        }
    """
    desired_seconds = max(1, min(desired_seconds, MAX_PLANNED_SHOT_SECONDS))

    tier_to_key = {
        "previs":    "veo3.1_lite",     # alias, same id as fast
        "fast":      "veo3.1_fast",
        "standard":  "veo3.1_standard",
        "gen4.5":    "gen4.5",
        "gen45":     "gen4.5",          # convenience alias
        "seedance2": "seedance2",
        "seedance":  "seedance2",
    }
    model_key = tier_to_key.get(tier, "veo3.1_fast")
    m = VIDEO_MODELS[model_key]

    # seedance2 lets us bust the 8s cap; everything else still snaps to ≤8s.
    if model_key == "seedance2":
        desired_seconds = max(1, min(desired_seconds, m["max_seconds"]))

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
                        "axis": {
                            "type": "string",
                            "enum": ["cinematography", "color", "sound", "acting",
                                     "continuity", "fidelity"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
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


def encode_image_for_claude(
    img_bytes_or_path: bytes | Path,
    max_long_edge: int = 1024,
    quality: int = 80,
) -> tuple[str, str]:
    """Downscale + JPEG-compress an image for transmission to Claude.

    Anthropic's API has a 32 MB total request size limit. Native first-
    frame PNGs at 1792x1024 are ~3 MB each base64; sending more than ~10
    of them will trip the 413 Request Too Large error. Downscaling the
    long edge to 1024 px and encoding JPEG q=80 brings each to
    ~50-150 KB while keeping the image fully reviewable.

    Accepts raw bytes (e.g. an extract_video_frame() return value) or a
    Path to read from disk. Returns (media_type, base64_data) tuple.
    """
    from PIL import Image
    import io

    if isinstance(img_bytes_or_path, (str, Path)):
        img = Image.open(img_bytes_or_path)
    else:
        img = Image.open(io.BytesIO(img_bytes_or_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max_long_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()


def _critic_claude_stills(script: dict, frames: list[Path]) -> dict:
    """Claude reviews representative stills as a second opinion.

    Frames are downscaled and JPEG-compressed via encode_image_for_claude
    before sending — at native 1792x1024 PNG, 16 frames base64-encoded
    blows past Anthropic's 32 MB request size limit.
    """
    content: list[dict] = [
        {"type": "text", "text": (
            "Score this generated film on the six axes via the tool.\n\n"
            f"SOURCE SCRIPT (truncated):\n{json.dumps(script)[:15000]}\n\n"
            f"REPRESENTATIVE STILLS:\n"
        )},
    ]
    for p in frames:
        media_type, data = encode_image_for_claude(p)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
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
