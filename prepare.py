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

# Source book — Crichton's Jurassic Park
BOOK_PDF_PATH = Path(
    os.getenv("BOOK_PDF_PATH", "/mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf")
)

# Cap on how many scenes to render per experiment. Keeps each run on a
# fixed budget, like autoresearch's 5-min training cap.
MAX_SCENES = int(os.getenv("MAX_SCENES", "3"))

# --- Models (SOTA stack as of May 2026, accessed via Runway + Anthropic + Stability) ---
# Anthropic: Claude Opus 4.7 (text/critic-stills, direct API).
# Google AI: Gemini 3 Pro (long-video critic only — Runway has no video-review LLM).
# Runway: image, video, and SFX generation. See https://docs.dev.runwayml.com/guides/models/
# Stability: Stable Audio 2.5 (music score — Runway has no music model).
CLAUDE_MODEL = "claude-opus-4-7"
GEMINI_PRO_MODEL = "gemini-3-pro"

# Runway image model IDs. The Runway endpoint is /v1/text_to_image with the
# `model` field selecting which generator. All three accept `referenceImages`
# (gen4_image_turbo *requires* it). Pricing is in credits at $0.01/credit.
GPT_IMAGE_MODEL    = "gpt_image_2"          # 1-41 credits/image (high@1K=20, high@4K=41)
NANO_BANANA_MODEL  = "gemini_image3_pro"    # 20 credits @ 1K/2K, 40 credits @ 4K
GEN4_IMAGE_MODEL   = "gen4_image"           # 5 credits @ 720p, 8 credits @ 1080p — native ref-image support
GEN4_IMAGE_TURBO   = "gen4_image_turbo"     # 2 credits, references REQUIRED
GEMINI_FLASH_MODEL = "gemini_2.5_flash"     # 5 credits, any resolution

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
        return cls(exp_id=exp_id, root=root)

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

    @property
    def book_slug(self) -> str:
        """Read the book slug stamped at experiment creation. Returns
        'unknown_book' for old flat-layout experiments that pre-date this
        field."""
        f = self.root / "book.txt"
        return f.read_text().strip() if f.exists() else "unknown_book"

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


def nano_banana(prompt: str, reference_images: list[bytes] | None = None) -> bytes:
    """Nano Banana — Gemini Image 3 Pro via Runway.

    Identity locking and multi-image fusion. Up to 3 reference images,
    matching Runway's referenceImages cap (the old direct API allowed up
    to 14 elements; this is a small regression but the gen4_image refs
    system more than compensates for character continuity).
    """
    return runway_image(prompt, reference_images=reference_images, model=NANO_BANANA_MODEL)


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


@api_retry
def stable_audio(prompt: str, duration_seconds: int = 30) -> bytes:
    """Stability Stable Audio 2.5 — instrumental cinematic cues.

    Kept on the direct Stability API; Runway has no music model.
    """
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
def runway_tts(text: str, voice_id: str = "Maya") -> bytes:
    """ElevenLabs text-to-speech via Runway's /v1/text_to_speech endpoint.

    NEW capability — useful for narrator/voiceover that the old pipeline
    couldn't generate (Veo's native audio only covers in-frame dialogue).
    1 credit per 50 chars (~$0.01). produce.py doesn't call this by
    default; expose it to the agent via the import surface.
    """
    task = runway_client().text_to_speech.create(
        model="eleven_multilingual_v2",
        prompt_text=text,
        # Note: nested dict keys go through verbatim — the SDK only
        # auto-converts top-level kwargs from snake_case to camelCase.
        # The API expects `presetId`, not `preset_id`.
        voice={"type": "runway-preset", "presetId": voice_id},
    ).wait_for_task_output()
    if not task.output:
        raise RuntimeError("Runway TTS: no output URL returned")
    return httpx.get(task.output[0], timeout=120.0).content


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
