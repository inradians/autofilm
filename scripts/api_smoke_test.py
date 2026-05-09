"""scripts/api_smoke_test.py — validate every image and video API endpoint.

Runs a minimal generation against every configured provider, reports
OK / FAIL / SKIP per endpoint, and prints a summary table with timing
and bytes returned.

SKIP = API key not set (not a failure)
FAIL = key set but the call errored

Usage
-----
    # All available endpoints (~$2-5 depending on keys set):
    python scripts/api_smoke_test.py

    # Images only:
    python scripts/api_smoke_test.py --image

    # Video only:
    python scripts/api_smoke_test.py --video

    # One specific endpoint:
    python scripts/api_smoke_test.py --only gpt_image
    python scripts/api_smoke_test.py --only flux-2-pro

    # List all endpoints without running:
    python scripts/api_smoke_test.py --list

What each test costs (approximate)
-----------------------------------
Image endpoints:
    gpt_image        ~$0.02   Runway / GPT Image 2 (low tier)
    openai_image     ~$0.04   OpenAI direct / GPT Image 2 (low quality)
    nano_banana      ~$0.02   Runway / Imagen 3 (with 1 ref)
    reve_create      ~$0.01   Reve / create (text-only, for health check)
    reve_remix       ~$0.01   Reve / remix (with 1 ref)
    flux-pro         ~$0.05   BFL / FLUX.1 Pro (text-only)
    flux-2-pro       ~$0.06   BFL / FLUX.2 Pro (with 1 ref)

Video endpoints:
    seedance         ~$0.36   Runway / SeedDance 2 (4s @ 720p)
    veo              ~$0.40   Runway / Veo 3.1 fast (5s @ 720p)
    google_veo       ~$0.15   Google / Veo 3.1 direct (5s)
    ltx-2-3-pro      ~$0.40   LTX / ltx-2-3-pro (6s @ 720p)
    ltx-2-3-fast     ~$0.24   LTX / ltx-2-3-fast (6s @ 720p)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Callable

# Make sure the project root is on the path so we can import prepare.py
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test frame / reference image ──────────────────────────────────────────────
# Generated programmatically so tests don't depend on any image API succeeding
# first. A simple 512×288 (16:9) gradient JPEG — valid image, fast to create.

def _make_test_frame() -> bytes:
    """Create a tiny 512×288 test JPEG without calling any API."""
    from PIL import Image as _PIL
    img = _PIL.new("RGB", (512, 288))
    pixels = img.load()
    for y in range(288):
        for x in range(512):
            r = int(x / 512 * 180) + 40
            g = int(y / 288 * 120) + 60
            b = 160
            pixels[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


TEST_PROMPT = (
    "Cinematic film still, 35mm anamorphic, shallow depth of field. "
    "A lone figure stands at the edge of a misty tropical coastline at golden hour. "
    "Warm amber light, lush palm fronds, crashing waves. Photorealistic."
)
TEST_VIDEO_PROMPT = (
    "Cinematic dolly shot, tropical coastline at golden hour, "
    "warm amber light, lush vegetation, ocean waves, atmospheric haze. "
    "Photorealistic, anamorphic lens flare. NO background music."
)


# ── Result tracking ───────────────────────────────────────────────────────────

class Result:
    def __init__(self, name: str):
        self.name = name
        self.status = "SKIP"   # SKIP | OK | FAIL
        self.elapsed = 0.0
        self.size_kb = 0
        self.note = ""

    def __repr__(self) -> str:
        icon = {"OK": "✓", "FAIL": "✗", "SKIP": "–"}.get(self.status, "?")
        extra = f"  {self.elapsed:.1f}s  {self.size_kb}kB" if self.status == "OK" else ""
        note  = f"  {self.note}" if self.note else ""
        return f"  {icon}  {self.name:<22}{extra}{note}"


results: list[Result] = []


def run_test(
    name: str,
    fn: Callable[[], bytes],
    required_key: str | None = None,
) -> Result:
    r = Result(name)
    results.append(r)

    if required_key and not os.environ.get(required_key):
        r.note = f"no {required_key}"
        return r

    t0 = time.time()
    try:
        data = fn()
        r.elapsed = time.time() - t0
        r.size_kb = len(data) // 1024
        r.status  = "OK"
        if r.size_kb < 2:
            r.status = "FAIL"
            r.note   = f"response too small ({r.size_kb}kB) — likely an error body"
    except Exception as e:
        r.elapsed = time.time() - t0
        r.status  = "FAIL"
        r.note    = str(e)[:120]

    return r


# ── Image tests ───────────────────────────────────────────────────────────────

def test_image_endpoints(only: set[str] | None) -> None:
    from prepare import (
        flux_image,
        gpt_image,
        nano_banana,
        openai_image,
        reve_image,
        FLUX_PRO_MODEL,
    )

    frame = _make_test_frame()

    def want(name: str) -> bool:
        return only is None or name in only

    print("\n── Image endpoints ──────────────────────────────────────────")

    if want("gpt_image"):
        run_test(
            "gpt_image",
            lambda: gpt_image(TEST_PROMPT, size="1344x768", quality="low"),
            required_key="RUNWAYML_API_SECRET",
        )

    if want("openai_image"):
        run_test(
            "openai_image",
            lambda: openai_image(TEST_PROMPT, size="1536x1024", quality="low"),
            required_key="OPENAI_API_KEY",
        )

    if want("nano_banana"):
        run_test(
            "nano_banana",
            lambda: nano_banana(TEST_PROMPT[:950], reference_images=[frame]),
            required_key="RUNWAYML_API_SECRET",
        )

    if want("reve_create"):
        run_test(
            "reve_create",
            lambda: reve_image(TEST_PROMPT, aspect_ratio="16:9"),
            required_key="REVE_API_KEY",
        )

    if want("reve_remix"):
        run_test(
            "reve_remix",
            lambda: reve_image(TEST_PROMPT, reference_images=[frame],
                               aspect_ratio="16:9"),
            required_key="REVE_API_KEY",
        )

    if want("flux-pro"):
        run_test(
            "flux-pro",
            lambda: flux_image(TEST_PROMPT, width=1344, height=768,
                               model=FLUX_PRO_MODEL),
            required_key="BFL_API_KEY",
        )

    if want("flux-2-pro"):
        run_test(
            "flux-2-pro",
            lambda: flux_image(TEST_PROMPT, reference_images=[frame],
                               width=1344, height=768),
            required_key="BFL_API_KEY",
        )


# ── Video tests ───────────────────────────────────────────────────────────────

def test_video_endpoints(only: set[str] | None) -> None:
    from prepare import (
        google_veo,
        ltx_video,
        seedance,
        veo,
        LTX_PRO_MODEL,
        LTX_FAST_MODEL,
    )

    frame = _make_test_frame()

    def want(name: str) -> bool:
        return only is None or name in only

    print("\n── Video endpoints ──────────────────────────────────────────")

    if want("seedance"):
        run_test(
            "seedance",
            lambda: seedance(TEST_VIDEO_PROMPT, first_frame=frame,
                             duration_seconds=4),
            required_key="RUNWAYML_API_SECRET",
        )

    if want("veo"):
        run_test(
            "veo",
            lambda: veo(TEST_VIDEO_PROMPT, first_frame=frame,
                        duration_seconds=5),
            required_key="RUNWAYML_API_SECRET",
        )

    if want("google_veo"):
        run_test(
            "google_veo",
            lambda: google_veo(TEST_VIDEO_PROMPT, first_frame=frame,
                               duration_seconds=5, resolution="720p"),
            required_key="GOOGLE_AI_API_KEY",
        )

    if want("ltx-2-3-pro"):
        run_test(
            "ltx-2-3-pro",
            lambda: ltx_video(TEST_VIDEO_PROMPT, first_frame=frame,
                               duration_seconds=6, resolution="720p",
                               model=LTX_PRO_MODEL),
            required_key="LTX_API_KEY",
        )

    if want("ltx-2-3-fast"):
        run_test(
            "ltx-2-3-fast",
            lambda: ltx_video(TEST_VIDEO_PROMPT, first_frame=frame,
                               duration_seconds=6, resolution="720p",
                               model=LTX_FAST_MODEL),
            required_key="LTX_API_KEY",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_IMAGE = {
    "gpt_image", "openai_image", "nano_banana",
    "reve_create", "reve_remix", "flux-pro", "flux-2-pro",
}
ALL_VIDEO = {
    "seedance", "veo", "google_veo", "ltx-2-3-pro", "ltx-2-3-fast",
}
ALL_ENDPOINTS = ALL_IMAGE | ALL_VIDEO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test every image and video API endpoint."
    )
    parser.add_argument("--image", action="store_true",
                        help="Test image endpoints only")
    parser.add_argument("--video", action="store_true",
                        help="Test video endpoints only")
    parser.add_argument("--only", metavar="NAME", action="append", default=[],
                        help="Test specific endpoint(s) by name (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="List all endpoint names and exit")
    args = parser.parse_args()

    if args.list:
        print("Image endpoints:")
        for n in sorted(ALL_IMAGE):
            print(f"  {n}")
        print("Video endpoints:")
        for n in sorted(ALL_VIDEO):
            print(f"  {n}")
        return

    # Load .env if present
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print(f"Loaded {env_path}")

    only: set[str] | None = set(args.only) if args.only else None

    run_image = not args.video or args.image
    run_video = not args.image or args.video

    t_start = time.time()

    if run_image:
        test_image_endpoints(only)
    if run_video:
        test_video_endpoints(only)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    ok   = [r for r in results if r.status == "OK"]
    fail = [r for r in results if r.status == "FAIL"]
    skip = [r for r in results if r.status == "SKIP"]

    print(f"\n{'─'*60}")
    print(f"  Results  ({total_elapsed:.1f}s total)")
    print(f"{'─'*60}")
    for r in results:
        print(r)
    print(f"{'─'*60}")
    print(
        f"  {len(ok)} OK  ·  {len(fail)} FAIL  ·  {len(skip)} SKIP "
        f"(key not set)"
    )

    if fail:
        print("\nFailed endpoints:")
        for r in fail:
            print(f"  ✗ {r.name}: {r.note}")
        sys.exit(1)


if __name__ == "__main__":
    main()
