"""scripts/runway_smoke_test.py — validate every Runway SDK call site
in prepare.py against the LIVE API.

The migration from OpenAI/Google AI/ElevenLabs to a Runway-consolidated
stack changed the SDK parameter names and request body shapes for every
image/video/SFX/TTS call. The static type-check in CI can't catch
mistakes like "promptImage vs prompt_image" or "presetId vs preset_id"
because those errors only surface against the live API.

This script is the live-validation step. It exercises each helper in
``prepare.py`` with the cheapest possible inputs and reports OK/FAIL
per call — pointing you at the exact line in prepare.py to fix when
something errors.

Designed to run on YOUR machine, not in the autofilm sandbox or the
agent's container — the agent's runner has a network allowlist that
doesn't include api.dev.runwayml.com, so live calls have to happen
locally.

Usage
-----

    # Quick sweep: all image + audio calls. ~$0.30 total, ~30 sec.
    RUNWAYML_API_SECRET=key_... python scripts/runway_smoke_test.py

    # Full sweep including Veo + Aleph (each ~$0.60). ~$1.50 total, 3-5 min.
    RUNWAYML_API_SECRET=key_... python scripts/runway_smoke_test.py --include-video

    # Just one specific call:
    python scripts/runway_smoke_test.py --only veo
    python scripts/runway_smoke_test.py --only runway_image --only nano_banana

What it tests
-------------

Image  (~$0.30):
    gpt_image            — `runway_image()` via Runway's /v1/text_to_image
                           with model=gpt_image_2, low quality
    nano_banana          — `runway_image()` with model=gemini_image3_pro
    runway_image+refs    — gen4_image_turbo with one reference image
                           (validates the {tag, uri} refs payload shape)

Audio (~$0.05):
    elevenlabs_sfx       — Runway /v1/sound_effect, 2 seconds
    runway_tts           — Runway /v1/text_to_speech, "hello world"
                           (validates voice = {type, presetId} shape)

Video (~$1.20, --include-video):
    veo                  — image-to-video with veo3.1_fast, 4 seconds
                           (validates promptImage + duration + ratio)
    aleph_video_to_video — gen4_aleph, 4 seconds, on the veo output above
                           (validates videoUri shape)

What it does NOT test
---------------------

- The full produce.py pipeline. That requires the book PDF + Anthropic
  + Stability keys + ~$22 of Runway credit and 15-25 minutes wall-clock.
  Use `scripts/preflight_e2e.py` for plumbing-only validation against
  mocks (free, ~10 sec).
- Image quality, video coherence, audio fidelity. The smoke test only
  proves the SDK calls don't error — it doesn't critique output.
- Rate limits or billing edge cases. The script makes one call per
  helper, sequentially.

Output
------

Each test prints OK or FAIL; FAIL includes the prepare.py line number
range to inspect, the full Runway error, and (where useful) the body
the SDK sent.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to sys.path so we can import prepare.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))


# ─── Output directory ──────────────────────────────────────────────
# Each smoke-test run gets its own timestamped subdir under
# experiments/_smoke_tests/. Outputs persist between runs so you can
# diff results, compare a generated image against last week's, etc.
# The leading underscore on _smoke_tests excludes it from the agent's
# experiment iteration (Experiment.load / iter_all_experiments).
_RUN_TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_OUTPUT_DIR = _ROOT / "experiments" / "_smoke_tests" / _RUN_TIMESTAMP


def _output_path(filename: str) -> Path:
    """Resolve a filename to a path inside this run's smoke-test dir."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR / filename


# ─── Test fixtures ──────────────────────────────────────────────────
def _build_test_png() -> bytes:
    """A minimal solid-color PNG (256x144) for use as a reference image
    or first frame. Generated on-the-fly so this script has no input
    requirements."""
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (256, 144), color=(60, 80, 100))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    except ImportError:
        # Fall back to a hand-rolled 1x1 PNG. Runway might reject if it's
        # too small, but worth trying before failing the script outright.
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108"
            "060000001f15c4890000000d4944415478da636060600000000"
            "5000115c2bb96d0000000049454e44ae426082"
        )


# ─── Test runner ────────────────────────────────────────────────────
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.status: str | None = None  # "OK" | "FAIL" | "SKIP"
        self.detail: str = ""
        self.elapsed_sec: float = 0.0
        self.output_path: Path | None = None  # where the artifact was saved

    def ok(self, detail: str = ""):
        self.status = "OK"
        self.detail = detail

    def fail(self, exc: Exception, lines: str = ""):
        self.status = "FAIL"
        msg = f"{type(exc).__name__}: {exc}"
        if lines:
            msg = f"{msg}\n        check prepare.py {lines}"
        self.detail = msg

    def skip(self, why: str):
        self.status = "SKIP"
        self.detail = why

    def render(self) -> str:
        color = {"OK": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}[self.status]
        reset = "\033[0m"
        if not sys.stdout.isatty():
            color = reset = ""
        line = f"  {color}{self.status:4s}{reset}  {self.name:24s}  ({self.elapsed_sec:5.1f}s)"
        if self.detail:
            for ln in self.detail.splitlines():
                line += f"\n          {ln}"
        return line


# ─── Individual tests ───────────────────────────────────────────────
def test_gpt_image(prepare) -> TestResult:
    """gpt_image() routes through runway_image() with model=gpt_image_2.
    Validates: SDK call to /v1/text_to_image, prompt_text + ratio params,
    output URL fetch."""
    r = TestResult("gpt_image")
    t0 = time.time()
    try:
        # quality="low" is the cheapest gpt_image_2 tier — ~1 credit.
        out = prepare.gpt_image(
            "minimal grey gradient, abstract, 1024x1024",
            size="1024x1024", quality="low",
        )
        if not isinstance(out, bytes) or len(out) < 1000:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("gpt_image.png")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB png → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 538-558  (gpt_image → runway_image)")
    r.elapsed_sec = time.time() - t0
    return r


def test_nano_banana(prepare) -> TestResult:
    """nano_banana() routes through runway_image() with model=gemini_image3_pro."""
    r = TestResult("nano_banana")
    t0 = time.time()
    try:
        out = prepare.nano_banana("minimal grey gradient, abstract")
        if not isinstance(out, bytes) or len(out) < 1000:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("nano_banana.png")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB png → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 560-572  (nano_banana → runway_image)")
    r.elapsed_sec = time.time() - t0
    return r


def test_runway_image_with_refs(prepare) -> TestResult:
    """runway_image() with reference_images. This is the most complex
    image call shape — validates the {tag, uri} refs payload."""
    r = TestResult("runway_image+refs")
    t0 = time.time()
    try:
        ref = _build_test_png()
        out = prepare.runway_image(
            "an abstract grey-blue scene, in the style of @style",
            reference_images=[ref],
            reference_tags=["style"],
            model=prepare.GEN4_IMAGE_TURBO,  # cheapest, ~2 credits
        )
        if not isinstance(out, bytes) or len(out) < 1000:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("runway_image_refs.png")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB png → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 487-535  (runway_image, refs payload)")
    r.elapsed_sec = time.time() - t0
    return r


def test_elevenlabs_sfx(prepare) -> TestResult:
    """elevenlabs_sfx() via Runway's /v1/sound_effect endpoint."""
    r = TestResult("elevenlabs_sfx")
    t0 = time.time()
    try:
        out = prepare.elevenlabs_sfx("rain on a window pane", duration_seconds=2)
        if not isinstance(out, bytes) or len(out) < 100:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("elevenlabs_sfx.wav")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB audio → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 688-703  (elevenlabs_sfx)")
    r.elapsed_sec = time.time() - t0
    return r


def test_runway_tts(prepare) -> TestResult:
    """runway_tts() via Runway's /v1/text_to_speech endpoint. Validates
    the voice = {type: runway-preset, presetId: ...} nested-dict shape."""
    r = TestResult("runway_tts")
    t0 = time.time()
    try:
        out = prepare.runway_tts("Hello, world.")
        if not isinstance(out, bytes) or len(out) < 100:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("runway_tts.mp3")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB audio → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 706-722  (runway_tts; presetId nested-dict shape)")
    r.elapsed_sec = time.time() - t0
    return r


def test_veo(prepare) -> TestResult:
    """veo() via /v1/image_to_video with veo3.1_fast, 4 seconds (cheapest
    Veo). Validates promptImage + ratio + duration."""
    r = TestResult("veo (4s)")
    t0 = time.time()
    try:
        first = _build_test_png()
        out = prepare.veo(
            "slow zoom on a still grey-blue gradient",
            first_frame=first,
            duration_seconds=4,
            resolution="720p",
        )
        if not isinstance(out, bytes) or len(out) < 10_000:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("veo.mp4")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB mp4 → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 573-622  (veo; promptImage + duration + ratio)")
    r.elapsed_sec = time.time() - t0
    return r


def test_aleph(prepare) -> TestResult:
    """aleph_video_to_video() via /v1/video_to_video with gen4_aleph.
    Reads the veo output from the previous test."""
    r = TestResult("aleph_v2v (4s)")
    t0 = time.time()
    veo_out = _output_path("veo.mp4")
    if not veo_out.exists():
        r.skip("requires veo test to run first (--include-video)")
        r.elapsed_sec = time.time() - t0
        return r
    try:
        out = prepare.aleph_video_to_video(
            "regrade warmer, golden hour",
            input_video=veo_out.read_bytes(),
        )
        if not isinstance(out, bytes) or len(out) < 10_000:
            raise RuntimeError(f"unexpected output: {len(out) if isinstance(out, bytes) else type(out)} bytes")
        path = _output_path("aleph.mp4")
        path.write_bytes(out)
        r.ok(f"received {len(out)//1024}kB mp4 → {path.name}")
        r.output_path = path
    except Exception as e:
        r.fail(e, "lines 624-647  (aleph_video_to_video; videoUri shape)")
    r.elapsed_sec = time.time() - t0
    return r


# ─── Catalog ────────────────────────────────────────────────────────
ALL_TESTS = {
    "gpt_image":        (test_gpt_image,        "image", 0.01),
    "nano_banana":      (test_nano_banana,      "image", 0.20),
    "runway_image":     (test_runway_image_with_refs, "image", 0.02),
    "elevenlabs_sfx":   (test_elevenlabs_sfx,   "audio", 0.02),
    "runway_tts":       (test_runway_tts,       "audio", 0.01),
    "veo":              (test_veo,              "video", 0.60),
    "aleph":            (test_aleph,            "video", 0.60),
}


# ─── Main ───────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Runway SDK calls in prepare.py against the live API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--include-video", action="store_true",
        help="Include Veo + Aleph tests (~$1.20 extra, 3-5 min wall-clock)",
    )
    parser.add_argument(
        "--only", action="append", choices=list(ALL_TESTS),
        help="Run only the named test(s). Repeatable. Mutually exclusive with default sweep.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the planned tests + estimated cost and exit without calling Runway.",
    )
    args = parser.parse_args()

    # Load .env so RUNWAYML_API_SECRET resolves whether it's exported in
    # the shell or stored in the project's .env file. Without this the
    # env-check below fires before prepare.py's module-level load_dotenv
    # runs, and users get the misleading "not set" error even when the
    # key is sitting in .env.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv missing; user must export the var manually

    # Pick which tests run.
    if args.only:
        chosen = list(args.only)
    elif args.include_video:
        chosen = list(ALL_TESTS.keys())
    else:
        chosen = [k for k, (_, kind, _) in ALL_TESTS.items() if kind in ("image", "audio")]

    estimated_cost = sum(ALL_TESTS[k][2] for k in chosen)

    print()
    print("Runway smoke test — live API validation for prepare.py")
    print("=" * 56)
    print(f"  tests       : {', '.join(chosen)}")
    print(f"  est. cost   : ~${estimated_cost:.2f} ({len(chosen)} call{'s' if len(chosen) != 1 else ''})")

    if args.dry_run:
        print("  --dry-run   : not making any calls.")
        return 0

    # Verify env. We don't validate the key shape — let Runway return 401
    # if it's wrong. That's a clean signal.
    if not os.environ.get("RUNWAYML_API_SECRET"):
        print()
        print("  ERROR: RUNWAYML_API_SECRET environment variable not set.")
        print("  Run with: RUNWAYML_API_SECRET=key_... python scripts/runway_smoke_test.py")
        return 2

    # Import prepare. This is where missing deps (runwayml SDK, httpx,
    # tenacity, etc.) surface — surface them clearly rather than as a
    # generic ImportError mid-test.
    print()
    print("  Importing prepare.py...")
    try:
        import prepare
    except ImportError as e:
        print(f"  ERROR: failed to import prepare.py: {e}")
        print( "         did you `uv sync` or `pip install -e .`?")
        return 2

    print(f"  prepare.py loaded; runway client lazy-init on first call.")
    print()
    print("  ─" * 28)

    results: list[TestResult] = []
    for name in chosen:
        fn, _, _ = ALL_TESTS[name]
        try:
            r = fn(prepare)
        except Exception as e:  # noqa: BLE001 — last-resort catch
            r = TestResult(name)
            r.fail(e, "(uncaught — see traceback below)")
            traceback.print_exc()
        results.append(r)
        print(r.render())

    print("  ─" * 28)
    n_ok   = sum(1 for r in results if r.status == "OK")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    print()
    print(f"  Summary: {n_ok} OK, {n_fail} FAIL, {n_skip} SKIP  (of {len(results)})")
    if n_fail == 0 and n_ok > 0:
        print("  All tested SDK calls succeeded against the live Runway API.")
        print("  prepare.py's migration is validated for the covered surface.")
    elif n_fail > 0:
        print("  Some calls failed. The detail line above each FAIL points")
        print("  at the prepare.py block that needs adjustment.")

    # Write a persistent summary markdown in the run directory so users
    # can diff results between runs, see what files were generated, etc.
    # This is the "bible for smoke tests" the user asked for.
    summary_md = _output_path("summary.md")
    with summary_md.open("w") as f:
        f.write(f"# Runway smoke test — {_RUN_TIMESTAMP.replace('_', ' ')}\n\n")
        f.write("| Test | Status | Output | Detail |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            output_cell = r.output_path.name if r.output_path else "—"
            detail_cell = r.detail.split("\n")[0] if r.detail else "—"
            f.write(f"| {r.name} | {r.status} | {output_cell} | {detail_cell} |\n")
        f.write(f"\n**Total cost:** ~${estimated_cost:.2f}\n")
        f.write(f"\n**Artifacts:** {_OUTPUT_DIR.relative_to(_ROOT)}\n")
    print(f"  Summary written: {summary_md.relative_to(_ROOT)}")
    print()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
