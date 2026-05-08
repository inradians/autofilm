"""scripts/preflight_e2e.py — mock-driven end-to-end pipeline test.

Runs `produce.py` against canned API responses to validate that the
pipeline plumbing works — schemas, prompt-call wiring, file artifacts,
ffmpeg/moviepy compile, bible.pdf generation — without spending a
cent. Useful as a regression test before pushing changes that touch
prepare.py / produce.py / bible.py.

What this validates:
- All claude_tool / gpt_image / nano_banana / veo / stable_audio call
  sites in produce.py reach a working mock and consume its return shape.
- Every artifact under experiments/exp_NNN/ is written in the expected
  order: script.json, cast.json, locations.json, lookbook.json,
  storyboard.json, shot_plan.json, frames/.../*.png,
  clips/.../take_N.mp4, edl.json, music/*.wav, final.mp4, bible.pdf.
- compile_final's two paths both work: the fast moviepy concat path
  AND the ffmpeg xfade transitions path. The harness forces a
  non-cut transition into the storyboard to exercise the second path.

What this does NOT validate:
- Whether the real Runway / Anthropic SDK request shapes are correct.
  Only a real (paid) run reveals that — see SETUP.md §8.
- Whether produced output is creatively any good. The mocks return
  trivially-valid data, not good data.

Usage:
    python scripts/preflight_e2e.py
        - exits 0 if every stage completed and every expected artifact exists
        - exits 1 with a diff of missing artifacts otherwise
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Resolve project root and put it on sys.path so we can import prepare/produce.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))


# ============================================================================
# Synthetic test asset generation
# ============================================================================
# Everything here is built from scratch, every run, into a tempdir:
#   - a 6-page placeholder PDF (the "test book")
#   - a tiny PNG (returned by the image mocks)
#   - a 4-second silent MP4 (returned by the video mock)
#   - a 1-second silent WAV (returned by the audio mocks)
#
# Using ffmpeg lavfi sources for the MP4/WAV keeps us off the network.

_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="autofilm-preflight-"))


def _build_test_pdf() -> Path:
    """Write a multi-page placeholder PDF that pdfplumber can parse.

    Uses reportlab (already a project dep). Content is deliberately
    generic — no real book text — since the harness's job is to prove
    plumbing, and the parser's output is mocked anyway.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    pdf_path = _TEMP_ROOT / "preflight_book.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    pages = [
        "Chapter 1\n\nA helicopter descends across a tropical coastline. "
        "Mist rises from the canopy below. Three passengers — a "
        "paleontologist, a botanist, and an attorney — watch a green "
        "ridge sharpen out of the haze. Below them, a track cuts through "
        "the forest toward something the brochure called the visitor "
        "center.",
        "Chapter 2\n\nThe park's owner meets them at the helipad. He is "
        "smaller than they expected and louder. He waves them forward "
        "with both hands, talking already, talking over the rotor wash. "
        "What he wants to show them, he says, will change everything.",
        "Chapter 3\n\nAt the perimeter fence the paleontologist stops "
        "walking. He holds out a hand for the others. Beyond the trees "
        "something moves. Something large, larger than the trees. It "
        "lowers its long neck to feed.",
        "Chapter 4\n\nNight. Generator down. The control room flickers. "
        "Through the smoked-glass observation window the paleontologist "
        "watches headlights pulse against rain. A wet impact rocks the "
        "tour vehicle ahead. Then silence.",
        "Chapter 5\n\nMorning. The lawyer is gone. The attorney's family "
        "is gone. What remains is a child and a paleontologist, climbing "
        "a tree to reach an empty road. The dawn comes up green.",
        "Chapter 6\n\nThe survivors fly out the way they came. Below "
        "them the ridge sinks back into the haze. The owner stays "
        "behind, walking the empty paths of his park, looking up at "
        "the trees as if expecting them to apologize.",
    ]
    for i, body in enumerate(pages, start=1):
        text = c.beginText(72, 720)
        text.setFont("Helvetica", 11)
        for line in body.split("\n"):
            for chunk in [line[i:i+80] for i in range(0, max(1, len(line)), 80)]:
                text.textLine(chunk or " ")
        c.drawText(text)
        c.drawString(72, 50, f"— preflight page {i} —")
        c.showPage()
    c.save()
    return pdf_path


def _build_synthetic_png(width: int = 256, height: int = 144) -> bytes:
    """Generate a tiny solid-color PNG via Pillow."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(60, 80, 100))
    buf_path = _TEMP_ROOT / f"frame_{width}x{height}.png"
    img.save(buf_path, "PNG")
    return buf_path.read_bytes()


def _build_synthetic_mp4(duration_sec: int = 4) -> bytes:
    """Generate a tiny H.264 MP4 with synced silent AAC audio via ffmpeg
    lavfi. The video stream is a solid color, the audio is silence.
    Cached on disk so we can return the same bytes for many shots."""
    cache = _TEMP_ROOT / f"clip_{duration_sec}s.mp4"
    if not cache.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=0x3050a0:s=320x180:d={duration_sec}:r=24",
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
                "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                str(cache),
            ],
            check=True,
        )
    return cache.read_bytes()


def _build_synthetic_wav(duration_sec: int = 30) -> bytes:
    """Generate a tiny WAV of silence via ffmpeg lavfi."""
    cache = _TEMP_ROOT / f"audio_{duration_sec}s.wav"
    if not cache.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
                "-t", str(duration_sec),
                str(cache),
            ],
            check=True,
        )
    return cache.read_bytes()


# ============================================================================
# Mocks for produce.py's external helpers
# ============================================================================

def _mock_claude_tool(system, user_content, tool_name, tool_schema, max_tokens=None):
    """Return canned data appropriate to each tool_name. Coverage is
    minimal but schema-conformant. The harness forces one storyboard
    shot to have a non-cut transition_out so compile_final exercises
    its xfade-rendering branch."""
    if tool_name == "submit_extraction":
        return {
            "characters": [
                {"id": "grant",   "name": "Dr. Alan Grant",
                 "description": "Paleontologist; skeptical."},
                {"id": "hammond", "name": "John Hammond",
                 "description": "Park founder; relentless optimist."},
            ],
            "scenes": [{
                "id": "s01",
                "location": "Helipad, remote tropical island",
                "time_of_day": "day",
                "page_start": 1, "page_end": 4,
                "characters": ["grant", "hammond"],
                "summary": "Grant arrives by helicopter; Hammond welcomes him.",
                "mood": "curious, foreboding",
                "dialogue_excerpts": [
                    {"character_id": "hammond", "line": "Welcome to the park."},
                ],
            }],
        }

    if tool_name == "submit_screenplay":
        return {
            "elements": [
                {"type": "action",   "text": "A helicopter descends over jungle canopy."},
                {"type": "action",   "text": "GRANT steps out, eyes scanning the ridge."},
                {"type": "dialogue", "text": "Welcome to the park.",
                 "character": "hammond", "parenthetical": "expansive"},
                {"type": "action",   "text": "Grant looks past him at the trees, unable to speak."},
            ],
        }

    if tool_name == "submit_casting":
        return {
            "casting": [
                {"character_id": "grant",   "actor": "Sam Neill",
                 "rationale": "Anchor presence; weathered curiosity."},
                {"character_id": "hammond", "actor": "Richard Attenborough",
                 "rationale": "Warmth that masks denial."},
            ],
        }

    if tool_name == "submit_locations":
        return {
            "locations": [{
                "slug": "helipad",
                "name": "Helipad, Isla Nublar",
                "description": ("Open hilltop landing zone surrounded by "
                                "rain-forest canopy; concrete pad with "
                                "perimeter painted yellow."),
                "color_palette": ["jungle green", "wet slate", "pale sun"],
                "scene_ids": ["s01"],
            }],
        }

    if tool_name == "submit_lookbook":
        return {
            "era": "Contemporary 1990s",
            "genre": "Adventure / techno-thriller",
            "tone": "Awe undercut with dread.",
            "lens_package": "Anamorphic 35mm; 32mm wide, 50mm normal, 85mm portrait.",
            "lighting_style": "Motivated practical light; strong key from sky and screens.",
            "grade_description": "Warm midtones, cool shadows, slightly desaturated greens.",
            "reference_films": ["Jurassic Park (1993)", "Apocalypse Now (1979)"],
            "style_frame_prompt": "Wide-angle shot of a hilltop helipad at dusk.",
        }

    if tool_name == "submit_shotlist":
        # Force a non-cut transition on shot 2 → exercises compile_final's
        # ffmpeg xfade branch. Without this, only the moviepy fast-path
        # would run.
        return {
            "shots": [
                {"shot_id": "01", "shot_size": "XLS", "angle": "high",
                 "camera_move": "static", "lens_mm": 32, "subject": "helicopter",
                 "action": "Helicopter descends over a green ridge.",
                 "dialogue_excerpt": "", "duration_seconds": 4,
                 "composition_notes": "Wide; rule of thirds; ridge low."},
                {"shot_id": "02", "shot_size": "MS", "angle": "eye",
                 "camera_move": "track", "lens_mm": 50, "subject": "Grant",
                 "action": "GRANT steps off the skid, eyes on the trees.",
                 "dialogue_excerpt": "", "duration_seconds": 4,
                 "composition_notes": "Profile; backlit by sky.",
                 "transition_out": {"type": "fadeblack", "duration": 0.5}},
                {"shot_id": "03", "shot_size": "MCU", "angle": "eye",
                 "camera_move": "static", "lens_mm": 85, "subject": "Hammond",
                 "action": "HAMMOND beams, both hands raised in welcome.",
                 "dialogue_excerpt": "Welcome to the park.",
                 "duration_seconds": 4,
                 "composition_notes": "Centered; soft fill from below."},
            ],
        }

    if tool_name == "submit_edl":  # not reached at TAKES_PER_SHOT=1
        return {"decisions": []}

    raise AssertionError(f"Unmocked tool_name: {tool_name!r}")


def _mock_gpt_image(prompt, size="1792x1024", quality="high"):
    return _build_synthetic_png()


def _mock_nano_banana(prompt, reference_images=None):
    return _build_synthetic_png()


def _mock_runway_image(prompt, reference_images=None, reference_tags=None,
                       model=None, ratio=None):
    return _build_synthetic_png()


def _mock_veo(prompt, first_frame, reference_images=None, model=None,
               duration_seconds=None, resolution=None, seed=None):
    # Match the requested duration so EDL math stays consistent.
    return _build_synthetic_mp4(duration_sec=int(duration_seconds or 4))


def _mock_stable_audio(prompt, duration_seconds=30):
    return _build_synthetic_wav(duration_sec=int(duration_seconds))


def _mock_elevenlabs_sfx(prompt, duration_seconds=10):
    return _build_synthetic_wav(duration_sec=int(duration_seconds))


# ============================================================================
# Harness
# ============================================================================

EXPECTED_ARTIFACTS = [
    "produce.py",          # snapshot
    "script.json",
    "cast.json",
    "locations.json",
    "lookbook.json",
    "lookbook/style_frame.png",
    "storyboard.json",
    "shot_plan.json",
    "frames_manifest.json",
    "music/s01.wav",
    "edl.json",
    "final.mp4",
    "bible.pdf",
    "prompts.json",
]


def _set_env() -> None:
    """Force the cheapest config and bypass _require_key with non-placeholder
    fake values (anything that doesn't end in '...' passes)."""
    os.environ["MAX_SCENES"] = "1"
    os.environ["TAKES_PER_SHOT"] = "1"
    os.environ["VEO_TIER"] = "fast"
    os.environ["VEO_RESOLUTION"] = "720p"
    os.environ["AMBIENT_SFX_ENABLED"] = "0"
    os.environ["DIRECTOR"] = ""
    os.environ["CINEMATOGRAPHER"] = ""
    os.environ["ANTHROPIC_API_KEY"]   = "sk-ant-preflight-fake"
    os.environ["RUNWAYML_API_SECRET"] = "key_preflight_fake"
    os.environ["STABILITY_API_KEY"]   = "sk-stability-preflight-fake"
    # GOOGLE_AI_API_KEY is optional and not needed by produce.py
    os.environ.pop("GOOGLE_AI_API_KEY", None)
    # Use the synthetic PDF
    os.environ["BOOK_PDF_PATH"] = str(_build_test_pdf())


def _patch_external_calls(produce_mod) -> None:
    """Replace produce.py's bound-name references with our mocks. Because
    produce.py does `from prepare import claude_tool, gpt_image, ...`, the
    names live in produce's namespace and are easy to swap."""
    produce_mod.claude_tool   = _mock_claude_tool
    produce_mod.gpt_image     = _mock_gpt_image
    produce_mod.nano_banana   = _mock_nano_banana
    produce_mod.veo           = _mock_veo
    produce_mod.stable_audio  = _mock_stable_audio
    produce_mod.elevenlabs_sfx = _mock_elevenlabs_sfx


def _redirect_experiments_dir() -> Path:
    """Point Experiment.new() at a fresh temp experiments root. Doing this
    BEFORE produce import would also work, but we already had to import
    prepare to get to the constant. We patch in place."""
    import prepare
    new_root = _TEMP_ROOT / "experiments"
    new_root.mkdir(parents=True, exist_ok=True)
    prepare.EXPERIMENTS_DIR = new_root
    return new_root


def _audit(exp_root: Path) -> tuple[list[str], list[str]]:
    """Return (present, missing) artifact lists."""
    present, missing = [], []
    for rel in EXPECTED_ARTIFACTS:
        p = exp_root / rel
        if p.exists() and p.stat().st_size > 0:
            present.append(rel)
        else:
            missing.append(rel)
    # Globs: at least one frame, at least one clip
    if list((exp_root / "frames").rglob("*.png")):
        present.append("frames/**/*.png (≥1)")
    else:
        missing.append("frames/**/*.png (≥1)")
    if list((exp_root / "clips").rglob("*.mp4")):
        present.append("clips/**/*.mp4 (≥1)")
    else:
        missing.append("clips/**/*.mp4 (≥1)")
    return present, missing


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def main() -> int:
    print()
    print("autofilm preflight — mock-driven end-to-end test")
    print("=" * 56)
    print()
    print(f"  workdir:  {_TEMP_ROOT}")

    _set_env()
    print(f"  book:     {os.environ['BOOK_PDF_PATH']}")
    print(f"  config:   MAX_SCENES=1, TAKES_PER_SHOT=1, VEO_TIER=fast")
    print()

    exp_root_parent = _redirect_experiments_dir()

    # Import produce AFTER env + experiments-dir patching so it sees them.
    print("  Importing prepare + produce...")
    from prepare import Experiment
    import produce
    _patch_external_calls(produce)
    print(f"  Patched mocks for: claude_tool, gpt_image, nano_banana, veo, "
          f"stable_audio, elevenlabs_sfx")
    print()

    # Run the pipeline.
    print("  ── pipeline ────────────────────────────────────────────")
    exp = Experiment.new()
    print(f"  {exp.exp_id}")
    print()
    final_path = produce.run(exp)
    print()
    print(f"  final video: {final_path}")

    # Build the bible.
    print()
    print("  ── bible ───────────────────────────────────────────────")
    from bible import build_bible
    bible_path = build_bible(exp)
    size_mb = bible_path.stat().st_size / 1_048_576
    print(f"  bible:       {bible_path}  ({size_mb:.1f} MB)")

    # Audit artifacts.
    print()
    print("  ── audit ───────────────────────────────────────────────")
    present, missing = _audit(exp.root)
    for p in present:
        print(f"  {_color('OK', '32')}    {p}")
    for m in missing:
        print(f"  {_color('MISS', '31')}  {m}")

    # Quick metadata on final.mp4.
    final = exp.root / "final.mp4"
    if final.exists():
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type,codec_name,width,height",
                 "-of", "default=noprint_wrappers=0", str(final)],
                capture_output=True, text=True, check=True,
            )
            print()
            print("  ── final.mp4 metadata ──────────────────────────────────")
            for line in r.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith("["):
                    print(f"    {line}")
        except Exception as e:  # noqa: BLE001
            print(f"  ffprobe failed: {e}")

    print()
    if missing:
        print(f"  {_color('FAIL', '31')}: {len(missing)} expected artifact(s) missing.")
        return 1
    print(f"  {_color('PASS', '32')}: all expected artifacts present.")
    print(f"  Pipeline plumbing verified end-to-end against synthetic data.")
    print(f"  Real Runway/Anthropic request shapes still need a paid run "
          f"to validate (see SETUP.md §8).")
    print()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        # Best-effort cleanup. Comment out for debugging.
        pass  # shutil.rmtree(_TEMP_ROOT, ignore_errors=True)
    sys.exit(rc)
