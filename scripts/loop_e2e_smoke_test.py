#!/usr/bin/env python3
"""End-to-end smoke test for the autoresearch loop.

Goal: exercise the FULL run_loop.py path with real APIs, but at the
smallest possible scope — one scene, one shot, one moodboard, one take —
so we can verify the loop:

  1. Parses a book → script
  2. Casts + locates → moodboard
  3. Builds a 1-shot storyboard
  4. Renders 1 first-frame still
  5. Generates 1 short video take
  6. Compiles to final.mp4 (with mixing fixes applied)
  7. Runs evaluate_film via Gemini → metric.json
  8. Carryover translates critic changes → invalidation flags
  9. Spawns a child experiment (exp_002) and starts iteration 2

…all in a separate experiments directory so production runs are
untouched. Each iteration's artifacts live in their own
exp_NNN/ subdir under the test root, just like real runs.

Cost: roughly $0.50–$1.00 per iteration, depending on which models
the cascades resolve to. Total $1–$2 for the default 2-iteration
test. Failures cascade through cheaper alternatives, so a single
provider outage doesn't block the run.

Usage:
    python scripts/loop_e2e_smoke_test.py
    python scripts/loop_e2e_smoke_test.py --iterations 2 --keep
    python scripts/loop_e2e_smoke_test.py --keep   # don't auto-cleanup

Environment:
    Reads the same API keys as the real loop — ANTHROPIC_API_KEY,
    GOOGLE_AI_API_KEY (mandatory critic), RUNWAYML_API_SECRET, etc.
    Without GOOGLE_AI_API_KEY the loop cannot score the film and
    will abort after iteration 1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Tiny test PDF generation ────────────────────────────────────────────────

def make_test_book(out_dir: Path) -> Path:
    """Generate a minimal one-scene book PDF the loop can parse.

    Kept short on purpose: parse_script chunks pages, so a tiny book
    means one chunk → fast parse → low Claude cost. The content is
    structured to map cleanly to one scene with one beat.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("ERROR: reportlab is required (it's already a project dep). "
              "Run `pip install reportlab` if missing.", file=sys.stderr)
        sys.exit(2)

    pdf_path = out_dir / "the_lighthouse_keeper.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Title page
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 1.5 * 72,
                        "The Lighthouse Keeper")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 2.0 * 72, "A Short Tale")
    c.showPage()

    # Single scene of prose. ~150 words is enough for parse_script to
    # find one scene with one or two characters and one location.
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 72, "Chapter One")
    c.setFont("Helvetica", 11)
    text = c.beginText(72, height - 100)
    text.setLeading(15)
    paragraph = (
        "On the cliff above the cold North Sea, Maren tended the lamp "
        "alone. The lighthouse had been her father's, and his father's "
        "before that, but tonight the storm came harder than any she "
        "could remember. She climbed the iron stair with the old brass "
        "key in her teeth, the spiral of stone steps slick with damp. "
        "At the top, the great lens turned in its slow circle, throwing "
        "white fire into the rain. Maren wiped salt from the glass and "
        "thought of her father's last words. \"The light is the only "
        "true thing,\" he had said. \"Keep it burning.\" She nodded to "
        "the empty room and set her hand on the wheel. Far out at sea, "
        "a small boat caught the beam and turned at last toward shore."
    )
    # Word-wrap manually — reportlab's text object doesn't auto-wrap.
    line: list[str] = []
    for word in paragraph.split():
        line.append(word)
        if len(" ".join(line)) > 80:
            text.textLine(" ".join(line))
            line = []
    if line:
        text.textLine(" ".join(line))
    c.drawText(text)
    c.showPage()
    c.save()
    return pdf_path


# ── Verification helpers ────────────────────────────────────────────────────

REQUIRED_PER_ITERATION = [
    "script.json",
    "cast.json",
    "locations.json",
    "lookbook.json",
    "storyboard.json",
    "final.mp4",
    "metric.json",
    "critique.md",
]


def verify_iteration(exp_dir: Path) -> tuple[bool, list[str]]:
    """Confirm every artifact the loop should have produced is present."""
    missing = [f for f in REQUIRED_PER_ITERATION if not (exp_dir / f).exists()]
    return (not missing), missing


def summarize_iteration(exp_dir: Path) -> dict:
    """Pull headline numbers out of the iteration's outputs."""
    summary: dict = {"name": exp_dir.name}
    metric_path = exp_dir / "metric.json"
    if metric_path.exists():
        m = json.loads(metric_path.read_text())
        summary["film_loss"] = round(m.get("film_loss", -1.0), 4)
        summary["n_changes"] = len(m.get("changes", []))
        summary["scores"]    = {k: round(v, 3)
                                for k, v in m.get("scores", {}).items()}
    final = exp_dir / "final.mp4"
    if final.exists():
        summary["final_mp4_bytes"] = final.stat().st_size
    sb = exp_dir / "storyboard.json"
    if sb.exists():
        story = json.loads(sb.read_text())
        summary["n_shots"] = sum(len(v) for v in story.values())
    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=2,
                    help="Iterations to run (default 2 — enough to verify "
                         "the loop spawns a child experiment).")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete the test directory on success.")
    ap.add_argument("--existing-pdf", type=Path,
                    help="Reuse an existing PDF instead of generating one.")
    args = ap.parse_args()

    print("════════════════════════════════════════════════════════════")
    print("  autofilm — run-loop end-to-end smoke test")
    print("════════════════════════════════════════════════════════════")
    print()

    # Isolated work directory — production experiments dir untouched.
    work = Path(tempfile.mkdtemp(prefix="autofilm_e2e_"))
    print(f"Test root: {work}")

    pdf = args.existing_pdf or make_test_book(work)
    print(f"Test book: {pdf.name}  ({pdf.stat().st_size:,} bytes)")
    print()

    # Constrain the loop to 1-of-everything via env. The loop itself
    # is not modified — we just lean on the existing knobs.
    env = os.environ.copy()
    env.update({
        "BOOK_PDF_PATH":            str(pdf),
        "AUTOFILM_EXPERIMENTS_DIR": str(work / "experiments"),
        "MAX_SCENES":               "1",
        "MAX_SHOTS_PER_SCENE":      "1",
        "TAKES_PER_SHOT":           "1",
        # Disable optional ambient SFX bed — saves a Runway call.
        "AMBIENT_SFX_ENABLED":      "0",
        # Force a fresh start (don't pick up any pre-existing exp).
        "FORCE_NEW":                "1",
    })

    # Run the loop. --target is set absurdly low so the loop never
    # decides "we hit the goal, stop early" — we want to confirm
    # iteration 2 actually starts.
    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_loop.py"),
        "--iterations", str(args.iterations),
        "--target",     "0.001",
        "--threshold",  "medium",
    ]
    print(f"$ {' '.join(cmd)}")
    print(f"  (env: MAX_SCENES=1, MAX_SHOTS_PER_SCENE=1, TAKES_PER_SHOT=1)")
    print()

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        return 130
    elapsed = time.time() - t0
    print()
    print(f"  loop exited in {elapsed:.0f}s with code {proc.returncode}")
    print()

    # ── Verify ──────────────────────────────────────────────────────────────
    exp_root = work / "experiments"
    if not exp_root.exists():
        print(f"FAIL: experiments dir was never created at {exp_root}")
        return 1

    # The book slug comes from the PDF stem (per _book_slug). Find it.
    book_dirs = [d for d in exp_root.iterdir() if d.is_dir()]
    if not book_dirs:
        print(f"FAIL: no book directory under {exp_root}")
        return 1
    book_root = book_dirs[0]
    print(f"Book directory: {book_root.name}/")

    # Collect every exp_NNN/ that was created. Sorted so exp_001 first.
    iter_dirs = sorted(
        d for d in book_root.iterdir() if d.is_dir() and d.name.startswith("exp_")
    )
    if not iter_dirs:
        print(f"FAIL: no exp_NNN/ directories under {book_root}")
        return 1

    print()
    print("Iteration artifacts:")
    print()
    all_pass = True
    summaries = []
    for d in iter_dirs:
        ok, missing = verify_iteration(d)
        status = "✓" if ok else "✗"
        print(f"  {status} {d.name}/")
        if not ok:
            all_pass = False
            print(f"      missing: {', '.join(missing)}")
        s = summarize_iteration(d)
        if "film_loss" in s:
            print(f"      film_loss      = {s['film_loss']}")
            print(f"      n_changes      = {s['n_changes']}")
            print(f"      n_shots        = {s.get('n_shots', '—')}")
            print(f"      final_mp4      = {s.get('final_mp4_bytes', 0):,} bytes")
        summaries.append(s)
        print()

    # Cross-iteration check: exp_002 should reference exp_001 as parent
    if len(iter_dirs) >= 2:
        bible_path = iter_dirs[1] / "production_bible.json"
        parent_ok = False
        if bible_path.exists():
            try:
                bib = json.loads(bible_path.read_text())
                parent_exp = bib.get("parent_exp") or ""
                # production_bible.py writes parent_exp as the qualified
                # form '<book_slug>/exp_NNN' — accept either that or the
                # bare 'exp_NNN' so this test is robust to either schema.
                expected_bare      = iter_dirs[0].name
                expected_qualified = f"{book_root.name}/{expected_bare}"
                if parent_exp in (expected_bare, expected_qualified):
                    parent_ok = True
                    print(f"  ✓ {iter_dirs[1].name} → parent_exp = "
                          f"{parent_exp!r} (carryover applied)")
                else:
                    print(f"  ✗ {iter_dirs[1].name} → parent_exp = "
                          f"{parent_exp!r}, expected "
                          f"{expected_bare!r} or {expected_qualified!r}")
            except Exception as e:                                 # noqa: BLE001
                print(f"  ⚠ couldn't parse {iter_dirs[1].name}/production_bible.json: {e}")
        else:
            print(f"  ⚠ {iter_dirs[1].name}/production_bible.json missing — "
                  f"can't verify parent linkage")
        if not parent_ok and proc.returncode == 0:
            all_pass = False

    print()
    print("════════════════════════════════════════════════════════════")
    if all_pass and proc.returncode == 0:
        print("  ✓ E2E SMOKE TEST PASSED")
        print(f"  {len(iter_dirs)} iteration(s) completed in {elapsed:.0f}s")
    else:
        print("  ✗ E2E SMOKE TEST FAILED")
        if proc.returncode != 0:
            print(f"  run_loop.py exited with code {proc.returncode}")
    print("════════════════════════════════════════════════════════════")

    if args.keep:
        print()
        print(f"Artifacts kept at: {work}")
        print(f"  Inspect with:    ls -la {book_root}/exp_*/")
    else:
        # Even on success, leave the dir IF anything failed — gives the
        # user something to inspect. On full success we tidy up.
        if all_pass and proc.returncode == 0:
            shutil.rmtree(work, ignore_errors=True)
            print()
            print("(Cleaned up. Use --keep to preserve artifacts.)")
        else:
            print()
            print(f"Artifacts left for inspection at: {work}")

    return 0 if (all_pass and proc.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
