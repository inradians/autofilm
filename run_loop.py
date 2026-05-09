"""run_loop.py — the autoresearch loop for autofilm.

Inspired by Karpathy's autoresearch and Shopify's autoresearch engineering
blog: an LLM-driven feedback loop where each iteration's critique becomes
the next iteration's plan.

Pipeline per iteration:
  1. Run produce.py (full Stage 1-10) → final.mp4
  2. Run evaluate_film() → metric.json + critique.md (per-axis scores +
     prose critique + structured `changes` array)
  3. Build production_bible.json (machine-readable artifact manifest)
  4. Build bible.pdf (human-readable production bible)
  5. plan_carryover(metric) → carryover plan (what to invalidate)
  6. Experiment.new_iteration(prev, carryover) → new exp dir with the
     non-invalidated artifacts copied forward
  7. Loop.

The pipeline's existing per-stage `if exp.has(...)` checks make this
work with no changes to produce.py: any artifact present in the new
exp dir is reused; missing artifacts are regenerated.

Stop conditions (any one ends the loop):
  - Reached --iterations N
  - film_loss <= --target (default 0.15)
  - Improvement plateaued: |prev_loss - new_loss| < --plateau (default 0.01)
    over --plateau-window iterations (default 2)

Usage
-----
    # First run (creates exp_001), then iterate up to 5 times:
    python run_loop.py --iterations 5

    # Continue from latest exp (skip produce on first iter if final.mp4 exists):
    python run_loop.py --iterations 3 --resume

    # Stricter critic threshold (only apply HIGH-priority changes per iter):
    python run_loop.py --iterations 5 --threshold high

    # Stop early when film_loss drops below 0.10:
    python run_loop.py --iterations 10 --target 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from prepare import Experiment, evaluate_film
from carryover import plan_carryover, plan_summary
from production_bible import build_production_bible_json


def _print_banner(iteration: int, total: int, exp_id: str) -> None:
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  ITERATION {iteration}/{total}   exp = {exp_id}")
    print(f"{bar}\n")


def _run_produce(exp: Experiment) -> Path:
    """Invoke produce.py's run() against the given experiment."""
    from produce import run as produce_run
    return produce_run(exp)


def _maybe_build_pdf_bible(exp: Experiment) -> None:
    try:
        from bible import build_bible
        path = build_bible(exp)
        size_mb = path.stat().st_size / 1_048_576
        print(f"    bible.pdf: {path}  ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"    bible.pdf: failed (non-fatal) — {e}")


def run_loop(
    *,
    iterations: int,
    target_loss: float,
    threshold: str,
    plateau_eps: float,
    plateau_window: int,
    resume: bool,
) -> int:
    """Run the autoresearch loop. Returns the count of iterations completed."""
    history: list[dict[str, Any]] = []   # per-iter: {exp_id, film_loss, scores}
    prev_exp: Experiment | None = None

    if resume:
        try:
            prev_exp = Experiment.latest()
            print(f"Resuming from latest experiment: "
                  f"{prev_exp.book_slug}/{prev_exp.exp_id}")
        except Exception:
            print("No prior experiments found; starting fresh.")
            prev_exp = None

    for i in range(1, iterations + 1):
        # ── Step 1: pick / create the experiment for this iteration ───
        if prev_exp is None:
            exp = Experiment.new_or_resume()
        else:
            # If prev_exp has no metric yet (resume after partial), evaluate
            # it first instead of creating a new one.
            if not prev_exp.has("metric.json") and prev_exp.has("final.mp4"):
                exp = prev_exp
            elif not prev_exp.has("final.mp4"):
                # Resume same exp — final never finished.
                exp = prev_exp
            else:
                # Fresh new iteration with carryover from prev.
                metric = prev_exp.read_json("metric.json")
                plan   = plan_carryover(metric, priority_threshold=threshold)
                print(f"\n  Carryover plan: {plan_summary(plan)}")
                if plan.get("manual_review"):
                    print(f"  ({len(plan['manual_review'])} change(s) flagged "
                          f"for manual review — not auto-applied)")
                exp = Experiment.new_iteration(prev_exp, carryover=plan)

        _print_banner(i, iterations, f"{exp.book_slug}/{exp.exp_id}")

        # ── Step 2: produce ───────────────────────────────────────────
        if exp.has("final.mp4"):
            print(f"  final.mp4 exists for {exp.exp_id} — skipping produce.")
        else:
            t0 = time.time()
            try:
                _run_produce(exp)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"\n  ✗ produce.py failed for {exp.exp_id}: {e}")
                print(f"  Aborting loop after {i-1} successful iteration(s).")
                return i - 1
            print(f"  produce: {time.time() - t0:.0f}s")

        # ── Step 3: evaluate ──────────────────────────────────────────
        if exp.has("metric.json"):
            print(f"  metric.json exists — skipping critic.")
            metric = exp.read_json("metric.json")
        else:
            t0 = time.time()
            try:
                metric = evaluate_film(exp)
                print(f"  evaluate: {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"\n  ✗ evaluate_film failed for {exp.exp_id}: {e}")
                print(f"  Aborting loop after {i-1} successful iteration(s).")
                return i - 1

        film_loss = metric.get("film_loss", float("inf"))
        scores    = metric.get("scores", {})
        history.append({
            "iteration": i,
            "exp_id":    f"{exp.book_slug}/{exp.exp_id}",
            "film_loss": film_loss,
            "scores":    scores,
        })

        # ── Step 4: build production_bible.json + bible.pdf ───────────
        try:
            pb_path = build_production_bible_json(exp)
            print(f"  production_bible.json: {pb_path.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  production_bible.json failed (non-fatal): {e}")
        _maybe_build_pdf_bible(exp)

        # ── Step 5: print scores ──────────────────────────────────────
        print(f"\n  film_loss = {film_loss:.4f}")
        for axis, score in scores.items():
            weight = metric.get("weights", {}).get(axis, 0.0)
            print(f"    {axis:>14}: {score:.3f}  (weight {weight:.2f})")

        # ── Step 6: stop conditions ───────────────────────────────────
        if film_loss <= target_loss:
            print(f"\n  ✓ Target film_loss ≤ {target_loss} reached at iter {i}. Stopping.")
            return i

        if (
            len(history) >= plateau_window + 1
            and all(
                abs(history[-k - 1]["film_loss"] - history[-k]["film_loss"]) < plateau_eps
                for k in range(1, plateau_window + 1)
            )
        ):
            print(f"\n  ✓ film_loss plateaued (Δ < {plateau_eps} over "
                  f"{plateau_window} iter(s)). Stopping.")
            return i

        prev_exp = exp

    print(f"\n  Reached {iterations}/{iterations} iterations.")
    return iterations


def _print_history(out_path: Path | None = None) -> None:
    """Print a summary table of all experiments in this book that have
    metric.json. Optionally write the same data as JSON."""
    from prepare import iter_all_experiments
    rows = []
    for exp_path in iter_all_experiments():
        if (exp_path / "metric.json").exists():
            metric = json.loads((exp_path / "metric.json").read_text())
            book   = (exp_path.parent.name)
            rows.append({
                "exp":       f"{book}/{exp_path.name}",
                "film_loss": metric.get("film_loss", 0.0),
                "scores":    metric.get("scores", {}),
            })
    rows.sort(key=lambda r: r["exp"])
    if not rows:
        print("(no scored experiments)")
        return
    print(f"\n{'experiment':<32} {'film_loss':>10}   "
          f"{'cinema':>6} {'color':>6} {'sound':>6} {'acting':>6} "
          f"{'cont':>6} {'fidel':>6}")
    print("─" * 92)
    for r in rows:
        s = r["scores"]
        print(
            f"{r['exp']:<32} {r['film_loss']:>10.4f}   "
            f"{s.get('cinematography', 0):>6.3f} {s.get('color', 0):>6.3f} "
            f"{s.get('sound', 0):>6.3f} {s.get('acting', 0):>6.3f} "
            f"{s.get('continuity', 0):>6.3f} {s.get('fidelity', 0):>6.3f}"
        )
    if out_path:
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Autoresearch loop: produce → evaluate → carryover → produce"
    )
    ap.add_argument("--iterations", type=int, default=3,
                    help="Maximum iterations to run (default 3)")
    ap.add_argument("--target", type=float, default=0.15,
                    help="Stop when film_loss <= this (default 0.15)")
    ap.add_argument("--threshold", default="medium",
                    choices=["low", "medium", "high"],
                    help="Apply changes at or above this priority (default medium)")
    ap.add_argument("--plateau", type=float, default=0.01,
                    help="Stop if film_loss change < this for plateau-window iters (default 0.01)")
    ap.add_argument("--plateau-window", type=int, default=2,
                    help="Window for plateau detection (default 2)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from the latest experiment")
    ap.add_argument("--history", action="store_true",
                    help="Print history of all scored experiments and exit")
    args = ap.parse_args()

    if args.history:
        _print_history()
        return

    completed = run_loop(
        iterations=args.iterations,
        target_loss=args.target,
        threshold=args.threshold,
        plateau_eps=args.plateau,
        plateau_window=args.plateau_window,
        resume=args.resume,
    )

    print(f"\n{'═' * 72}")
    print(f"  Loop completed: {completed}/{args.iterations} iteration(s)")
    print(f"{'═' * 72}")
    _print_history()


if __name__ == "__main__":
    main()
