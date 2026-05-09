"""evaluate.py — score a finished experiment's film.

Usage:
    python evaluate.py exp_001                  # score one experiment (searches all books)
    python evaluate.py jurassic_park/exp_001    # fully-qualified id
    python evaluate.py latest                   # score the most recent experiment
    python evaluate.py --all                    # score every experiment that has a final.mp4 but no metric.json
"""
from __future__ import annotations

import argparse
import json
import sys

from prepare import Experiment, evaluate_film, iter_all_experiments


def _score(exp_id: str) -> tuple[str, dict]:
    exp = Experiment.load(exp_id)
    # Use the experiment's own id for display (which may now be qualified
    # like "jurassic_park/exp_001" if the caller passed one) so the output
    # disambiguates same-numbered experiments across different books.
    display_id = f"{exp.book_slug}/{exp.exp_id}"
    if not exp.path("final.mp4").exists():
        print(f"  {display_id}: no final.mp4, skipping")
        return display_id, {}
    if exp.has("metric.json"):
        print(f"  {display_id}: metric.json exists; loading")
        return display_id, exp.read_json("metric.json")
    print(f"  {display_id}: evaluating film...")
    metric = evaluate_film(exp)
    # Refresh the bible now that the critique section can be filled in.
    try:
        from bible import build_bible
        bible_path = build_bible(exp)
        size_mb = bible_path.stat().st_size / 1_048_576
        print(f"    refreshed bible: {bible_path.name}  ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"    bible refresh failed (non-fatal): {e}")
    # Refresh the machine-readable production bible JSON so run_loop.py
    # can read up-to-date metric data when planning the next iteration.
    try:
        from production_bible import build_production_bible_json
        pb_path = build_production_bible_json(exp)
        print(f"    refreshed production_bible.json: {pb_path.name}")
    except Exception as e:  # noqa: BLE001
        print(f"    production_bible.json refresh failed (non-fatal): {e}")
    # Print suggested changes summary so the human knows what the next
    # iteration will try to fix.
    try:
        from carryover import plan_carryover, plan_summary
        plan = plan_carryover(metric)
        print(f"    next-iteration carryover plan: {plan_summary(plan)}")
    except Exception as e:  # noqa: BLE001
        print(f"    carryover plan failed (non-fatal): {e}")
    return display_id, metric


def _print_summary(display_id: str, metric: dict) -> None:
    if not metric:
        return
    print(f"\n=== {display_id} ===")
    print(f"film_loss = {metric['film_loss']:.4f}")
    for axis, score in metric["scores"].items():
        weight = metric["weights"][axis]
        print(f"  {axis:>14}: {score:.3f}  (weight {weight:.2f})")
    n_changes = len(metric.get("changes", []))
    high = sum(1 for c in metric.get("changes", []) if c.get("priority") == "high")
    print(f"  → {n_changes} suggested changes ({high} high-priority)")


def _all_targets() -> list[str]:
    """Return every experiment as a 'book_slug/exp_id' string for loading."""
    out = []
    for p in iter_all_experiments():
        # Reconstruct the qualified id: parent dir is the book slug,
        # unless this is an old flat-layout experiment under EXPERIMENTS_DIR.
        if p.parent.name == "experiments":
            out.append(p.name)  # old flat layout
        else:
            out.append(f"{p.parent.name}/{p.name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", nargs="?", help="experiment id, or 'latest', or '--all'")
    parser.add_argument("--all", action="store_true",
                        help="evaluate all experiments missing a metric.json")
    args = parser.parse_args()

    if args.all or args.exp_id == "--all":
        targets = _all_targets()
    elif args.exp_id == "latest" or args.exp_id is None:
        all_t = _all_targets()
        if not all_t:
            print("No experiments found.")
            return 1
        targets = [all_t[-1]]
    else:
        targets = [args.exp_id]

    results = []
    for t in targets:
        try:
            display_id, metric = _score(t)
            if metric:
                results.append((display_id, metric))
                _print_summary(display_id, metric)
        except Exception as e:
            print(f"  {t}: evaluation failed: {e}")

    if len(results) > 1:
        print("\n=== Leaderboard (lower film_loss is better) ===")
        results.sort(key=lambda r: r[1]["film_loss"])
        for display_id, m in results:
            print(f"  {display_id}: {m['film_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
