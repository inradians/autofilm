"""evaluate.py — score a finished experiment's film.

Usage:
    python evaluate.py exp_001         # score one experiment
    python evaluate.py latest          # score the most recent experiment
    python evaluate.py --all           # score every experiment that has a final.mp4 but no metric.json
"""
from __future__ import annotations

import argparse
import json
import sys

from prepare import EXPERIMENTS_DIR, Experiment, evaluate_film


def _score(exp_id: str) -> dict:
    exp = Experiment.load(exp_id)
    if not exp.path("final.mp4").exists():
        print(f"  {exp_id}: no final.mp4, skipping")
        return {}
    if exp.has("metric.json"):
        print(f"  {exp_id}: metric.json exists; loading")
        return exp.read_json("metric.json")
    print(f"  {exp_id}: evaluating film...")
    metric = evaluate_film(exp)
    # Refresh the bible now that the critique section can be filled in.
    try:
        from bible import build_bible
        bible_path = build_bible(exp)
        size_mb = bible_path.stat().st_size / 1_048_576
        print(f"    refreshed bible: {bible_path.name}  ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"    bible refresh failed (non-fatal): {e}")
    return metric


def _print_summary(exp_id: str, metric: dict) -> None:
    if not metric:
        return
    print(f"\n=== {exp_id} ===")
    print(f"film_loss = {metric['film_loss']:.4f}")
    for axis, score in metric["scores"].items():
        weight = metric["weights"][axis]
        print(f"  {axis:>14}: {score:.3f}  (weight {weight:.2f})")
    n_changes = len(metric.get("changes", []))
    high = sum(1 for c in metric.get("changes", []) if c.get("priority") == "high")
    print(f"  → {n_changes} suggested changes ({high} high-priority)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", nargs="?", help="experiment id, or 'latest', or '--all'")
    parser.add_argument("--all", action="store_true",
                        help="evaluate all experiments missing a metric.json")
    args = parser.parse_args()

    if args.all or args.exp_id == "--all":
        targets = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
    elif args.exp_id == "latest" or args.exp_id is None:
        targets = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
        if not targets:
            print("No experiments found.")
            return 1
        targets = [targets[-1]]
    else:
        targets = [args.exp_id]

    results = []
    for t in targets:
        try:
            metric = _score(t)
            if metric:
                results.append((t, metric))
                _print_summary(t, metric)
        except Exception as e:
            print(f"  {t}: evaluation failed: {e}")

    if len(results) > 1:
        print("\n=== Leaderboard (lower film_loss is better) ===")
        results.sort(key=lambda r: r[1]["film_loss"])
        for exp_id, m in results:
            print(f"  {exp_id}: {m['film_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
