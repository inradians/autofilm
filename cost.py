"""cost.py — estimate the production cost of an experiment.

Every API call we make has an approximate USD cost. This module:
  1. Maintains a lookup table of per-model rates (per-call or per-second).
  2. Walks an experiment's prompts.json (logged by exp.log_prompt) and
     adds an estimated USD cost to each entry.
  3. Aggregates totals + per-model breakdown for display in the UI.

Rates are approximate and based on public pricing pages as of 2026-05.
They're rounded to 2-3 sig figs. Treat the totals as estimates to within
±20%, not invoices. Real billing comes from the providers.

Usage
-----
    from cost import aggregate_costs
    summary = aggregate_costs(exp)
    # {"total_usd": 4.23, "by_model": {"veo": 1.62, "gpt_image": 0.40, ...},
    #   "by_stage": {...}, "items": [{model, target, cost_usd, ...}]}

CLI
---
    python cost.py exp_001
    python cost.py latest
    python cost.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Per-model rate table ──────────────────────────────────────────────────────
#
# Each entry is {kind, rate}:
#   kind = "per_call"   → rate is USD per generation
#   kind = "per_second" → rate is USD per second of audio/video duration
#
# Models with multiple tiers (gpt_image, ltx-2-3-pro/fast, veo low/standard,
# stable_image core/sd3/ultra) use the most-common tier as the default rate.

RATES: dict[str, dict[str, Any]] = {
    # Image — Runway-relayed
    "gpt_image":          {"kind": "per_call", "rate": 0.020},
    "gen4_image":         {"kind": "per_call", "rate": 0.050},
    "gen4_image_turbo":   {"kind": "per_call", "rate": 0.020},
    "nano_banana":        {"kind": "per_call", "rate": 0.020},
    # Image — direct providers
    "openai_image":          {"kind": "per_call", "rate": 0.040},
    "google_nano_banana":    {"kind": "per_call", "rate": 0.040},
    "reve_create":           {"kind": "per_call", "rate": 0.010},
    "reve_remix":            {"kind": "per_call", "rate": 0.010},
    "flux-pro":              {"kind": "per_call", "rate": 0.050},
    "flux-2-pro":            {"kind": "per_call", "rate": 0.060},
    "flux-pro-1.1":          {"kind": "per_call", "rate": 0.050},
    "flux-2-pro-preview":    {"kind": "per_call", "rate": 0.060},
    "stable_image":          {"kind": "per_call", "rate": 0.030},
    "stable-image-core":     {"kind": "per_call", "rate": 0.030},
    "stable-image-sd3":      {"kind": "per_call", "rate": 0.065},
    "stable-image-ultra":    {"kind": "per_call", "rate": 0.080},

    # Video — Runway-relayed
    "seedance":              {"kind": "per_second", "rate": 0.090},
    "veo":                   {"kind": "per_second", "rate": 0.080},
    "veo3.1":                {"kind": "per_second", "rate": 0.080},
    "veo-3.1-fast":          {"kind": "per_second", "rate": 0.080},
    # Video — direct providers
    "google_veo":            {"kind": "per_second", "rate": 0.025},
    "veo-3.1-generate-preview": {"kind": "per_second", "rate": 0.025},
    "ltx-2-3-pro":           {"kind": "per_second", "rate": 0.067},
    "ltx-2-3-fast":          {"kind": "per_second", "rate": 0.040},

    # Audio
    "stable-audio-2.5":      {"kind": "per_second", "rate": 0.0021},  # ~$0.10/47s
    "stable_audio":          {"kind": "per_second", "rate": 0.0021},
    "elevenlabs_sfx":        {"kind": "per_second", "rate": 0.0125},  # ~$0.05/4s
    "eleven_multilingual_v2":{"kind": "per_call",   "rate": 0.050},   # rough
    "runway_tts":            {"kind": "per_call",   "rate": 0.050},

    # Text — Anthropic / Claude. Token-based but charged per call here for
    # simplicity. The dominant cost in a film run is media, not text.
    "claude-opus-4-7":       {"kind": "per_call", "rate": 0.060},
    "claude-opus-4-6":       {"kind": "per_call", "rate": 0.050},
    "claude-sonnet-4-6":     {"kind": "per_call", "rate": 0.012},
    "claude-haiku-4-5-20251001": {"kind": "per_call", "rate": 0.003},

    # Critic — Gemini for video review
    "gemini-3.1-pro-preview":          {"kind": "per_call", "rate": 0.080},
}


def _normalize_model(model: str) -> str:
    """Lowercase + trim suffixes like '*' (rephrased fallback marker) and
    '+refs' (with-refs variant). NOTE: ``str.rstrip`` strips a *set* of
    characters, not a substring, so we use ``removesuffix`` here to avoid
    accidentally chewing through real model name characters
    (e.g. rstrip('+refs') would strip the 'e' off 'gen4_image')."""
    if not model:
        return ""
    m = model.strip()
    for suffix in ("+refs", "*"):
        if m.endswith(suffix):
            m = m[:-len(suffix)].strip()
    return m


def estimate_cost_for_entry(entry: dict[str, Any]) -> float | None:
    """USD cost for one prompts.json entry, or None if model is unknown."""
    raw_model = entry.get("model") or ""
    model = _normalize_model(raw_model)
    rate  = RATES.get(model)
    if rate is None:
        # Try the model-with-suffix in case rate matches a versioned form.
        for key, r in RATES.items():
            if model.startswith(key) or key.startswith(model):
                rate = r
                break
    if rate is None:
        return None

    if rate["kind"] == "per_call":
        return float(rate["rate"])
    if rate["kind"] == "per_second":
        secs = entry.get("duration_seconds") or entry.get("duration") or 0
        try:
            secs = float(secs)
        except (TypeError, ValueError):
            secs = 0.0
        return float(rate["rate"]) * secs
    return None


def aggregate_costs(exp_or_path: Any) -> dict[str, Any]:
    """Walk an experiment's prompts.json and return:
        {
            "total_usd":  float,
            "by_model":   {model: usd, ...},
            "by_stage":   {stage: usd, ...},
            "items":      [{model, target, stage, cost_usd, ...}, ...],
            "n_calls":    int,
            "n_unknown":  int,   # entries whose model isn't in RATES
        }

    Accepts either an Experiment object or a path to an exp dir or a
    path to a prompts.json file directly.
    """
    if isinstance(exp_or_path, (str, Path)):
        p = Path(exp_or_path)
        if p.is_dir():
            prompts_path = p / "prompts.json"
        else:
            prompts_path = p
    else:
        # Experiment instance
        prompts_path = exp_or_path.root / "prompts.json"

    if not prompts_path.exists():
        return {
            "total_usd": 0.0, "by_model": {}, "by_stage": {},
            "items": [], "n_calls": 0, "n_unknown": 0,
        }

    log = json.loads(prompts_path.read_text())
    items: list[dict[str, Any]] = []
    by_model: dict[str, float] = {}
    by_stage: dict[str, float] = {}
    total = 0.0
    n_unknown = 0

    # log is keyed by "target" → entry; flatten for accounting.
    if isinstance(log, dict):
        entries = list(log.values())
    elif isinstance(log, list):
        entries = log
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cost = estimate_cost_for_entry(entry)
        model = _normalize_model(entry.get("model") or "")
        stage = entry.get("stage") or "unknown"
        if cost is None:
            n_unknown += 1
            cost = 0.0   # don't poison totals; flag via n_unknown instead.
        total += cost
        by_model[model] = by_model.get(model, 0.0) + cost
        by_stage[stage] = by_stage.get(stage, 0.0) + cost
        items.append({
            "model":            model,
            "target":           entry.get("target"),
            "stage":            stage,
            "cost_usd":         round(cost, 4),
            "duration_seconds": entry.get("duration_seconds"),
        })

    # Sort the breakdowns for stable display.
    by_model = dict(sorted(by_model.items(), key=lambda kv: -kv[1]))
    by_stage = dict(sorted(by_stage.items(), key=lambda kv: -kv[1]))

    return {
        "total_usd":  round(total, 4),
        "by_model":   {m: round(v, 4) for m, v in by_model.items()},
        "by_stage":   {s: round(v, 4) for s, v in by_stage.items()},
        "items":      items,
        "n_calls":    len(items),
        "n_unknown":  n_unknown,
    }


def aggregate_chain(exp: Any) -> dict[str, Any]:
    """Sum costs across an experiment AND all of its ancestors via parent_exp.

    Useful for showing the cumulative cost of an iteration chain in the UI.
    Returns the same shape as aggregate_costs but with an extra
    ``per_exp`` list giving the breakdown by ancestor.
    """
    from prepare import Experiment  # late import to avoid cycle

    chain: list[Any] = []
    cur = exp
    seen: set[str] = set()
    while cur and cur.exp_id not in seen:
        seen.add(cur.exp_id)
        chain.append(cur)
        parent_id = cur.parent_exp_id
        if not parent_id:
            break
        try:
            cur = Experiment.load(f"{cur.book_slug}/{parent_id}")
        except Exception:
            break

    per_exp = []
    total = 0.0
    by_model_total: dict[str, float] = {}
    by_stage_total: dict[str, float] = {}
    for e in chain:
        c = aggregate_costs(e)
        per_exp.append({
            "exp_id":    f"{e.book_slug}/{e.exp_id}",
            "total_usd": c["total_usd"],
            "n_calls":   c["n_calls"],
        })
        total += c["total_usd"]
        for m, v in c["by_model"].items():
            by_model_total[m] = by_model_total.get(m, 0.0) + v
        for s, v in c["by_stage"].items():
            by_stage_total[s] = by_stage_total.get(s, 0.0) + v

    return {
        "total_usd":  round(total, 4),
        "by_model":   {m: round(v, 4) for m, v in
                       sorted(by_model_total.items(), key=lambda kv: -kv[1])},
        "by_stage":   {s: round(v, 4) for s, v in
                       sorted(by_stage_total.items(), key=lambda kv: -kv[1])},
        "per_exp":    per_exp,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _fmt_usd(v: float) -> str:
    if v >= 100:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:.4f}"


def _print_summary(label: str, summary: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  total: {_fmt_usd(summary['total_usd'])} ({summary['n_calls']} calls, "
          f"{summary.get('n_unknown', 0)} unknown-model)")
    print(f"\n  by model:")
    for m, v in summary["by_model"].items():
        print(f"    {m:<32} {_fmt_usd(v):>10}")
    print(f"\n  by stage:")
    for s, v in summary["by_stage"].items():
        print(f"    {s:<32} {_fmt_usd(v):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate the cost of an experiment.")
    ap.add_argument("exp_id", nargs="?", help="exp_NNN, book/exp_NNN, or 'latest'")
    ap.add_argument("--all", action="store_true",
                    help="Print summaries for all experiments")
    ap.add_argument("--chain", action="store_true",
                    help="Sum costs across the iteration chain (parent + ancestors)")
    args = ap.parse_args()

    from prepare import Experiment, iter_all_experiments

    if args.all:
        for exp_path in iter_all_experiments():
            try:
                exp = Experiment(exp_id=exp_path.name, root=exp_path)
                s = aggregate_costs(exp)
                _print_summary(f"{exp.book_slug}/{exp.exp_id}", s)
            except Exception as e:  # noqa: BLE001
                print(f"  {exp_path.name}: failed — {e}")
        return

    if not args.exp_id:
        ap.print_help()
        sys.exit(1)

    exp = Experiment.latest() if args.exp_id == "latest" else Experiment.load(args.exp_id)
    if args.chain:
        s = aggregate_chain(exp)
        _print_summary(f"{exp.book_slug}/{exp.exp_id} (chain)", s)
        print(f"\n  per exp:")
        for e in s["per_exp"]:
            print(f"    {e['exp_id']:<32} {_fmt_usd(e['total_usd']):>10}  "
                  f"({e['n_calls']} calls)")
    else:
        s = aggregate_costs(exp)
        _print_summary(f"{exp.book_slug}/{exp.exp_id}", s)


if __name__ == "__main__":
    main()
