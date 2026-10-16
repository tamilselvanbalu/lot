#!/usr/bin/env python3
"""
Walk-forward check: do actual winning draws mix hot / mild / cold tiers?

For each draw, tiers are computed from ALL prior draws only (no future leakage).
Compares simple frequency tiers (hybrid filter) and decay-weighted tiers (enhanced rank).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_predictions import classify_tiers, count_tiers, frequency_tables, load_draws
from prediction_engine import weighted_counts

MAIN_RANGE = range(1, 51)
STAR_RANGE = range(1, 13)


def tier_mix_label(mix: Counter[str]) -> str:
    parts = []
    if mix.get("hot", 0):
        parts.append(f"{mix['hot']}H")
    if mix.get("mild", 0):
        parts.append(f"{mix['mild']}M")
    if mix.get("cold", 0):
        parts.append(f"{mix['cold']}C")
    return "+".join(parts) if parts else "?"


def analyze_walk_forward(
    draws_chrono: list[dict],
    min_prior: int,
    decay: float | None,
) -> list[dict]:
    rows: list[dict] = []
    for i, draw in enumerate(draws_chrono):
        prior = draws_chrono[:i]
        if len(prior) < min_prior:
            continue
        if decay is not None:
            main_c, star_c = weighted_counts(list(reversed(prior)), decay)
            main_c = Counter({k: int(v) for k, v in main_c.items()})
            star_c = Counter({k: int(v) for k, v in star_c.items()})
            method = f"weighted_decay_{decay}"
        else:
            main_c, star_c = frequency_tables(prior)
            method = "simple_frequency"

        main_tiers = classify_tiers(main_c, MAIN_RANGE)
        star_tiers = classify_tiers(star_c, STAR_RANGE)
        main_mix = count_tiers(draw["numbers"], main_tiers)
        star_mix = count_tiers(draw["stars"], star_tiers)

        rows.append(
            {
                "date": draw["date"],
                "method": method,
                "main_mix": dict(main_mix),
                "star_mix": dict(star_mix),
                "main_all_three": all(main_mix.get(t, 0) >= 1 for t in ("hot", "mild", "cold")),
                "main_has_cold": main_mix.get("cold", 0) >= 1,
                "star_all_three": all(star_mix.get(t, 0) >= 1 for t in ("hot", "mild", "cold")),
                "prior_draws": len(prior),
            }
        )
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    if not n:
        return {"label": label, "draws": 0}

    main_all_three = sum(1 for r in rows if r["main_all_three"])
    main_has_cold = sum(1 for r in rows if r["main_has_cold"])
    star_all_three = sum(1 for r in rows if r["star_all_three"])

    main_patterns: Counter[str] = Counter()
    for r in rows:
        main_patterns[tier_mix_label(Counter(r["main_mix"]))] += 1

    fusion_style = sum(
        1
        for r in rows
        if r["main_has_cold"]
        and r["main_mix"].get("hot", 0) <= 3
        and r["main_mix"].get("hot", 0) + r["main_mix"].get("mild", 0) >= 3
    )

    return {
        "label": label,
        "draws": n,
        "main_all_three_pct": round(100 * main_all_three / n, 1),
        "main_has_cold_pct": round(100 * main_has_cold / n, 1),
        "star_all_three_pct": round(100 * star_all_three / n, 1),
        "fusion_style_main_pct": round(100 * fusion_style / n, 1),
        "top_main_patterns": main_patterns.most_common(8),
    }


def print_report(simple: dict, weighted: dict) -> None:
    print("=" * 64)
    print("HOT / MILD / COLD MIX IN REAL DRAWS (walk-forward)")
    print("=" * 64)
    print(
        "\nFor each draw, tiers use only earlier history — same idea as playing "
        "'the future' without peeking.\n"
    )
    for s in (simple, weighted):
        print(f"--- {s['label']} ({s['draws']} draws) ---")
        if not s["draws"]:
            print("  (no data)\n")
            continue
        print(f"  Main numbers include ALL three tiers (H+M+C): {s['main_all_three_pct']}%")
        print(f"  Main numbers include at least 1 COLD:           {s['main_has_cold_pct']}%")
        print(f"  Main matches fusion-style mix (≥1 cold, ≤3 hot, ≥3 hot+mild): {s['fusion_style_main_pct']}%")
        print(f"  Stars include ALL three tiers (2 balls — rare): {s['star_all_three_pct']}%")
        print("  Most common main tier patterns (5 balls):")
        for pat, cnt in s["top_main_patterns"]:
            pct = 100 * cnt / s["draws"]
            print(f"    {pat:12s}  {cnt:4d}  ({pct:.1f}%)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hot/mild/cold mix in historical wins")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "euromillions_last_5_years.json",
    )
    parser.add_argument(
        "--min-prior",
        type=int,
        default=52,
        help="Minimum prior draws before scoring a draw (default ~1 year)",
    )
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("-o", type=Path, default=None, help="Save JSON summary")
    args = parser.parse_args()

    draws = load_draws(args.input)
    chrono = sorted(draws, key=lambda d: d["date"])

    simple_rows = analyze_walk_forward(chrono, args.min_prior, decay=None)
    weighted_rows = analyze_walk_forward(chrono, args.min_prior, decay=args.decay)

    simple = summarize(simple_rows, "Simple frequency (hybrid cold/tier filter)")
    weighted = summarize(weighted_rows, f"Weighted decay {args.decay} (enhanced ranking)")

    print_report(simple, weighted)

    if args.o:
        out = {
            "min_prior_draws": args.min_prior,
            "simple_summary": simple,
            "weighted_summary": weighted,
            "sample_recent": simple_rows[-5:],
        }
        args.o.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.o}")


if __name__ == "__main__":
    main()
