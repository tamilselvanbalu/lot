#!/usr/bin/env python3
"""
Generate EuroMillions line suggestions scored against historical statistical rules.

Uses sum band (70% rule), parity, high/low balance, hot-mild-cold mix, frequency,
overdue tracking, and delta patterns from draw history.

DISCLAIMER: Past patterns do not predict future draws. Each draw is independent.
Suggestions mirror historical distributions only; they do not increase odds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

MAIN_MIN, MAIN_MAX = 1, 50
STAR_MIN, STAR_MAX = 1, 12
MAIN_PICK, STAR_PICK = 5, 2
LOW_CUTOFF = 25  # 1-25 low, 26-50 high

# Empirical EuroMillions bands (calibrated from 5-year history; recomputed on load)
DEFAULT_SUM_BAND = (95, 160)
PREFERRED_ODD_COUNTS = {2, 3}
PREFERRED_LOW_COUNTS = {2, 3}


def load_draws(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["draws"]


def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * p / 100)))
    return sorted_vals[idx]


def compute_sum_band(draws: list[dict]) -> tuple[int, int, float]:
    sums = sorted(sum(d["numbers"]) for d in draws)
    lo, hi = percentile(sums, 15), percentile(sums, 85)
    in_band = sum(1 for s in sums if lo <= s <= hi)
    pct = 100 * in_band / len(sums) if sums else 0
    return lo, hi, pct


def window_draws(draws: list[dict], n: int) -> list[dict]:
    """Use full history when n is 0 or greater than available draws."""
    if n <= 0 or n >= len(draws):
        return draws
    return draws[:n]


def lookback_label(n: int, total: int) -> str:
    if n <= 0 or n >= total:
        return f"all {total}"
    return str(n)


def position_band_percentiles(band_pct: float) -> tuple[float, float]:
    """
    Convert a central band width to lower/upper percentile indices.
    e.g. 70 -> (15, 85), 90 -> (5, 95).
    """
    if not 0 < band_pct <= 100:
        raise ValueError("position band must be > 0 and <= 100")
    tail = (100 - band_pct) / 2
    return tail, 100 - tail


def analyze_sorted_positions(draws: list[dict], band_pct: float = 70) -> dict:
    """
    Per-position stats after sorting mains (pos 1 = smallest … pos 5 = largest).
    Matches how results are shown on TV.
    """
    lo_pct, hi_pct = position_band_percentiles(band_pct)
    main_cols: list[list[int]] = [[] for _ in range(MAIN_PICK)]
    star_cols: list[list[int]] = [[] for _ in range(STAR_PICK)]

    for d in draws:
        nums = sorted(d["numbers"])
        for i, n in enumerate(nums):
            main_cols[i].append(n)
        stars = sorted(d["stars"])
        for i, s in enumerate(stars):
            star_cols[i].append(s)

    def summarize(values: list[int], pos: int, label: str) -> dict:
        if not values:
            return {"position": pos, "label": label}
        sv = sorted(values)
        n = len(sv)
        c = Counter(sv)
        top3 = [{"value": v, "count": cnt} for v, cnt in c.most_common(3)]

        def pct(p: float) -> int:
            return sv[min(n - 1, max(0, int(n * p / 100)))]

        lo, hi = pct(lo_pct), pct(hi_pct)
        band = {"from": lo, "to": hi, "band_pct": band_pct, "percentile_lo": lo_pct, "percentile_hi": hi_pct}
        return {
            "position": pos,
            "label": label,
            "min": sv[0],
            "max": sv[-1],
            "mean": round(sum(sv) / n, 1),
            "range_band": band,
            "range_70_percent": band if band_pct == 70 else {"from": lo, "to": hi},
            "absolute_range_in_history": {"from": sv[0], "to": sv[-1]},
            "most_common": top3,
        }

    main_positions = [
        summarize(main_cols[i], i + 1, f"main_{i + 1}_smallest_to_largest")
        for i in range(MAIN_PICK)
    ]
    star_positions = [
        summarize(star_cols[i], i + 1, f"star_{i + 1}_smallest_to_largest")
        for i in range(STAR_PICK)
    ]
    return {
        "band_pct": band_pct,
        "percentile_lo": lo_pct,
        "percentile_hi": hi_pct,
        "note": (
            f"Position 1 = lowest main; position 5 = highest. "
            f"range_band is where ~{band_pct:.0f}% of historical values fell "
            f"({lo_pct:g}th–{hi_pct:g}th percentile)."
        ),
        "main_positions": main_positions,
        "lucky_star_positions": star_positions,
    }


def print_position_ranges(pos: dict) -> None:
    print("\n--- Sorted position ranges (from history) ---")
    print(f"  {pos['note']}\n")
    for p in pos["main_positions"]:
        r70 = p.get("range_band") or p.get("range_70_percent", {})
        ab = p["absolute_range_in_history"]
        common = ", ".join(f"{x['value']}({x['count']}×)" for x in p["most_common"])
        print(
            f"  Main pos {p['position']} ({p['label']}):  "
            f"typical {r70['from']}–{r70['to']}  |  "
            f"ever seen {ab['from']}–{ab['to']}  |  "
            f"mean {p['mean']}  |  top: {common}"
        )
    for p in pos["lucky_star_positions"]:
        r70 = p.get("range_band") or p.get("range_70_percent", {})
        ab = p["absolute_range_in_history"]
        common = ", ".join(f"{x['value']}({x['count']}×)" for x in p["most_common"])
        print(
            f"  Star pos {p['position']}:  "
            f"typical {r70['from']}–{r70['to']}  |  "
            f"ever seen {ab['from']}–{ab['to']}  |  "
            f"mean {p['mean']}  |  top: {common}"
        )


def position_limits_from_draws(draws: list[dict], band_pct: float = 70) -> dict | None:
    """
    Per sorted-slot bands used as fusion filters (central band_pct of history).
    Main position 1 = smallest main; star position 1 = smaller lucky star.
    """
    if not draws:
        return None
    analysis = analyze_sorted_positions(draws, band_pct=band_pct)
    main_limits: list[dict] = []
    for p in analysis["main_positions"]:
        r = p.get("range_band") or p.get("range_70_percent")
        if not r:
            return None
        main_limits.append(
            {
                "position": p["position"],
                "from": r["from"],
                "to": r["to"],
                "label": p.get("label", ""),
            }
        )
    star_limits: list[dict] = []
    for p in analysis["lucky_star_positions"]:
        r = p.get("range_band") or p.get("range_70_percent")
        if not r:
            return None
        star_limits.append({"position": p["position"], "from": r["from"], "to": r["to"]})
    return {
        "band_pct": band_pct,
        "percentile_lo": analysis["percentile_lo"],
        "percentile_hi": analysis["percentile_hi"],
        "main": main_limits,
        "stars": star_limits,
    }


def line_passes_position_ranges(
    main: list[int],
    stars: list[int],
    limits: dict | None,
) -> tuple[bool, list[dict]]:
    """True if each sorted main/star sits in its historical typical band."""
    if not limits:
        return True, []
    failures: list[dict] = []
    for i, slot in enumerate(sorted(main)):
        band = limits["main"][i]
        lo, hi = band["from"], band["to"]
        if not (lo <= slot <= hi):
            failures.append(
                {
                    "slot": "main",
                    "position": band["position"],
                    "number": slot,
                    "from": lo,
                    "to": hi,
                }
            )
    for i, slot in enumerate(sorted(stars)):
        band = limits["stars"][i]
        lo, hi = band["from"], band["to"]
        if not (lo <= slot <= hi):
            failures.append(
                {
                    "slot": "star",
                    "position": band["position"],
                    "number": slot,
                    "from": lo,
                    "to": hi,
                }
            )
    return len(failures) == 0, failures


def frequency_tables(draws: list[dict]) -> tuple[Counter[int], Counter[int]]:
    main_c: Counter[int] = Counter()
    star_c: Counter[int] = Counter()
    for d in draws:
        for x in d["numbers"]:
            main_c[x] += 1
        for x in d["stars"]:
            star_c[x] += 1
    return main_c, star_c


def draws_since_last_seen(draws: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    main_gap: dict[int, int] = {}
    star_gap: dict[int, int] = {}
    for idx, d in enumerate(draws):
        for n in d["numbers"]:
            if n not in main_gap:
                main_gap[n] = idx
        for s in d["stars"]:
            if s not in star_gap:
                star_gap[s] = idx
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        main_gap.setdefault(n, len(draws))
    for s in range(STAR_MIN, STAR_MAX + 1):
        star_gap.setdefault(s, len(draws))
    return main_gap, star_gap


def long_due_mains_on_line(
    main: list[int], main_gaps: dict[int, int], min_gap: int
) -> list[dict]:
    """Mains absent at least min_gap draws, sorted by gap descending."""
    rows = [
        {"number": n, "gap": main_gaps.get(n, 0)}
        for n in main
        if main_gaps.get(n, 0) >= min_gap
    ]
    rows.sort(key=lambda x: x["gap"], reverse=True)
    return rows


def line_meets_long_due(
    main: list[int],
    main_gaps: dict[int, int],
    min_gap: int,
    min_count: int,
) -> bool:
    return len(long_due_mains_on_line(main, main_gaps, min_gap)) >= min_count


def classify_tiers(counts: Counter[int], pool: range) -> dict[int, str]:
    ranked = sorted(((n, counts.get(n, 0)) for n in pool), key=lambda x: x[1], reverse=True)
    n = len(ranked)
    third = max(1, n // 3)
    tiers: dict[int, str] = {}
    for i, (num, _) in enumerate(ranked):
        if i < third:
            tiers[num] = "hot"
        elif i >= n - third:
            tiers[num] = "cold"
        else:
            tiers[num] = "mild"
    return tiers


def common_deltas(draws: list[dict], top_n: int = 8) -> list[tuple[int, ...]]:
    delta_counter: Counter[tuple[int, ...]] = Counter()
    for d in draws:
        sorted_nums = sorted(d["numbers"])
        deltas = tuple(sorted_nums[i] - sorted_nums[i - 1] for i in range(1, len(sorted_nums)))
        delta_counter[deltas] += 1
    return [d for d, _ in delta_counter.most_common(top_n)]


def score_number(
    n: int,
    main_counts: Counter[int],
    main_gaps: dict[int, int],
    tiers: dict[int, str],
    n_draws: int,
    is_star: bool = False,
) -> float:
    pool_size = STAR_MAX if is_star else MAIN_MAX
    picks_per_draw = STAR_PICK if is_star else MAIN_PICK
    expected = n_draws * picks_per_draw / pool_size if n_draws else 1
    freq = main_counts.get(n, 0)
    freq_ratio = freq / expected if expected else 0

    gap = main_gaps.get(n, n_draws)
    gap_score = min(gap / max(n_draws, 1), 1.0)

    tier = tiers.get(n, "mild")
    tier_bonus = {"hot": 0.15, "mild": 0.25, "cold": 0.10}[tier]

    # Blend: frequent-but-not-extreme + slight overdue boost
    score = 0.55 * min(freq_ratio, 1.5) / 1.5 + 0.30 * gap_score + tier_bonus
    return round(score, 4)


def delta_similarity(nums: list[int], common: list[tuple[int, ...]]) -> float:
    sorted_nums = sorted(nums)
    deltas = tuple(sorted_nums[i] - sorted_nums[i - 1] for i in range(1, len(sorted_nums)))
    if deltas in common:
        return 1.0
    best = 0.0
    for ref in common:
        diff = sum(abs(a - b) for a, b in zip(deltas, ref))
        sim = max(0, 1 - diff / 20)
        best = max(best, sim)
    return best


def count_tiers(nums: list[int], tiers: dict[int, str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for n in nums:
        c[tiers.get(n, "mild")] += 1
    return c


def line_passes_rules(nums: list[int], sum_band: tuple[int, int]) -> dict:
    s = sum(nums)
    odd = sum(1 for n in nums if n % 2)
    low = sum(1 for n in nums if n <= LOW_CUTOFF)
    return {
        "sum": s,
        "sum_in_band": sum_band[0] <= s <= sum_band[1],
        "odd": odd,
        "even": MAIN_PICK - odd,
        "parity_ok": odd in PREFERRED_ODD_COUNTS,
        "low": low,
        "high": MAIN_PICK - low,
        "high_low_ok": low in PREFERRED_LOW_COUNTS,
    }


def score_line(
    nums: list[int],
    stars: list[int],
    sum_band: tuple[int, int],
    main_tiers: dict[int, str],
    star_tiers: dict[int, str],
    main_scores: dict[int, float],
    star_scores: dict[int, float],
    top_deltas: list[tuple[int, ...]],
) -> tuple[float, dict]:
    rules = line_passes_rules(nums, sum_band)
    tier_mix = count_tiers(nums, main_tiers)

    score = 0.0
    breakdown: dict = {}

    if rules["sum_in_band"]:
        score += 30
    breakdown["sum_rule"] = 30 if rules["sum_in_band"] else 0

    if rules["parity_ok"]:
        score += 25
    breakdown["parity_rule"] = 25 if rules["parity_ok"] else 0

    if rules["high_low_ok"]:
        score += 25
    breakdown["high_low_rule"] = 25 if rules["high_low_ok"] else 0

    # 3-1-1 style: 3 hot/mild + 1 cold + flexibility (at least 1 cold, not all hot)
    hot, mild, cold = tier_mix.get("hot", 0), tier_mix.get("mild", 0), tier_mix.get("cold", 0)
    tier_ok = cold >= 1 and hot <= 3 and (hot + mild) >= 3
    if tier_ok:
        score += 20
    breakdown["tier_mix_rule"] = 20 if tier_ok else 0

    num_score = sum(main_scores.get(n, 0) for n in nums) / MAIN_PICK
    score += num_score * 15
    breakdown["number_strength"] = round(num_score * 15, 2)

    dsim = delta_similarity(nums, top_deltas)
    score += dsim * 10
    breakdown["delta_pattern"] = round(dsim * 10, 2)

    star_score = sum(star_scores.get(s, 0) for s in stars) / STAR_PICK
    score += star_score * 10
    breakdown["stars"] = round(star_score * 10, 2)

    breakdown["total"] = round(score, 2)
    breakdown["rules"] = rules
    breakdown["tiers"] = dict(tier_mix)

    return score, breakdown


def rank_individual_numbers(
    draws: list[dict],
    lookback: int,
) -> dict:
    recent = window_draws(draws, lookback)
    main_c, star_c = frequency_tables(recent)
    main_gaps, star_gaps = draws_since_last_seen(draws)
    main_tiers = classify_tiers(main_c, range(MAIN_MIN, MAIN_MAX + 1))
    star_tiers = classify_tiers(star_c, range(STAR_MIN, STAR_MAX + 1))

    main_ranked = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        sc = score_number(n, main_c, main_gaps, main_tiers, len(recent))
        main_ranked.append(
            {
                "number": n,
                "score": sc,
                "tier": main_tiers[n],
                "frequency_last_window": main_c.get(n, 0),
                "draws_since_last_seen": main_gaps[n],
                "parity": "odd" if n % 2 else "even",
                "field": "low" if n <= LOW_CUTOFF else "high",
            }
        )
    main_ranked.sort(key=lambda x: x["score"], reverse=True)

    star_ranked = []
    for s in range(STAR_MIN, STAR_MAX + 1):
        sc = score_number(s, star_c, star_gaps, star_tiers, len(recent), is_star=True)
        star_ranked.append(
            {
                "number": s,
                "score": sc,
                "tier": star_tiers[s],
                "frequency_last_window": star_c.get(s, 0),
                "draws_since_last_seen": star_gaps[s],
            }
        )
    star_ranked.sort(key=lambda x: x["score"], reverse=True)

    return {
        "main": main_ranked,
        "stars": star_ranked,
        "main_tiers": main_tiers,
        "star_tiers": star_tiers,
        "main_scores": {r["number"]: r["score"] for r in main_ranked},
        "star_scores": {r["number"]: r["score"] for r in star_ranked},
    }


def generate_candidate_lines(
    draws: list[dict],
    lookback: int,
    pool_size: int = 28,
    attempts: int = 8000,
    seed: int | None = None,
) -> list[tuple[float, list[int], list[int], dict]]:
    if seed is not None:
        random.seed(seed)

    recent = window_draws(draws, lookback)
    sum_band = compute_sum_band(draws)[:2]
    main_c, star_c = frequency_tables(recent)
    main_gaps, star_gaps = draws_since_last_seen(draws)
    main_tiers = classify_tiers(main_c, range(MAIN_MIN, MAIN_MAX + 1))
    star_tiers = classify_tiers(star_c, range(STAR_MIN, STAR_MAX + 1))
    ranked = rank_individual_numbers(draws, lookback)

    top_mains = [r["number"] for r in ranked["main"][:pool_size]]
    top_stars = [r["number"] for r in ranked["stars"][:8]]

    hot = [n for n, t in main_tiers.items() if t == "hot" and n in top_mains]
    mild = [n for n, t in main_tiers.items() if t == "mild" and n in top_mains]
    # Cold balls often sit outside the top-scored pool — keep full cold tier available
    cold = [n for n, t in main_tiers.items() if t == "cold"]
    if not cold:
        cold = [n for n, t in main_tiers.items() if t == "cold" and n in top_mains]

    top_deltas = common_deltas(window_draws(draws, lookback))

    candidates: list[tuple[float, list[int], list[int], dict]] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(attempts):
        pick: list[int] = []
        if hot and mild and cold:
            pick.append(random.choice(hot))
            pick.append(random.choice(mild))
            pick.append(random.choice(cold))
            rest_pool = [n for n in top_mains if n not in pick]
            pick.extend(random.sample(rest_pool, MAIN_PICK - len(pick)))
        else:
            pick = random.sample(top_mains, MAIN_PICK)

        nums = sorted(set(pick))
        if len(nums) != MAIN_PICK:
            continue

        stars = sorted(random.sample(top_stars, STAR_PICK))
        key = tuple(nums + stars)
        if key in seen:
            continue
        seen.add(key)

        rules = line_passes_rules(nums, sum_band)
        if not (rules["sum_in_band"] and rules["parity_ok"] and rules["high_low_ok"]):
            continue

        tier_mix = count_tiers(nums, main_tiers)
        cold_n = tier_mix.get("cold", 0)
        hot_n = tier_mix.get("hot", 0)
        if cold_n < 1 or hot_n > 3 or (tier_mix.get("hot", 0) + tier_mix.get("mild", 0)) < 3:
            continue

        sc, breakdown = score_line(
            nums,
            stars,
            sum_band,
            main_tiers,
            star_tiers,
            ranked["main_scores"],
            ranked["star_scores"],
            top_deltas,
        )
        candidates.append((sc, nums, stars, breakdown))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def select_diverse_lines(
    candidates: list[tuple[float, list[int], list[int], dict]],
    top_n: int,
    max_per_star_pair: int = 2,
) -> list[dict]:
    """Top lines by score, but cap how many share the same star pair."""
    sorted_c = sorted(candidates, key=lambda x: x[0], reverse=True)
    selected: list[dict] = []
    seen_keys: set[tuple[int, ...]] = set()
    star_pair_use: Counter[tuple[int, ...]] = Counter()

    def try_add(sc, nums, stars, breakdown, enforce_cap: bool) -> bool:
        key = tuple(nums + stars)
        if key in seen_keys:
            return False
        sp = tuple(stars)
        if enforce_cap and star_pair_use[sp] >= max_per_star_pair:
            return False
        seen_keys.add(key)
        star_pair_use[sp] += 1
        selected.append(
            {
                "rank": len(selected) + 1,
                "score": sc,
                "main_numbers": nums,
                "lucky_stars": stars,
                "sum": sum(nums),
                "breakdown": breakdown,
            }
        )
        return True

    for sc, nums, stars, breakdown in sorted_c:
        if len(selected) >= top_n:
            break
        try_add(sc, nums, stars, breakdown, enforce_cap=True)

    if len(selected) < top_n:
        for sc, nums, stars, breakdown in sorted_c:
            if len(selected) >= top_n:
                break
            try_add(sc, nums, stars, breakdown, enforce_cap=False)

    for i, line in enumerate(selected, 1):
        line["rank"] = i
    return selected


def build_consensus_line(
    lines: list[dict],
    main_tiers: dict[int, str],
    star_tiers: dict[int, str],
    sum_band: tuple[int, int],
    position_limits: dict | None = None,
) -> dict | None:
    """
    One final line: the 5+2 combination that appears most across suggested lines
    and still passes sum / parity / high-low / cold rules.
    """
    if not lines:
        return None

    main_freq: Counter[int] = Counter()
    star_freq: Counter[int] = Counter()
    for line in lines:
        for n in line["main_numbers"]:
            main_freq[n] += 1
        for s in line["lucky_stars"]:
            star_freq[s] += 1

    main_pool = [n for n, _ in main_freq.most_common(15)]
    star_pool = [s for s, _ in star_freq.most_common(6)]

    best: dict | None = None
    best_repeat = -1

    for nums in combinations(main_pool, MAIN_PICK):
        nums = list(sorted(nums))
        rules = line_passes_rules(nums, sum_band)
        tier_mix = count_tiers(nums, main_tiers)
        if tier_mix.get("cold", 0) < 1:
            continue
        if not (rules["sum_in_band"] and rules["parity_ok"] and rules["high_low_ok"]):
            continue
        main_repeat = sum(main_freq[n] for n in nums)
        for stars in combinations(star_pool, STAR_PICK):
            stars = list(sorted(stars))
            pos_ok, _ = line_passes_position_ranges(nums, stars, position_limits)
            if not pos_ok:
                continue
            total_repeat = main_repeat + sum(star_freq[s] for s in stars)
            if total_repeat > best_repeat:
                best_repeat = total_repeat
                best = {
                    "main_numbers": nums,
                    "lucky_stars": stars,
                    "sum": sum(nums),
                    "tiers": dict(tier_mix),
                    "repeat_score": total_repeat,
                    "lines_used": len(lines),
                    "main_appearances": {str(n): main_freq[n] for n in nums},
                    "star_appearances": {str(s): star_freq[s] for s in stars},
                    "most_repeated_mains": [
                        {"number": n, "count": c, "tier": main_tiers.get(n, "?")}
                        for n, c in main_freq.most_common(10)
                    ],
                    "most_repeated_stars": [
                        {"number": s, "count": c, "tier": star_tiers.get(s, "?")}
                        for s, c in star_freq.most_common(4)
                    ],
                }
    return best


def build_output(draws: list[dict], lookback: int, top_n: int, seed: int | None) -> dict:
    sum_lo, sum_hi, sum_pct = compute_sum_band(draws)
    ranked = rank_individual_numbers(draws, lookback)
    candidates = generate_candidate_lines(draws, lookback, seed=seed)

    unique_lines = select_diverse_lines(candidates, top_n, max_per_star_pair=2)

    position_ranges = analyze_sorted_positions(draws)

    # Fallback: relax if not enough strict lines
    if len(unique_lines) < top_n:
        seen_lines: set[tuple[int, ...]] = {tuple(ln["main_numbers"] + ln["lucky_stars"]) for ln in unique_lines}
        ranked_nums = [r["number"] for r in ranked["main"][:15]]
        star_nums = [r["number"] for r in ranked["stars"][:4]]
        for combo in combinations(ranked_nums, MAIN_PICK):
            if len(unique_lines) >= top_n:
                break
            nums = sorted(combo)
            rules = line_passes_rules(nums, (sum_lo, sum_hi))
            if not rules["parity_ok"]:
                continue
            for stars in combinations(star_nums, STAR_PICK):
                sc, breakdown = score_line(
                    nums,
                    list(stars),
                    (sum_lo, sum_hi),
                    ranked["main_tiers"],
                    ranked["star_tiers"],
                    ranked["main_scores"],
                    ranked["star_scores"],
                    common_deltas(draws),
                )
                key = tuple(nums + list(stars))
                if key in seen_lines:
                    continue
                seen_lines.add(key)
                unique_lines.append(
                    {
                        "rank": len(unique_lines) + 1,
                        "score": sc,
                        "main_numbers": nums,
                        "lucky_stars": sorted(stars),
                        "sum": sum(nums),
                        "breakdown": breakdown,
                    }
                )
                if len(unique_lines) >= top_n:
                    break
        unique_lines.sort(key=lambda x: x["score"], reverse=True)
        for i, line in enumerate(unique_lines, 1):
            line["rank"] = i

    consensus = build_consensus_line(
        unique_lines,
        ranked["main_tiers"],
        ranked["star_tiers"],
        (sum_lo, sum_hi),
    )

    return {
        "disclaimer": (
            "These suggestions are based on historical pattern matching only. "
            "Lottery draws are random; no line has higher true probability than any other."
        ),
        "draws_analyzed": len(draws),
        "lookback_window": lookback,
        "seed": seed,
        "combinatorics": {
            "total_possible_tickets": math.comb(50, 5) * math.comb(12, 2),
            "formula": "C(50,5) * C(12,2) = 139,838,160",
        },
        "rules_applied": {
            "sum_70_percent_band": {"min": sum_lo, "max": sum_hi, "historical_pct_in_band": round(sum_pct, 1)},
            "parity": "2 odd / 3 even OR 3 odd / 2 even",
            "high_low": "2 low (1-25) / 3 high OR 3 low / 2 high",
            "tier_mix": "3 hot/mild + at least 1 cold",
            "scoring": ["sum", "parity", "high_low", "tier_mix", "frequency", "delta", "stars"],
        },
        "top_10_main_numbers_by_score": ranked["main"][:10],
        "top_lucky_stars_by_score": ranked["stars"][:4],
        "top_suggested_lines": unique_lines[:top_n],
        "top_10_suggested_lines": unique_lines[:top_n],
        "consensus_line": consensus,
        "main_tiers": ranked["main_tiers"],
        "star_tiers": ranked["star_tiers"],
        "sorted_position_ranges": position_ranges,
    }


def print_report(out: dict) -> None:
    print("=" * 64)
    print("EUROMILLIONS — TOP SUGGESTIONS (pattern-scored)")
    print("=" * 64)
    print(f"\n{out['disclaimer']}\n")
    rules = out["rules_applied"]
    band = rules["sum_70_percent_band"]
    seed_note = f"  |  Seed: {out.get('seed', 'n/a')}"
    lb = lookback_label(out["lookback_window"], out["draws_analyzed"])
    print(f"Draws analyzed: {out['draws_analyzed']}  |  Frequency window: {lb}{seed_note}")
    print(f"70% sum band: {band['min']}–{band['max']} ({band['historical_pct_in_band']}% of history)")
    print(f"Total possible tickets: {out['combinatorics']['total_possible_tickets']:,}\n")

    if out.get("sorted_position_ranges"):
        print_position_ranges(out["sorted_position_ranges"])

    print("--- Top 10 main numbers (individual strength) ---")
    for r in out["top_10_main_numbers_by_score"]:
        print(
            f"  {r['number']:2d}  score={r['score']:.3f}  {r['tier']:5s}  "
            f"freq={r['frequency_last_window']:2d}  gap={r['draws_since_last_seen']:2d}  "
            f"{r['parity']}/{r['field']}"
        )

    print("\n--- Top Lucky Stars ---")
    for r in out["top_lucky_stars_by_score"]:
        print(
            f"  {r['number']:2d}  score={r['score']:.3f}  {r['tier']:5s}  "
            f"freq={r['frequency_last_window']:2d}  gap={r['draws_since_last_seen']:2d}"
        )

    tiers = out.get("main_tiers", {})
    tiers = {**tiers, **out.get("star_tiers", {})}
    print_consensus_line(out.get("consensus_line"), tiers)

    lines = out.get("top_suggested_lines") or out["top_10_suggested_lines"]
    print(f"\n--- Top {len(lines)} suggested full lines (5 + 2) ---")
    print("  (Lines require ≥1 cold main number in the 3-1-1 mix)\n")
    for line in lines:
        m = ", ".join(f"{n:02d}" for n in line["main_numbers"])
        s = ", ".join(f"{n:02d}" for n in line["lucky_stars"])
        tier_mix = line.get("breakdown", {}).get("tiers", {})
        print(f"\n  #{line['rank']}  score={line['score']:.1f}  sum={line['sum']}  tiers={tier_mix}")
        print(f"      Main:  {m}")
        print(f"      Stars: {s}")
    print("\n" + "=" * 64)


def print_enhanced_report(out: dict) -> None:
    print("=" * 64)
    print("EUROMILLIONS — ENHANCED PREDICTIONS")
    print("=" * 64)
    print(f"\n{out['disclaimer']}\n")
    band = out["rules_applied"]["sum_70_percent_band"]
    lb = lookback_label(out["lookback_window"], out["draws_analyzed"])
    print(
        f"Mode: enhanced  |  Decay: {out['decay_factor']}  |  Draws: {out['draws_analyzed']}"
        f"  |  Frequency window: {lb}  |  Seed: {out.get('seed', 42)}"
    )
    print(f"Sum band: {band['min']}–{band['max']}  |  Search: {out['rules_applied']['search']}\n")

    if out.get("sorted_position_ranges"):
        print_position_ranges(out["sorted_position_ranges"])

    print("--- Top 15 main numbers ---")
    for r in out["top_15_main_numbers_by_score"][:15]:
        print(
            f"  {r['number']:2d}  score={r['score']:.3f}  {r['tier']:5s}  "
            f"wfreq={r['weighted_frequency']:.1f}  gap={r['draws_since_last_seen']:2d}"
        )

    print("\n--- Top Lucky Stars ---")
    for r in out["top_lucky_stars_by_score"]:
        print(
            f"  {r['number']:2d}  score={r['score']:.3f}  {r['tier']:5s}  "
            f"wfreq={r['weighted_frequency']:.1f}  gap={r['draws_since_last_seen']:2d}"
        )

    tiers = out.get("main_tiers") or {r["number"]: r["tier"] for r in out["top_15_main_numbers_by_score"]}
    star_tiers = out.get("star_tiers") or {r["number"]: r["tier"] for r in out["top_lucky_stars_by_score"]}
    tiers = {**tiers, **star_tiers}

    print_consensus_line(out.get("consensus_line"), tiers)

    print("\n--- Top suggested lines (5 + 2) ---")
    print("  (Each line aims for ≥1 cold main — see tier tags: H=hot M=mild C=cold)\n")
    for line in out["top_suggested_lines"]:
        m = ", ".join(f"{n:02d}" for n in line["main_numbers"])
        s = ", ".join(f"{n:02d}" for n in line["lucky_stars"])
        tier_mix = line.get("breakdown", {}).get("tiers", {})
        print(f"\n  #{line['rank']}  score={line['score']:.1f}  sum={line['sum']}  tiers={tier_mix}")
        print(f"      Main:  {m}")
        print(f"      Stars: {s}")
        tags = " ".join(
            f"{n:02d}{tiers.get(n, 'mild')[0].upper()}" for n in line["main_numbers"]
        )
        print(f"      Tags:  {tags}")
    print("\n" + "=" * 64)


def print_consensus_line(consensus: dict | None, tiers: dict[int, str]) -> None:
    print("\n--- FINAL CONSENSUS LINE (most repeated across your top lines) ---")
    if not consensus:
        print("  Could not build a valid consensus line from this run.")
        return
    m = ", ".join(f"{n:02d}" for n in consensus["main_numbers"])
    s = ", ".join(f"{n:02d}" for n in consensus["lucky_stars"])
    tags = " ".join(
        f"{n:02d}{tiers.get(n, 'mild')[0].upper()}" for n in consensus["main_numbers"]
    )
    print(f"  Main:  {m}")
    print(f"  Stars: {s}")
    print(f"  Tags:  {tags}")
    print(f"  Sum:   {consensus['sum']}  |  Tiers: {consensus['tiers']}")
    print(f"  Repeat score: {consensus['repeat_score']} appearances (from {consensus['lines_used']} lines)")
    print("\n  Most repeated mains in your list:")
    for item in consensus["most_repeated_mains"][:8]:
        print(f"    {item['number']:2d}  ×{item['count']}  {item['tier']}")
    print("  Most repeated stars:")
    for item in consensus["most_repeated_stars"]:
        print(f"    {item['number']:2d}  ×{item['count']}  {item['tier']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate top 10 pattern-compliant EuroMillions suggestions"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "euromillions_last_5_years.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "predictions.json",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=0,
        help="Draws for hot/cold frequency (0 = all draws in JSON, default)",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=10,
        help="Number of lines to output (e.g. 20 for more tickets)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible lines in basic mode (default 42)",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use weighted decay, pair affinity, exhaustive pool search (recommended)",
    )
    parser.add_argument("--decay", type=float, default=0.97, help="Per-draw decay for enhanced mode")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    draws = load_draws(args.input)
    if args.enhanced:
        from prediction_engine import build_enhanced_output

        out = build_enhanced_output(draws, args.lookback, args.lines, args.seed, args.decay)
        print_enhanced_report(out)
    else:
        out = build_output(draws, args.lookback, args.lines, args.seed)
        print_report(out)

    if not args.no_save:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
