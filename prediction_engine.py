"""
Enhanced EuroMillions prediction engine.

Improvements over basic generator:
  - Exponential decay weighting (recent draws matter more)
  - Pair/triplet co-occurrence affinity when building lines
  - Exhaustive search over top-N number pools (not random sampling)
  - Sum targeted toward historical median (15th–85th band center)
  - Expanded top-number shortlist (15 mains, 6 stars)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations

from generate_predictions import (
    MAIN_MAX,
    MAIN_MIN,
    MAIN_PICK,
    LOW_CUTOFF,
    PREFERRED_LOW_COUNTS,
    PREFERRED_ODD_COUNTS,
    STAR_MAX,
    STAR_MIN,
    STAR_PICK,
    classify_tiers,
    common_deltas,
    compute_sum_band,
    count_tiers,
    delta_similarity,
    draws_since_last_seen,
    line_passes_rules,
    load_draws,
    percentile,
    score_line,
    window_draws,
)

Path = __import__("pathlib").Path


def weighted_counts(draws: list[dict], decay: float = 0.97) -> tuple[Counter[int], Counter[int]]:
    """Newest draw (index 0) has weight 1.0; older draws decay."""
    main_c: Counter[int] = Counter()
    star_c: Counter[int] = Counter()
    for i, d in enumerate(draws):
        w = decay**i
        for n in d["numbers"]:
            main_c[n] += w
        for s in d["stars"]:
            star_c[s] += w
    return main_c, star_c


def pair_counts(draws: list[dict], size: int = 2) -> Counter[tuple[int, ...]]:
    c: Counter[tuple[int, ...]] = Counter()
    for d in draws:
        for combo in combinations(sorted(d["numbers"]), size):
            c[combo] += 1
    return c


def score_number_enhanced(
    n: int,
    w_main: Counter[int],
    main_gaps: dict[int, int],
    tiers: dict[int, str],
    total_weight: float,
    pair_c: Counter[tuple[int, ...]],
    top_pool: set[int],
    is_star: bool = False,
) -> float:
    pool = STAR_MAX if is_star else MAIN_MAX
    picks = STAR_PICK if is_star else MAIN_PICK
    expected = total_weight * picks / pool if total_weight else 1.0
    freq = w_main.get(n, 0)
    freq_ratio = freq / expected if expected else 0

    # Sweet spot: slightly above average, not extreme hot
    if freq_ratio <= 1.0:
        freq_component = freq_ratio * 0.85
    elif freq_ratio <= 1.35:
        freq_component = 0.85 + (freq_ratio - 1.0) * 0.4
    else:
        freq_component = max(0.5, 1.0 - (freq_ratio - 1.35) * 0.5)

    gap = main_gaps.get(n, 0)
    gap_component = min(gap / 25.0, 1.0) * 0.35

    tier = tiers.get(n, "mild")
    tier_component = {"hot": 0.12, "mild": 0.22, "cold": 0.18}[tier]

    # Pair affinity with other strong candidates
    pair_bonus = 0.0
    if not is_star and top_pool:
        links = 0.0
        for other in top_pool:
            if other == n:
                continue
            key = tuple(sorted((n, other)))
            links += pair_c.get(key, 0)
        pair_bonus = min(links / 30.0, 0.2)

    return round(freq_component + gap_component + tier_component + pair_bonus, 4)


def pair_cohesion(nums: list[int], pair_c: Counter[tuple[int, ...]]) -> float:
    total = 0.0
    for a, b in combinations(nums, 2):
        total += pair_c.get(tuple(sorted((a, b))), 0)
    max_pairs = math.comb(MAIN_PICK, 2)
    return min(total / (max_pairs * 8), 1.0)


def sum_target_score(s: int, band: tuple[int, int]) -> float:
    mid = (band[0] + band[1]) / 2
    half = (band[1] - band[0]) / 2 or 1
    dist = abs(s - mid) / half
    return max(0, 1 - dist)


def rank_numbers_enhanced(draws: list[dict], lookback: int, decay: float) -> dict:
    recent = window_draws(draws, lookback)
    w_main, w_star = weighted_counts(recent, decay)
    total_w = sum(decay**i for i in range(len(recent))) or 1.0
    main_gaps, star_gaps = draws_since_last_seen(draws)
    main_tiers = classify_tiers(
        Counter({k: int(v) for k, v in w_main.items()}), range(MAIN_MIN, MAIN_MAX + 1)
    )
    star_tiers = classify_tiers(
        Counter({k: int(v) for k, v in w_star.items()}), range(STAR_MIN, STAR_MAX + 1)
    )
    pair_c = pair_counts(recent, 2)

    # Initial pool for pair scoring
    prelim = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        prelim.append((n, w_main.get(n, 0)))
    prelim.sort(key=lambda x: x[1], reverse=True)
    top_pool = {n for n, _ in prelim[:22]}

    main_ranked = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        sc = score_number_enhanced(
            n, w_main, main_gaps, main_tiers, total_w, pair_c, top_pool, is_star=False
        )
        main_ranked.append(
            {
                "number": n,
                "score": sc,
                "tier": main_tiers[n],
                "weighted_frequency": round(w_main.get(n, 0), 2),
                "draws_since_last_seen": main_gaps[n],
                "parity": "odd" if n % 2 else "even",
                "field": "low" if n <= LOW_CUTOFF else "high",
            }
        )
    main_ranked.sort(key=lambda x: x["score"], reverse=True)

    star_prelim = {n for n, _ in sorted(w_star.items(), key=lambda x: x[1], reverse=True)[:8]}
    star_ranked = []
    for s in range(STAR_MIN, STAR_MAX + 1):
        sc = score_number_enhanced(
            s, w_star, star_gaps, star_tiers, total_w, pair_c, star_prelim, is_star=True
        )
        star_ranked.append(
            {
                "number": s,
                "score": sc,
                "tier": star_tiers[s],
                "weighted_frequency": round(w_star.get(s, 0), 2),
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
        "pair_counts": pair_c,
        "total_weight": total_w,
    }


def score_line_enhanced(
    nums: list[int],
    stars: list[int],
    ctx: dict,
    sum_band: tuple[int, int],
    top_deltas: list[tuple[int, ...]],
) -> tuple[float, dict]:
    base, breakdown = score_line(
        nums,
        stars,
        sum_band,
        ctx["main_tiers"],
        ctx["star_tiers"],
        ctx["main_scores"],
        ctx["star_scores"],
        top_deltas,
    )
    cohesion = pair_cohesion(nums, ctx["pair_counts"])
    cohesion_pts = cohesion * 12
    sum_pts = sum_target_score(sum(nums), sum_band) * 8
    base += cohesion_pts + sum_pts
    breakdown["pair_cohesion"] = round(cohesion_pts, 2)
    breakdown["sum_center"] = round(sum_pts, 2)
    breakdown["total"] = round(base, 2)
    return base, breakdown


def generate_enhanced_lines(
    draws: list[dict],
    lookback: int = 0,
    decay: float = 0.97,
    main_pool_size: int = 20,
    star_pool_size: int = 12,
    top_n: int = 10,
    seed: int | None = None,
) -> tuple[list[tuple[float, list[int], list[int], dict]], dict]:
    if seed is not None:
        random.seed(seed)

    sum_band = compute_sum_band(draws)[:2]
    ctx = rank_numbers_enhanced(draws, lookback, decay)
    recent = window_draws(draws, lookback)
    top_deltas = common_deltas(recent)

    top_mains = [r["number"] for r in ctx["main"][:main_pool_size]]
    # Ensure cold-tier mains are searchable even if low individual score
    cold_tier = [n for n, t in ctx["main_tiers"].items() if t == "cold"]
    for n in cold_tier[:8]:
        if n not in top_mains:
            top_mains.append(n)
    top_stars = [r["number"] for r in ctx["stars"][:star_pool_size]]

    candidates: list[tuple[float, list[int], list[int], dict]] = []

    for nums in combinations(top_mains, MAIN_PICK):
        nums = list(sorted(nums))
        rules = line_passes_rules(nums, sum_band)
        if not (rules["sum_in_band"] and rules["parity_ok"] and rules["high_low_ok"]):
            continue
        tier_mix = count_tiers(nums, ctx["main_tiers"])
        if tier_mix.get("cold", 0) < 1 or tier_mix.get("hot", 0) > 3:
            continue
        for stars in combinations(top_stars, STAR_PICK):
            stars = list(sorted(stars))
            sc, breakdown = score_line_enhanced(nums, stars, ctx, sum_band, top_deltas)
            candidates.append((sc, nums, stars, breakdown))

    # Diversify: if too few strict lines, relax sum band slightly
    if len(candidates) < top_n:
        lo, hi = sum_band
        wide = (lo - 10, hi + 10)
        for nums in combinations(top_mains[:18], MAIN_PICK):
            nums = list(sorted(nums))
            rules = line_passes_rules(nums, wide)
            if not rules["parity_ok"] or not rules["high_low_ok"]:
                continue
            for stars in combinations(top_stars, STAR_PICK):
                stars = list(sorted(stars))
                sc, breakdown = score_line_enhanced(nums, stars, ctx, sum_band, top_deltas)
                candidates.append((sc, nums, stars, breakdown))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates, ctx


def build_enhanced_output(
    draws: list[dict],
    lookback: int = 0,
    top_n: int = 10,
    seed: int | None = 42,
    decay: float = 0.97,
) -> dict:
    sum_lo, sum_hi, sum_pct = compute_sum_band(draws)
    candidates, ctx = generate_enhanced_lines(
        draws, lookback, decay, top_n=top_n, seed=seed
    )

    from generate_predictions import select_diverse_lines

    unique_lines = select_diverse_lines(candidates, top_n, max_per_star_pair=2)

    from generate_predictions import analyze_sorted_positions, build_consensus_line

    position_ranges = analyze_sorted_positions(draws)
    consensus = build_consensus_line(
        unique_lines,
        ctx["main_tiers"],
        ctx["star_tiers"],
        (sum_lo, sum_hi),
    )

    return {
        "mode": "enhanced",
        "disclaimer": (
            "Enhanced scoring uses weighted history and pair affinity. "
            "Lottery draws remain random; this does not guarantee better odds."
        ),
        "draws_analyzed": len(draws),
        "lookback_window": lookback,
        "seed": seed,
        "decay_factor": decay,
        "combinatorics": {
            "total_possible_tickets": math.comb(50, 5) * math.comb(12, 2),
        },
        "rules_applied": {
            "sum_70_percent_band": {"min": sum_lo, "max": sum_hi, "historical_pct_in_band": round(sum_pct, 1)},
            "parity": "2-3 odd/even",
            "high_low": "2-3 low/high",
            "search": "exhaustive top-20 mains x top-8 stars (filtered)",
            "scoring": ["base_rules", "pair_cohesion", "sum_center", "delta", "stars"],
        },
        "top_15_main_numbers_by_score": ctx["main"][:15],
        "top_lucky_stars_by_score": ctx["stars"][:6],
        "top_suggested_lines": unique_lines,
        "top_10_main_numbers_by_score": ctx["main"][:10],
        "top_10_suggested_lines": unique_lines[:10],
        "consensus_line": consensus,
        "main_tiers": ctx["main_tiers"],
        "star_tiers": ctx["star_tiers"],
        "sorted_position_ranges": position_ranges,
    }
