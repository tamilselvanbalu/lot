#!/usr/bin/env python3
"""
Fusion draw: fair random simulation + enhanced historical scoring.

Unlike pattern-only picks (searched combinations) or pure Lucky Dip (no filter),
this generates thousands of truly random tickets, keeps those that match how
real draws behave (sum, parity, high/low, cold mix), then ranks by enhanced score.

  python3 hybrid_draw.py
  python3 hybrid_draw.py --pool 30000 --lines 10 --seed 42
  python3 update_results.py && python3 hybrid_draw.py

DISCLAIMER: Still random lottery odds. This is a hybrid workflow, not an edge.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from pathlib import Path

from generate_predictions import (
    analyze_sorted_positions,
    build_consensus_line,
    compute_sum_band,
    count_tiers,
    draws_since_last_seen,
    line_meets_long_due,
    line_passes_position_ranges,
    line_passes_rules,
    load_draws,
    long_due_mains_on_line,
    lookback_label,
    position_limits_from_draws,
    print_consensus_line,
    select_diverse_lines,
    window_draws,
    STAR_MIN,
    STAR_MAX,
)
from prediction_engine import (
    build_enhanced_output,
    generate_enhanced_lines,
    rank_numbers_enhanced,
    score_line_enhanced,
)
from simulate_draw import MAIN_PICK, simulate_euromillions_draw


def parse_exclude_stars(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    stars = {int(x.strip()) for x in raw.split(",") if x.strip()}
    bad = [s for s in stars if s < STAR_MIN or s > STAR_MAX]
    if bad:
        raise SystemExit(f"--exclude-stars out of range 1-12: {bad}")
    if len(stars) >= STAR_MAX:
        raise SystemExit("--exclude-stars must leave at least one star available")
    return stars


def apply_flat_stars(ctx: dict) -> None:
    """Equal star scores so history does not bias which stars reach top lines."""
    flat = 1.0
    ctx["star_scores"] = {s: flat for s in range(STAR_MIN, STAR_MAX + 1)}
    ctx["star_tiers"] = {s: "mild" for s in range(STAR_MIN, STAR_MAX + 1)}
    for row in ctx["stars"]:
        row["score"] = flat
        row["tier"] = "mild"


def random_valid_ticket(
    sum_band: tuple[int, int],
    main_tiers: dict[int, str],
    position_limits: dict | None = None,
    exclude_stars: set[int] | None = None,
    main_gaps: dict[int, int] | None = None,
    long_due_gap: int = 0,
    long_due_min: int = 0,
    max_tries: int = 200,
) -> tuple[list[int], list[int]] | None:
    """Fair mechanical draw; retry until structural rules + ≥1 cold main (+ position bands)."""
    for _ in range(max_tries):
        main, stars = simulate_euromillions_draw()
        if exclude_stars and (set(stars) & exclude_stars):
            continue
        rules = line_passes_rules(main, sum_band)
        if not (rules["sum_in_band"] and rules["parity_ok"] and rules["high_low_ok"]):
            continue
        if count_tiers(main, main_tiers).get("cold", 0) < 1:
            continue
        if main_gaps and long_due_min > 0:
            if not line_meets_long_due(main, main_gaps, long_due_gap, long_due_min):
                continue
        pos_ok, _ = line_passes_position_ranges(main, stars, position_limits)
        if not pos_ok:
            continue
        return main, stars
    return None


def build_fusion_pool(
    draws: list[dict],
    pool_size: int,
    lookback: int,
    decay: float,
    position_filter: bool = True,
    position_band: float = 70,
    flat_stars: bool = False,
    exclude_stars: set[int] | None = None,
) -> tuple[list[tuple[float, list[int], list[int], dict]], dict, int, dict | None]:
    attempts = 0
    sum_band = compute_sum_band(draws)[:2]
    position_limits = (
        position_limits_from_draws(draws, band_pct=position_band) if position_filter else None
    )
    ctx = rank_numbers_enhanced(draws, lookback, decay)
    if flat_stars:
        apply_flat_stars(ctx)
    recent = window_draws(draws, lookback)

    from generate_predictions import common_deltas

    top_deltas = common_deltas(recent)

    candidates: list[tuple[float, list[int], list[int], dict]] = []
    seen: set[tuple[int, ...]] = set()
    # Tighter bands (e.g. 70%) reject more random tickets than wide bands (e.g. 90%).
    if position_limits:
        tail = (100 - position_band) / 2
        attempt_mult = 30 + int(tail * 1.4)
    else:
        attempt_mult = 25
    max_attempts = pool_size * attempt_mult

    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        ticket = random_valid_ticket(
            sum_band, ctx["main_tiers"], position_limits, exclude_stars
        )
        if ticket is None:
            continue
        main, stars = ticket
        key = tuple(main + stars)
        if key in seen:
            continue
        seen.add(key)
        sc, breakdown = score_line_enhanced(main, stars, ctx, sum_band, top_deltas)
        breakdown["origin"] = "fair_random_simulation"
        candidates.append((sc, main, stars, breakdown))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates, ctx, attempts, position_limits


def fusion_line_as_ticket(line: dict, source: str) -> dict:
    tiers = line.get("breakdown", {}) or {}
    return {
        "main_numbers": line["main_numbers"],
        "lucky_stars": line["lucky_stars"],
        "sum": line["sum"],
        "tiers": dict(tiers.get("tiers", {})),
        "ticket_from": source,
        "fusion_rank": line.get("rank"),
        "fusion_score": line.get("score"),
    }


def fusion_candidate_as_ticket(
    score: float,
    main: list[int],
    stars: list[int],
    breakdown: dict,
    source: str,
    pool_rank: int | None = None,
    pool_size: int | None = None,
    main_gaps: dict[int, int] | None = None,
    long_due_gap: int = 0,
) -> dict:
    ticket = {
        "main_numbers": main,
        "lucky_stars": stars,
        "sum": sum(main),
        "tiers": dict(breakdown.get("tiers", {})),
        "ticket_from": source,
        "fusion_rank": pool_rank,
        "fusion_score": round(score, 2),
        "pool_size": pool_size,
        "pool_percentile": (
            round(100 * pool_rank / pool_size, 1) if pool_rank and pool_size else None
        ),
    }
    if main_gaps and long_due_gap > 0:
        ticket["long_due_mains"] = long_due_mains_on_line(main, main_gaps, long_due_gap)
        ticket["long_due_gap_threshold"] = long_due_gap
    return ticket


def _pick_random_pool_ticket(
    fusion_candidates: list[tuple[float, list[int], list[int], dict]],
    rng: random.Random,
    source: str,
    main_gaps: dict[int, int] | None = None,
    long_due_gap: int = 0,
) -> dict:
    score, main, stars, breakdown = rng.choice(fusion_candidates)
    pool_size = len(fusion_candidates)
    pool_rank = next(
        (
            i + 1
            for i, row in enumerate(fusion_candidates)
            if row[1] == main and row[2] == stars
        ),
        None,
    )
    return fusion_candidate_as_ticket(
        score,
        main,
        stars,
        breakdown,
        source,
        pool_rank,
        pool_size,
        main_gaps=main_gaps,
        long_due_gap=long_due_gap,
    )


def resolve_play_ticket(
    fusion_lines: list[dict],
    consensus: dict | None,
    ticket_from: str,
    random_from: int,
    seed: int | None,
    fusion_candidates: list[tuple[float, list[int], list[int], dict]] | None = None,
    main_gaps: dict[int, int] | None = None,
    long_due_gap: int = 0,
) -> dict | None:
    """Which 5+2 to play: consensus, top line, random top-N, or uniform random from full pool."""
    rng = random.Random(seed) if seed is not None else random.SystemRandom()

    if ticket_from in ("pool", "long-due") and fusion_candidates:
        return _pick_random_pool_ticket(
            fusion_candidates,
            rng,
            ticket_from,
            main_gaps=main_gaps,
            long_due_gap=long_due_gap if ticket_from == "long-due" else 0,
        )

    if ticket_from == "top" and fusion_lines:
        return fusion_line_as_ticket(fusion_lines[0], "top")
    if ticket_from == "random" and fusion_lines:
        pool = fusion_lines[: max(1, random_from)]
        return fusion_line_as_ticket(rng.choice(pool), "random")
    if consensus:
        ticket = dict(consensus)
        ticket["ticket_from"] = "consensus"
        return ticket
    if fusion_lines:
        return fusion_line_as_ticket(fusion_lines[0], "top")
    return None


def print_play_line(ticket: dict | None, tiers: dict[int, str], source: str = "") -> None:
    label = f"PLAY LINE ({source})" if source else "PLAY LINE"
    print(f"\n--- {label} ---")
    if not ticket:
        print("  No ticket selected.")
        return
    m = ", ".join(f"{n:02d}" for n in ticket["main_numbers"])
    s = ", ".join(f"{n:02d}" for n in ticket["lucky_stars"])
    tags = " ".join(
        f"{n:02d}{tiers.get(n, 'mild')[0].upper()}" for n in ticket["main_numbers"]
    )
    print(f"  Main:  {m}")
    print(f"  Stars: {s}")
    print(f"  Tags:  {tags}")
    print(f"  Sum:   {ticket['sum']}  |  Tiers: {ticket.get('tiers', {})}")
    if ticket.get("fusion_rank") is not None:
        print(
            f"  Fusion: #{ticket['fusion_rank']}  score={ticket.get('fusion_score', '?')}"
        )


def print_fusion_report(out: dict) -> None:
    print("=" * 68)
    print("FUSION DRAW — random machine + enhanced pattern rank")
    print("=" * 68)
    print(f"\n{out['disclaimer']}\n")
    meta = out["fusion_meta"]
    print(f"Draws in history:     {out['draws_analyzed']}")
    print(f"Random pool size:     {meta['pool_generated']:,} fair tickets")
    print(f"Pool attempts:        {meta['pool_attempts']:,}")
    print(f"Enhanced search pool: {meta['enhanced_candidates']:,} scored combos")
    print(
        f"Frequency window:     {lookback_label(meta['lookback'], out['draws_analyzed'])}  |  "
        f"decay: {meta['decay']}"
    )
    print(f"Seed:                 {meta['seed']}")

    print("\n--- How this differs ---")
    for line in out["unique_approach"]:
        print(f"  • {line}")

    tiers = {**out.get("main_tiers", {}), **out.get("star_tiers", {})}
    play = out.get("play_line")
    src = out.get("ticket_from", "consensus")
    if play:
        print_play_line(play, tiers, src)
    if src != "consensus" and out.get("consensus_line"):
        print("\n--- CONSENSUS (vote across top lines; not used as play line) ---")
        print_consensus_line(out["consensus_line"], tiers)
    elif src == "consensus":
        pass
    elif not play:
        print_consensus_line(out.get("consensus_line"), tiers)

    print("\n--- TOP FUSION LINES (random-born, pattern-ranked) ---")
    for line in out["fusion_lines"]:
        m = ", ".join(f"{n:02d}" for n in line["main_numbers"])
        s = ", ".join(f"{n:02d}" for n in line["lucky_stars"])
        print(
            f"\n  #{line['rank']}  fusion={line['score']:.1f}  sum={line['sum']}  "
            f"tiers={line.get('breakdown', {}).get('tiers', {})}"
        )
        print(f"      Main:  {m}")
        print(f"      Stars: {s}")

    if out.get("enhanced_reference_lines"):
        print("\n--- REFERENCE: top enhanced-only (searched, not random-born) ---")
        ref = out["enhanced_reference_lines"][0]
        m = ", ".join(f"{n:02d}" for n in ref["main_numbers"])
        s = ", ".join(f"{n:02d}" for n in ref["lucky_stars"])
        print(f"  Enhanced #1: {m}  +  {s}  (score {ref['score']:.1f})")
        if out["fusion_lines"]:
            f0 = out["fusion_lines"][0]
            same = (
                f0["main_numbers"] == ref["main_numbers"]
                and f0["lucky_stars"] == ref["lucky_stars"]
            )
            print(f"  Same as fusion #1: {'yes' if same else 'no — fusion is independent'}")

    print("\n" + "=" * 68)


def _rank_lookup(ranked: list[dict]) -> dict[int, dict]:
    return {r["number"]: r for r in ranked}


def print_morning_explanation(out: dict) -> None:
    consensus = out.get("play_line") or out.get("consensus_line")
    if not consensus:
        return

    fusion_lines = out.get("fusion_lines", [])
    ticket_from = out.get("ticket_from", "consensus")
    meta = out.get("fusion_meta", {})
    band = out.get("sum_band", {})
    main_rank = _rank_lookup(out.get("all_main_ranked", out.get("top_15_main_numbers_by_score", [])))
    star_rank = _rank_lookup(out.get("all_stars_ranked", out.get("top_lucky_stars_by_score", [])))
    main_tiers = out.get("main_tiers", {})
    star_tiers = out.get("star_tiers", {})
    n_lines = consensus.get("lines_used", len(fusion_lines))
    pool = meta.get("pool_generated", "?")

    print("--- WHY THESE NUMBERS? ---\n")
    print("How this ticket was built:")
    print(f"  1. Generated {pool:,} random tickets (fair draw, like the machine).")
    pos_on = out.get("position_filter", False)
    step2 = "sum, odd/even, high/low, ≥1 cold"
    if pos_on:
        pct = out.get("position_band", 70)
        step2 += f", sorted position bands (~{pct:.0f}% history)"
    if meta.get("flat_stars"):
        step2 += ", flat star scoring"
    ex = meta.get("exclude_stars") or []
    if ex:
        step2 += f", excluding stars {', '.join(f'{s:02d}' for s in ex)}"
    print(f"  2. Kept only tickets matching history-shaped rules ({step2}).")
    print(f"  3. Scored survivors using all {out.get('draws_analyzed', '?')} draws in your JSON.")
    if ticket_from == "consensus":
        print(
            f"  4. Took top {n_lines} lines, voted most repeated 5+2 → consensus "
            f"(same history ⇒ similar picks each run).\n"
        )
    elif ticket_from == "top":
        print("  4. Play line = fusion #1 (best score in this pool; changes when pool is re-rolled).\n")
    elif ticket_from == "pool":
        pool_n = meta.get("pool_generated", meta.get("pool_target", "?"))
        print(
            f"  4. Play line = uniform random from all {pool_n:,} valid pool tickets "
            f"(score-neutral; not limited to top ranked lines).\n"
        )
    else:
        rf = out.get("random_from", 5)
        print(f"  4. Play line = random choice from top {rf} fusion lines (most variety).\n")

    rules = line_passes_rules(consensus["main_numbers"], (band.get("min", 0), band.get("max", 999)))
    print("This line passes the filters:")
    print(f"  • Sum {consensus['sum']} in band {band.get('min')}–{band.get('max')}: {rules['sum_in_band']}")
    print(
        f"  • Odd/even {rules['odd']} odd / {rules['even']} even: {rules['parity_ok']} "
        f"(target 2–3 odd)"
    )
    print(
        f"  • Low/high {rules['low']} low / {rules['high']} high: {rules['high_low_ok']} "
        f"(target 2–3 low)"
    )
    print(f"  • Tier mix {consensus.get('tiers', {})} (need at least 1 cold main)")

    limits = out.get("position_limits")
    if limits:
        sm = sorted(consensus["main_numbers"])
        print("  • Sorted position bands (smallest→largest main, then stars):")
        for i, n in enumerate(sm):
            band = limits["main"][i]
            ok = band["from"] <= n <= band["to"]
            print(f"      Main pos {band['position']}: {n:02d} in {band['from']}–{band['to']}: {ok}")
        ss = sorted(consensus["lucky_stars"])
        for i, s in enumerate(ss):
            band = limits["stars"][i]
            ok = band["from"] <= s <= band["to"]
            print(f"      Star pos {band['position']}: {s:02d} in {band['from']}–{band['to']}: {ok}")
    print()

    print("Each MAIN number:")
    for n in consensus["main_numbers"]:
        tier = main_tiers.get(n, "?")
        in_top = sum(1 for ln in fusion_lines if n in ln["main_numbers"])
        app = consensus.get("main_appearances", {}).get(str(n), in_top)
        info = main_rank.get(n, {})
        score = info.get("score", "—")
        gap = info.get("draws_since_last_seen", "—")
        freq = info.get("weighted_frequency", info.get("frequency_last_window", "—"))
        field = "low" if n <= 25 else "high"
        parity = "odd" if n % 2 else "even"
        print(
            f"  {n:02d}  tier={tier:5s}  in {in_top}/{n_lines} top lines  "
            f"(×{app} weighted)  score={score}  gap={gap} draws  "
            f"hist_freq={freq}  {parity}/{field}"
        )

    print("\nEach STAR:")
    for s in consensus["lucky_stars"]:
        tier = star_tiers.get(s, "?")
        in_top = sum(1 for ln in fusion_lines if s in ln["lucky_stars"])
        app = consensus.get("star_appearances", {}).get(str(s), in_top)
        info = star_rank.get(s, {})
        score = info.get("score", "—")
        gap = info.get("draws_since_last_seen", "—")
        freq = info.get("weighted_frequency", info.get("frequency_last_window", "—"))
        print(
            f"  {s:02d}  tier={tier:5s}  in {in_top}/{n_lines} top lines  "
            f"(×{app} weighted)  score={score}  gap={gap} draws  hist_freq={freq}"
        )

    print(f"\nRepeat score: {consensus.get('repeat_score')} = how often these 7 balls")
    print("appeared together across the top lines (higher = stronger agreement).\n")

    print("Top fusion lines that drove this (random-born, highest scores):")
    for ln in fusion_lines[:3]:
        m = ", ".join(f"{x:02d}" for x in ln["main_numbers"])
        st = ", ".join(f"{x:02d}" for x in ln["lucky_stars"])
        print(f"  #{ln['rank']}  score={ln['score']:.1f}  main=[{m}]  stars=[{st}]")

    print("\nNote: This explains pattern fit, not future luck. Each draw is still random.\n")


def print_morning_ticket(consensus: dict | None, draws_count: int, data_to: str) -> None:
    print()
    print("=" * 56)
    print("  EUROMILLIONS — TODAY'S CONSENSUS TICKET")
    print("=" * 56)
    print(f"  History: {draws_count} draws (latest in file: {data_to})")
    if not consensus:
        print("  Could not build a valid consensus line.")
        print("=" * 56)
        return
    m = ", ".join(f"{n:02d}" for n in consensus["main_numbers"])
    s = ", ".join(f"{n:02d}" for n in consensus["lucky_stars"])
    print(f"  Main:   {m}")
    print(f"  Stars:  {s}")
    print(f"  Sum:    {consensus['sum']}")
    print("=" * 56)
    print("  Enter these on your National Lottery slip or app.")
    print("  Odds are unchanged — play only what you can afford.")
    print()


def print_compact_ticket(play_line: dict | None) -> None:
    """Minimal output: main, stars, sum only."""
    if not play_line:
        print("No valid ticket generated.")
        return
    m = ", ".join(f"{n:02d}" for n in play_line["main_numbers"])
    s = ", ".join(f"{n:02d}" for n in play_line["lucky_stars"])
    print(f"Main:   {m}")
    print(f"Stars:  {s}")
    print(f"Sum:    {play_line['sum']}")
    rank = play_line.get("fusion_rank")
    pct = play_line.get("pool_percentile")
    if play_line.get("ticket_from") in ("pool", "long-due") and rank and pct is not None:
        label = "long-due random" if play_line.get("ticket_from") == "long-due" else "not top-scored"
        print(f"Pool:   rank {rank:,} of {play_line.get('pool_size', '?'):,} ({pct}% — {label})")
    long_due = play_line.get("long_due_mains") or []
    if long_due:
        gap_thr = play_line.get("long_due_gap_threshold", "?")
        parts = ", ".join(f"{x['number']:02d} ({x['gap']} draws)" for x in long_due)
        print(f"Long due (≥{gap_thr} draws): {parts}")


def build_fusion_output(
    draws: list[dict],
    pool_size: int = 20000,
    lookback: int = 0,
    decay: float = 0.97,
    top_n: int = 10,
    seed: int | None = None,
    include_enhanced_reference: bool = True,
    position_filter: bool = True,
    position_band: float = 70,
    ticket_from: str = "consensus",
    random_from: int = 5,
    flat_stars: bool = False,
    exclude_stars: set[int] | None = None,
) -> dict:
    fusion_candidates, ctx, pool_attempts, position_limits = build_fusion_pool(
        draws,
        pool_size,
        lookback,
        decay,
        position_filter=position_filter,
        position_band=position_band,
        flat_stars=flat_stars,
        exclude_stars=exclude_stars,
    )
    fusion_lines = select_diverse_lines(fusion_candidates, top_n, max_per_star_pair=2)

    enhanced_out = None
    enhanced_candidates: list = []
    if include_enhanced_reference:
        enhanced_out = build_enhanced_output(draws, lookback, top_n=3, seed=seed, decay=decay)
        enhanced_candidates, _ = generate_enhanced_lines(
            draws, lookback, decay, top_n=top_n, seed=seed
        )

    sum_lo, sum_hi, _ = compute_sum_band(draws)
    consensus = build_consensus_line(
        fusion_lines,
        ctx["main_tiers"],
        ctx["star_tiers"],
        (sum_lo, sum_hi),
        position_limits=position_limits,
    )

    position_ranges = (
        analyze_sorted_positions(draws, band_pct=position_band) if position_filter else None
    )
    play_line = resolve_play_ticket(
        fusion_lines,
        consensus,
        ticket_from,
        random_from,
        seed,
        fusion_candidates=fusion_candidates if ticket_from == "pool" else None,
    )

    return {
        "mode": "fusion",
        "ticket_from": ticket_from,
        "random_from": random_from,
        "play_line": play_line,
        "position_filter": position_filter,
        "position_band": position_band,
        "position_limits": position_limits,
        "sorted_position_ranges": position_ranges,
        "sum_band": {"min": sum_lo, "max": sum_hi},
        "disclaimer": (
            "Fusion lines are fair random draws filtered and ranked by history-shaped rules. "
            "They are not more likely to win than any other valid ticket."
        ),
        "draws_analyzed": len(draws),
        "fusion_meta": {
            "pool_generated": len(fusion_candidates),
            "pool_attempts": pool_attempts,
            "pool_target": pool_size,
            "enhanced_candidates": len(enhanced_candidates),
            "lookback": lookback,
            "decay": decay,
            "seed": seed,
            "flat_stars": flat_stars,
            "exclude_stars": sorted(exclude_stars) if exclude_stars else [],
        },
        "unique_approach": [
            "Random: each ticket from secrets (mechanical draw), not combinatorial search",
            f"Filter: sum, parity, high/low, cold-ball mix, sorted position bands (~{position_band:.0f}% history)",
            "Rank: enhanced score (decay frequency, pairs, sum centre, stars)",
            "Diversify: star pairs capped so list is not identical",
        ],
        "fusion_lines": fusion_lines,
        "top_suggested_lines": fusion_lines,
        "consensus_line": consensus,
        "enhanced_reference_lines": (
            enhanced_out.get("top_suggested_lines", [])[:3] if enhanced_out else []
        ),
        "top_15_main_numbers_by_score": ctx["main"][:15],
        "top_lucky_stars_by_score": ctx["stars"][:6],
        "all_main_ranked": ctx["main"],
        "all_stars_ranked": ctx["stars"],
        "main_tiers": ctx["main_tiers"],
        "star_tiers": ctx["star_tiers"],
    }


def run_morning(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    lot_dir = Path(__file__).parent
    print("Updating latest results...")
    subprocess.run(
        [sys.executable, str(lot_dir / "update_results.py")],
        cwd=lot_dir,
        check=True,
    )
    draws = load_draws(args.input)
    out = build_fusion_output(
        draws,
        pool_size=args.pool,
        lookback=args.lookback,
        decay=args.decay,
        top_n=args.lines,
        seed=args.seed,
        include_enhanced_reference=False,
        position_filter=not args.no_position_filter,
        position_band=args.position_band,
        ticket_from=args.ticket_from,
        random_from=args.random_from,
        flat_stars=args.flat_stars,
        exclude_stars=parse_exclude_stars(args.exclude_stars),
    )
    latest = draws[0]["date"] if draws else "?"
    print_morning_ticket(out.get("play_line"), len(draws), latest)
    if args.explain:
        print_morning_explanation(out)
    if not args.no_save:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Full details saved: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusion: simulate_draw randomness + enhanced prediction scoring"
    )
    parser.add_argument(
        "--morning",
        action="store_true",
        help="Update JSON, run fusion, print consensus ticket only (no --seed needed)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "euromillions_last_5_years.json",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path(__file__).parent / "fusion_predictions.json")
    parser.add_argument("--pool", type=int, default=20000, help="Random tickets to generate and score")
    parser.add_argument("--lookback", type=int, default=0, help="0 = all draws in JSON")
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("-n", "--lines", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional: fix random pool for repeatability (omit for fresh picks)",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show why each consensus number was chosen (default on for --morning)",
    )
    parser.add_argument(
        "--no-position-filter",
        action="store_true",
        help="Disable per-slot filters (sorted main pos 1–5, star pos 1–2)",
    )
    parser.add_argument(
        "--position-band",
        type=float,
        default=70,
        metavar="PCT",
        help="%% of history inside each sorted-slot band (default 70 = 15th-85th; 90 = 5th-95th)",
    )
    parser.add_argument(
        "--ticket-from",
        choices=("consensus", "top", "random", "pool"),
        default="consensus",
        help="Play line: consensus (default), top, random from top-N, or pool (uniform from all valid tickets)",
    )
    parser.add_argument(
        "--random-from",
        type=int,
        default=5,
        metavar="N",
        help="With --ticket-from random, pick from top N fusion lines (default 5)",
    )
    parser.add_argument(
        "--flat-stars",
        action="store_true",
        help="Treat all 12 stars as equal in scoring (more star variety)",
    )
    parser.add_argument(
        "--exclude-stars",
        metavar="LIST",
        default=None,
        help="Comma-separated stars to reject, e.g. 2,10",
    )
    args = parser.parse_args()
    if not 0 < args.position_band <= 100:
        raise SystemExit("--position-band must be > 0 and <= 100")

    if args.morning:
        run_morning(args)
        return

    draws = load_draws(args.input)
    out = build_fusion_output(
        draws,
        pool_size=args.pool,
        lookback=args.lookback,
        decay=args.decay,
        top_n=args.lines,
        seed=args.seed,
        include_enhanced_reference=True,
        position_filter=not args.no_position_filter,
        position_band=args.position_band,
        ticket_from=args.ticket_from,
        random_from=args.random_from,
        flat_stars=args.flat_stars,
        exclude_stars=parse_exclude_stars(args.exclude_stars),
    )
    print_fusion_report(out)
    if args.explain and out.get("play_line"):
        print_morning_explanation(out)

    if not args.no_save:
        # strip heavy breakdown for json size optional - keep full for now
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
