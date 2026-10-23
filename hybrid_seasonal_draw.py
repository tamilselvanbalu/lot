#!/usr/bin/env python3
"""
Seasonal fusion draw: hybrid_draw + seasonal_analysis.json bias.

Same random pool and filters as hybrid_draw, but adds a soft seasonal score
from analyze_seasonal.py (month / half-month profiles + surviving patterns).

Defaults favour variety: flat stars + random pick from top lines.

  python3 hybrid_seasonal_draw.py --morning --pool 50000
  python3 hybrid_seasonal_draw.py --refresh-seasonal --pool 50000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from generate_predictions import (
    analyze_sorted_positions,
    build_consensus_line,
    common_deltas,
    compute_sum_band,
    line_meets_long_due,
    load_draws,
    long_due_mains_on_line,
    lookback_label,
    position_limits_from_draws,
    print_consensus_line,
    select_diverse_lines,
    window_draws,
)
from generate_predictions import draws_since_last_seen
from hybrid_draw import (
    apply_flat_stars,
    parse_exclude_stars,
    print_compact_ticket,
    print_morning_explanation,
    print_morning_ticket,
    print_play_line,
    random_valid_ticket,
    resolve_play_ticket,
)
from horoscope_align import (
    get_profile,
    line_meets_horoscope,
    list_profiles,
    resolve_horoscope_profile,
    score_horoscope_fit,
)
from vedic_transit import list_places
from prediction_engine import rank_numbers_enhanced, score_line_enhanced
from seasonal_scoring import (
    build_seasonal_context,
    load_seasonal_analysis,
    parse_play_date,
    play_calendar,
    score_seasonal_fit,
)


def ensure_seasonal_analysis(path: Path, refresh: bool, quiet: bool = False) -> None:
    if path.exists() and not refresh:
        return
    lot_dir = Path(__file__).parent
    venv_py = lot_dir / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    script = lot_dir / "analyze_seasonal.py"
    if not script.exists():
        raise SystemExit("analyze_seasonal.py not found")
    if not quiet:
        print("Building seasonal analysis (analyze_seasonal.py)...")
    cmd = [py, str(script), "-o", str(path)]
    if quiet:
        cmd.append("--quiet")
    subprocess.run(cmd, cwd=lot_dir, check=True)


def build_seasonal_fusion_pool(
    draws: list[dict],
    pool_size: int,
    lookback: int,
    decay: float,
    seasonal_ctx: dict,
    seasonal_weight: float,
    position_filter: bool = True,
    position_band: float = 70,
    flat_stars: bool = True,
    exclude_stars: set[int] | None = None,
    long_due_gap: int = 0,
    long_due_min: int = 0,
    horoscope_profile: dict | None = None,
    horoscope_min_mains: int = 2,
    horoscope_min_stars: int = 1,
    horoscope_weight: float = 1.0,
) -> tuple[list[tuple[float, list[int], list[int], dict]], dict, int, dict | None, dict[int, int]]:
    attempts = 0
    sum_band = compute_sum_band(draws)[:2]
    position_limits = (
        position_limits_from_draws(draws, band_pct=position_band) if position_filter else None
    )
    main_gaps, _ = draws_since_last_seen(draws)
    ctx = rank_numbers_enhanced(draws, lookback, decay)
    if flat_stars:
        apply_flat_stars(ctx)
    recent = window_draws(draws, lookback)
    top_deltas = common_deltas(recent)

    candidates: list[tuple[float, list[int], list[int], dict]] = []
    seen: set[tuple[int, ...]] = set()
    if position_limits:
        tail = (100 - position_band) / 2
        attempt_mult = 30 + int(tail * 1.4)
    else:
        attempt_mult = 25
    if long_due_min > 0:
        attempt_mult += 20 + long_due_min * 8
    if horoscope_profile:
        attempt_mult += 25 + horoscope_min_mains * 10 + horoscope_min_stars * 8
    max_attempts = pool_size * attempt_mult

    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        ticket = random_valid_ticket(
            sum_band,
            ctx["main_tiers"],
            position_limits,
            exclude_stars,
            main_gaps=main_gaps,
            long_due_gap=long_due_gap,
            long_due_min=long_due_min,
        )
        if ticket is None:
            continue
        main, stars = ticket
        if horoscope_profile and not line_meets_horoscope(
            main,
            stars,
            horoscope_profile,
            min_mains=horoscope_min_mains,
            min_stars=horoscope_min_stars,
        ):
            continue
        key = tuple(main + stars)
        if key in seen:
            continue
        seen.add(key)
        base, breakdown = score_line_enhanced(main, stars, ctx, sum_band, top_deltas)
        seasonal_pts, s_break = score_seasonal_fit(main, stars, seasonal_ctx, seasonal_weight)
        breakdown["seasonal"] = s_break
        horo_pts = 0.0
        if horoscope_profile:
            horo_pts, h_break = score_horoscope_fit(
                main,
                stars,
                horoscope_profile,
                main_gaps=main_gaps,
                main_tiers=ctx["main_tiers"],
                long_due_gap=long_due_gap or 15,
                weight=horoscope_weight,
            )
            breakdown["horoscope"] = h_break
        breakdown["origin"] = "fair_random_simulation"
        breakdown["long_due_mains"] = long_due_mains_on_line(main, main_gaps, long_due_gap)
        total = base + seasonal_pts + horo_pts
        breakdown["total"] = round(total, 2)
        candidates.append((total, main, stars, breakdown))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates, ctx, attempts, position_limits, main_gaps


def print_seasonal_banner(seasonal_ctx: dict, weight: float, analysis_path: Path) -> None:
    print("--- SEASONAL CONTEXT ---")
    print(f"  Analysis file: {analysis_path.name}")
    print(
        f"  Play window:   {seasonal_ctx['month_name']} "
        f"(month {seasonal_ctx['month']})  half={seasonal_ctx['half']}"
    )
    p = seasonal_ctx.get("profile") or {}
    if p:
        print(
            f"  Month profile: mean sum {p.get('mean_sum', '?')}, "
            f"2-low rate {p.get('pct_2low', 0):.0%}, "
            f"first ball single-digit {p.get('pct_first_single', 0):.0%}"
        )
    print(
        f"  BH survivors for this window: {len(seasonal_ctx.get('survivors', []))} "
        f"(of {seasonal_ctx.get('survivors_bh_total', '?')} total in file)"
    )
    hints = seasonal_ctx.get("star_hints") or []
    if hints:
        stars = ", ".join(f"{h['star']:02d}" for h in hints)
        print(f"  Star hints (month deviation): {stars}")
    print(f"  Seasonal weight: {weight} (added to fusion score; soft bias only)\n")


def print_seasonal_fusion_report(out: dict) -> None:
    print("=" * 68)
    print("SEASONAL FUSION — random pool + history rank + month/half bias")
    print("=" * 68)
    print(f"\n{out['disclaimer']}\n")
    meta = out["fusion_meta"]
    sc = out.get("seasonal_context", {})
    print(f"Draws in history:     {out['draws_analyzed']}")
    print(f"Play date window:     {out.get('play_date', '?')}  ({sc.get('month_name')} {sc.get('half')})")
    print(f"Random pool size:     {meta['pool_generated']:,} fair tickets")
    print(f"Pool attempts:        {meta['pool_attempts']:,}")
    print(
        f"Frequency window:     {lookback_label(meta['lookback'], out['draws_analyzed'])}  |  "
        f"decay: {meta['decay']}  |  seasonal weight: {meta.get('seasonal_weight', 1.0)}"
    )
    print(f"Seed:                 {meta['seed']}")
    if meta.get("flat_stars"):
        print("Stars:                flat scores (history-neutral)")
    if meta.get("exclude_stars"):
        print(f"Excluded stars:       {meta['exclude_stars']}")

    print("\n--- How this differs ---")
    for line in out["unique_approach"]:
        print(f"  • {line}")

    tiers = {**out.get("main_tiers", {}), **out.get("star_tiers", {})}
    play = out.get("play_line")
    src = out.get("ticket_from", "random")
    if play:
        print_play_line(play, tiers, src)
    if src != "consensus" and out.get("consensus_line"):
        print("\n--- CONSENSUS (vote across top lines; not used as play line) ---")
        print_consensus_line(out["consensus_line"], tiers)

    print("\n--- TOP FUSION LINES (random-born, history + seasonal rank) ---")
    for line in out["fusion_lines"]:
        m = ", ".join(f"{n:02d}" for n in line["main_numbers"])
        s = ", ".join(f"{n:02d}" for n in line["lucky_stars"])
        seasonal = line.get("breakdown", {}).get("seasonal", {}).get("seasonal_total", 0)
        print(
            f"\n  #{line['rank']}  fusion={line['score']:.1f}  seasonal+{seasonal:.1f}  "
            f"sum={line['sum']}  tiers={line.get('breakdown', {}).get('tiers', {})}"
        )
        print(f"      Main:  {m}")
        print(f"      Stars: {s}")

    print("\n" + "=" * 68)


def _seasonal_breakdown_for_play(out: dict) -> dict | None:
    play = out.get("play_line") or {}
    pm = tuple(play.get("main_numbers") or [])
    ps = tuple(play.get("lucky_stars") or [])
    if not pm:
        return None
    for line in out.get("fusion_lines", []):
        if tuple(line.get("main_numbers") or []) == pm and tuple(line.get("lucky_stars") or []) == ps:
            return line.get("breakdown", {}).get("seasonal")
    return None


def print_seasonal_explanation_extra(out: dict) -> None:
    play = out.get("play_line") or {}
    meta = out.get("fusion_meta", {})
    sc = out.get("seasonal_context", {})
    print("--- SEASONAL FIT (this ticket) ---")
    print(
        f"  Window: {sc.get('month_name')} {sc.get('half')}  |  "
        f"weight={meta.get('seasonal_weight', 1.0)}"
    )
    b = _seasonal_breakdown_for_play(out)
    if not b and out.get("fusion_lines"):
        b = out["fusion_lines"][0].get("breakdown", {}).get("seasonal", {})
    if b:
        print(f"  Seasonal bonus: {b.get('seasonal_total', 0)}")
        for k, v in b.items():
            if k != "seasonal_total":
                print(f"    {k}: {v}")
    print()


def build_seasonal_output(
    draws: list[dict],
    seasonal_analysis: dict,
    play_when: date,
    analysis_path: Path,
    pool_size: int = 20000,
    lookback: int = 0,
    decay: float = 0.97,
    top_n: int = 10,
    seed: int | None = None,
    position_filter: bool = True,
    position_band: float = 70,
    ticket_from: str = "long-due",
    random_from: int = 10,
    flat_stars: bool = True,
    exclude_stars: set[int] | None = None,
    seasonal_weight: float = 1.0,
    long_due_gap: int = 15,
    long_due_min: int = 2,
    horoscope_name: str | None = None,
    horoscope_min_mains: int = 2,
    horoscope_min_stars: int = 1,
    horoscope_weight: float = 1.0,
    horoscope_live: bool = True,
    horoscope_place: str = "ireland",
    include_enhanced_reference: bool = False,
) -> dict:
    month, half, _ = play_calendar(play_when)
    seasonal_ctx = build_seasonal_context(seasonal_analysis, month, half)
    horoscope_profile = resolve_horoscope_profile(
        horoscope_name,
        play_when,
        live=horoscope_live,
        place=horoscope_place,
    )

    use_long_due = ticket_from == "long-due"
    fusion_candidates, ctx, pool_attempts, position_limits, main_gaps = build_seasonal_fusion_pool(
        draws,
        pool_size,
        lookback,
        decay,
        seasonal_ctx,
        seasonal_weight,
        position_filter=position_filter,
        position_band=position_band,
        flat_stars=flat_stars,
        exclude_stars=exclude_stars,
        long_due_gap=long_due_gap if use_long_due else 0,
        long_due_min=long_due_min if use_long_due else 0,
        horoscope_profile=horoscope_profile,
        horoscope_min_mains=horoscope_min_mains,
        horoscope_min_stars=horoscope_min_stars,
        horoscope_weight=horoscope_weight,
    )
    fusion_lines = select_diverse_lines(fusion_candidates, top_n, max_per_star_pair=2)

    sum_lo, sum_hi, _ = compute_sum_band(draws)
    consensus = build_consensus_line(
        fusion_lines,
        ctx["main_tiers"],
        ctx["star_tiers"],
        (sum_lo, sum_hi),
        position_limits=position_limits,
    )
    play_line = resolve_play_ticket(
        fusion_lines,
        consensus,
        ticket_from,
        random_from,
        seed,
        fusion_candidates=fusion_candidates
        if ticket_from in ("pool", "long-due")
        else None,
        main_gaps=main_gaps,
        long_due_gap=long_due_gap if use_long_due else 0,
    )
    if play_line and horoscope_profile:
        horo = (play_line.get("breakdown") or {}).get("horoscope")
        if not horo:
            _, horo = score_horoscope_fit(
                play_line["main_numbers"],
                play_line["lucky_stars"],
                horoscope_profile,
                main_gaps=main_gaps,
                main_tiers=ctx["main_tiers"],
                long_due_gap=long_due_gap if use_long_due else 15,
                weight=horoscope_weight,
            )
        play_line["horoscope"] = horo
    position_ranges = (
        analyze_sorted_positions(draws, band_pct=position_band) if position_filter else None
    )

    horo_filter = ""
    if horoscope_profile:
        horo_filter = (
            f", horoscope {horoscope_name} "
            f"(≥{horoscope_min_mains} mains, ≥{horoscope_min_stars} stars)"
        )

    return {
        "mode": "seasonal_fusion",
        "ticket_from": ticket_from,
        "random_from": random_from,
        "play_line": play_line,
        "play_date": play_when.isoformat(),
        "seasonal_context": seasonal_ctx,
        "seasonal_analysis_file": str(analysis_path),
        "horoscope": {
            "profile": horoscope_name,
            "label": horoscope_profile.get("label") if horoscope_profile else None,
            "live": bool(horoscope_profile.get("live")) if horoscope_profile else False,
            "place": horoscope_place if horoscope_profile else None,
            "mains": list(horoscope_profile["mains"]) if horoscope_profile else [],
            "stars": list(horoscope_profile["stars"]) if horoscope_profile else [],
            "live_mains": list(horoscope_profile.get("live_mains") or [])
            if horoscope_profile
            else [],
            "live_stars": list(horoscope_profile.get("live_stars") or [])
            if horoscope_profile
            else [],
            "overlap_mains": list(horoscope_profile.get("overlap_mains") or [])
            if horoscope_profile
            else [],
            "overlap_stars": list(horoscope_profile.get("overlap_stars") or [])
            if horoscope_profile
            else [],
            "transit": horoscope_profile.get("transit") if horoscope_profile else None,
            "judgment": horoscope_profile.get("judgment") if horoscope_profile else None,
            "min_mains": horoscope_min_mains if horoscope_profile else None,
            "min_stars": horoscope_min_stars if horoscope_profile else None,
            "weight": horoscope_weight if horoscope_profile else None,
        },
        "position_filter": position_filter,
        "position_band": position_band,
        "position_limits": position_limits,
        "sorted_position_ranges": position_ranges,
        "sum_band": {"min": sum_lo, "max": sum_hi},
        "disclaimer": (
            "Seasonal fusion adds a soft month/half bias from seasonal_analysis.json. "
            "Horoscope alignment is traditional preference only. "
            "Draws remain random; this does not improve jackpot odds."
        ),
        "draws_analyzed": len(draws),
        "fusion_meta": {
            "pool_generated": len(fusion_candidates),
            "pool_attempts": pool_attempts,
            "pool_target": pool_size,
            "lookback": lookback,
            "decay": decay,
            "seed": seed,
            "flat_stars": flat_stars,
            "exclude_stars": sorted(exclude_stars) if exclude_stars else [],
            "seasonal_weight": seasonal_weight,
            "seasonal_month": month,
            "seasonal_half": half,
            "long_due_gap": long_due_gap if use_long_due else None,
            "long_due_min": long_due_min if use_long_due else None,
            "horoscope": horoscope_name,
            "horoscope_live": horoscope_live if horoscope_profile else None,
            "horoscope_place": horoscope_place if horoscope_profile else None,
            "horoscope_weight": horoscope_weight if horoscope_profile else None,
        },
        "unique_approach": [
            "Random: fair mechanical draw (secrets)",
            (
                f"Filter: sum, parity, high/low, cold"
                + (
                    f", position bands (~{position_band:.0f}%)"
                    if position_filter
                    else " (no position bands)"
                )
                + (
                    f", ≥{long_due_min} long-due mains (≥{long_due_gap} draws absent)"
                    if use_long_due
                    else ""
                )
                + horo_filter
            ),
            (
                "Rank: enhanced history + seasonal + horoscope "
                "(aligned ∩ overdue/cold get extra weight; reference only)"
                if horoscope_profile
                else "Rank: enhanced history score + seasonal month/half bonus (reference only)"
            ),
            "Stars: flat scoring by default (more variety)",
            (
                f"Pick: dead random from {pool_size:,} valid tickets with long-due mains"
                if use_long_due
                else (
                    f"Pick: uniform random from all {pool_size:,} valid pool tickets (score-neutral)"
                    if ticket_from == "pool"
                    else f"Pick: {ticket_from} from top {random_from} ranked lines"
                )
            ),
        ],
        "fusion_lines": fusion_lines,
        "top_suggested_lines": fusion_lines,
        "consensus_line": consensus,
        "top_15_main_numbers_by_score": ctx["main"][:15],
        "top_lucky_stars_by_score": ctx["stars"][:6],
        "all_main_ranked": ctx["main"],
        "all_stars_ranked": ctx["stars"],
        "main_tiers": ctx["main_tiers"],
        "star_tiers": ctx["star_tiers"],
    }


def run_morning(args: argparse.Namespace) -> None:
    lot_dir = Path(__file__).parent
    quiet = args.quiet or not args.verbose
    update_cmd = [sys.executable, str(lot_dir / "update_results.py")]
    if quiet:
        update_cmd.append("--quiet")
    if not quiet:
        print("Updating latest results...")
    subprocess.run(update_cmd, cwd=lot_dir, check=True)
    ensure_seasonal_analysis(args.seasonal_file, args.refresh_seasonal, quiet=quiet)
    analysis = load_seasonal_analysis(args.seasonal_file)
    draws = load_draws(args.input)
    play_when = parse_play_date(args.play_date) or date.today()

    out = build_seasonal_output(
        draws,
        analysis,
        play_when,
        args.seasonal_file,
        pool_size=args.pool,
        lookback=args.lookback,
        decay=args.decay,
        top_n=args.lines,
        seed=args.seed,
        position_filter=not args.no_position_filter,
        position_band=args.position_band,
        ticket_from=args.ticket_from,
        random_from=args.random_from,
        flat_stars=not args.no_flat_stars,
        exclude_stars=parse_exclude_stars(args.exclude_stars),
        seasonal_weight=args.seasonal_weight,
        long_due_gap=args.long_due_gap,
        long_due_min=args.long_due_min,
        horoscope_name=args.horoscope,
        horoscope_min_mains=args.horoscope_min_mains,
        horoscope_min_stars=args.horoscope_min_stars,
        horoscope_weight=args.horoscope_weight,
        horoscope_live=not args.no_live_horoscope,
        horoscope_place=args.horoscope_place,
    )
    if quiet:
        print_compact_ticket(out.get("play_line"))
    else:
        print_seasonal_banner(out["seasonal_context"], args.seasonal_weight, args.seasonal_file)
        if out.get("horoscope", {}).get("profile"):
            print_horoscope_banner(out["horoscope"])
        latest = draws[0]["date"] if draws else "?"
        print_morning_ticket(out.get("play_line"), len(draws), latest)
    if args.explain and not quiet:
        print_morning_explanation(out)
        print_seasonal_explanation_extra(out)
    if not args.no_save:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        if not quiet:
            print(f"Full details saved: {args.output}")


def print_horoscope_banner(horo: dict) -> None:
    print("--- HOROSCOPE ALIGN ---")
    print(f"  Profile: {horo.get('label') or horo.get('profile')}")
    live = horo.get("live")
    print(f"  Live sky: {'yes' if live else 'no (static birth numbers only)'}")
    if horo.get("place"):
        print(f"  Place:    {horo['place']}")
    transit = horo.get("transit") or {}
    if transit:
        nak = transit.get("moon_nakshatra") or {}
        print(
            f"  When:     {transit.get('when_local')}  "
            f"({transit.get('place')})"
        )
        print(
            f"  Moon:     {nak.get('name')} pada {nak.get('pada')} "
            f"(lord {nak.get('lord')}, # {nak.get('planet_number')})  "
            f"rashi { (transit.get('moon_rashi') or {}).get('name') }"
        )
        print(
            f"  Day lord: {transit.get('day_lord')} (#{transit.get('day_number')})  |  "
            f"Jupiter in {(transit.get('jupiter_rashi') or {}).get('name')}  |  "
            f"Ketu in {(transit.get('ketu_rashi') or {}).get('name')}"
        )
    judgment = horo.get("judgment") or {}
    if judgment:
        print(f"  Judgment: {judgment.get('tone')} (score {judgment.get('score')})")
        for note in (judgment.get("notes") or [])[:4]:
            print(f"    • {note}")
    mains = ", ".join(f"{n:02d}" for n in horo.get("mains") or [])
    stars = ", ".join(f"{n:02d}" for n in horo.get("stars") or [])
    print(f"  Filter mains: {mains}")
    print(f"  Filter stars: {stars}")
    if live:
        om = ", ".join(f"{n:02d}" for n in horo.get("overlap_mains") or [])
        os_ = ", ".join(f"{n:02d}" for n in horo.get("overlap_stars") or [])
        print(f"  Birth∩today mains: [{om}]  stars: [{os_}]")
    print(
        f"  Require: ≥{horo.get('min_mains')} aligned mains, "
        f"≥{horo.get('min_stars')} aligned stars  |  weight={horo.get('weight')}"
    )
    print()


def main() -> None:
    lot_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Seasonal fusion: hybrid_draw + seasonal_analysis.json bias"
    )
    parser.add_argument("--morning", action="store_true", help="Update JSON + seasonal file, then run (quiet ticket only)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="With --morning: show full report (default is ticket only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print ticket only (also default with --morning)",
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=lot_dir / "euromillions_last_5_years.json"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=lot_dir / "seasonal_fusion_predictions.json"
    )
    parser.add_argument(
        "--seasonal-file",
        type=Path,
        default=lot_dir / "seasonal_analysis.json",
    )
    parser.add_argument(
        "--refresh-seasonal",
        action="store_true",
        help="Rebuild seasonal_analysis.json before running",
    )
    parser.add_argument("--pool", type=int, default=20000)
    parser.add_argument("--lookback", type=int, default=0)
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("-n", "--lines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-position-filter", action="store_true")
    parser.add_argument("--position-band", type=float, default=70)
    parser.add_argument(
        "--ticket-from",
        choices=("consensus", "top", "random", "pool", "long-due"),
        default="long-due",
        help="long-due=random among valid tickets with overdue mains (default); pool=any valid ticket",
    )
    parser.add_argument("--random-from", type=int, default=10)
    parser.add_argument(
        "--long-due-gap",
        type=int,
        default=15,
        help="Mains absent at least this many draws count as long-due (default 15)",
    )
    parser.add_argument(
        "--long-due-min",
        type=int,
        default=2,
        help="Require at least this many long-due mains per ticket (default 2)",
    )
    parser.add_argument(
        "--no-flat-stars",
        action="store_true",
        help="Use history-weighted star scores (default: flat stars for variety)",
    )
    parser.add_argument("--exclude-stars", default=None)
    parser.add_argument(
        "--seasonal-weight",
        type=float,
        default=1.0,
        help="Multiplier on seasonal bonus points (default 1.0)",
    )
    parser.add_argument(
        "--play-date",
        default=None,
        help="Simulate seasonal window for date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--horoscope",
        default=None,
        metavar="PROFILE",
        help=(
            "Align tickets to traditional lucky numbers. "
            f"Profiles: {', '.join(list_profiles())}"
        ),
    )
    parser.add_argument(
        "--horoscope-min-mains",
        type=int,
        default=2,
        help="Min aligned main numbers required when --horoscope is set (default 2)",
    )
    parser.add_argument(
        "--horoscope-min-stars",
        type=int,
        default=1,
        help="Min aligned lucky stars required when --horoscope is set (default 1)",
    )
    parser.add_argument(
        "--horoscope-weight",
        type=float,
        default=1.0,
        help="Multiplier on horoscope bonus (extra for aligned ∩ overdue/cold)",
    )
    parser.add_argument(
        "--no-live-horoscope",
        action="store_true",
        help="Use static birth lucky numbers only (skip today's Moon/planets)",
    )
    parser.add_argument(
        "--horoscope-place",
        default="ireland",
        help=f"Place for local sky time. Choices: {', '.join(list_places())} (default ireland)",
    )
    args = parser.parse_args()

    if not 0 < args.position_band <= 100:
        raise SystemExit("--position-band must be > 0 and <= 100")
    if args.long_due_gap < 1:
        raise SystemExit("--long-due-gap must be >= 1")
    if args.long_due_min < 1 or args.long_due_min > 5:
        raise SystemExit("--long-due-min must be between 1 and 5")
    if args.horoscope_min_mains < 0 or args.horoscope_min_mains > 5:
        raise SystemExit("--horoscope-min-mains must be between 0 and 5")
    if args.horoscope_min_stars < 0 or args.horoscope_min_stars > 2:
        raise SystemExit("--horoscope-min-stars must be between 0 and 2")
    if args.horoscope:
        get_profile(args.horoscope)  # validate early

    if args.morning and not args.verbose:
        args.quiet = True
        args.explain = False

    if args.morning:
        run_morning(args)
        return

    ensure_seasonal_analysis(args.seasonal_file, args.refresh_seasonal, quiet=args.quiet)
    analysis = load_seasonal_analysis(args.seasonal_file)
    draws = load_draws(args.input)
    play_when = parse_play_date(args.play_date) or date.today()
    out = build_seasonal_output(
        draws,
        analysis,
        play_when,
        args.seasonal_file,
        pool_size=args.pool,
        lookback=args.lookback,
        decay=args.decay,
        top_n=args.lines,
        seed=args.seed,
        position_filter=not args.no_position_filter,
        position_band=args.position_band,
        ticket_from=args.ticket_from,
        random_from=args.random_from,
        flat_stars=not args.no_flat_stars,
        exclude_stars=parse_exclude_stars(args.exclude_stars),
        seasonal_weight=args.seasonal_weight,
        long_due_gap=args.long_due_gap,
        long_due_min=args.long_due_min,
        horoscope_name=args.horoscope,
        horoscope_min_mains=args.horoscope_min_mains,
        horoscope_min_stars=args.horoscope_min_stars,
        horoscope_weight=args.horoscope_weight,
        horoscope_live=not args.no_live_horoscope,
        horoscope_place=args.horoscope_place,
        include_enhanced_reference=False,
    )
    if args.quiet:
        print_compact_ticket(out.get("play_line"))
    else:
        print_seasonal_banner(out["seasonal_context"], args.seasonal_weight, args.seasonal_file)
        if out.get("horoscope", {}).get("profile"):
            print_horoscope_banner(out["horoscope"])
        print_seasonal_fusion_report(out)
        if args.explain and out.get("play_line"):
            print_morning_explanation(out)
            print_seasonal_explanation_extra(out)
    if not args.no_save:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
