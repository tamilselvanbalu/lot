#!/usr/bin/env python3
"""
Fetch latest EuroMillions results and update euromillions_last_5_years.json.

File name is historical; default window is **10 years** of draws.

Examples:
  python3 update_results.py                    # merge new draws (recent years)
  python3 update_results.py --full             # refetch all years in the window
  python3 update_results.py --full --years 10  # first-time 10-year build
  python3 update_results.py --dry-run          # preview only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

API_YEAR = "https://api-euromillions.nunofcguerreiro.com/results/year/{year}?format=json"
SOURCE = "https://api-euromillions.nunofcguerreiro.com"


def fetch_year_curl(year: int) -> list[dict]:
    url = API_YEAR.format(year=year)
    proc = subprocess.run(
        ["curl", "-sL", url],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed for {year}: {proc.stderr}")
    raw = json.loads(proc.stdout)
    draws = []
    for item in raw:
        draws.append(
            {
                "date": item["date"],
                "numbers": sorted(item["numbers"]),
                "stars": sorted(item["stars"]),
            }
        )
    return draws


def years_for_window(years_back: int) -> list[int]:
    today = date.today()
    return list(range(today.year - years_back, today.year + 1))


def merge_draws(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_date: dict[str, dict] = {}
    for d in existing + incoming:
        by_date[d["date"]] = d
    merged = sorted(by_date.values(), key=lambda x: x["date"], reverse=True)
    return merged


def filter_from_date(draws: list[dict], from_date: date) -> list[dict]:
    cutoff = from_date.isoformat()
    return [d for d in draws if d["date"] >= cutoff]


def build_document(draws: list[dict], years_back: int) -> dict:
    today = date.today()
    from_d = date(today.year - years_back, today.month, today.day)
    filtered = filter_from_date(draws, from_d)
    if not filtered:
        return {
            "game": "EuroMillions",
            "source": SOURCE,
            "period": {
                "from": from_d.isoformat(),
                "to": today.isoformat(),
                "description": f"Last {years_back} years of EuroMillions draws",
            },
            "total_draws": 0,
            "draws": [],
        }
    return {
        "game": "EuroMillions",
        "source": SOURCE,
        "period": {
            "from": filtered[-1]["date"],
            "to": filtered[0]["date"],
            "description": f"Last {years_back} years of EuroMillions draws",
        },
        "total_draws": len(filtered),
        "draws": filtered,
    }


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("draws", [])


def print_summary(path: Path, before: int, doc: dict, added_dates: list[str]) -> None:
    after = doc["total_draws"]
    print(f"File: {path}")
    print(f"Draws: {before} → {after}  (+{after - before} net)")
    print(f"Date range: {doc['period']['from']} → {doc['period']['to']}")
    if added_dates:
        print(f"New draw dates: {', '.join(added_dates[:10])}" + (" ..." if len(added_dates) > 10 else ""))
    else:
        print("No new draws (already up to date).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update EuroMillions results JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "euromillions_last_5_years.json",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Keep draws within this many years (default 10)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Refetch every year in the window (slower, complete refresh)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing the file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No stdout (for use from hybrid_seasonal_draw --morning)",
    )
    args = parser.parse_args()

    existing = load_existing(args.output)
    before_dates = {d["date"] for d in existing}

    if args.full:
        fetch_years = years_for_window(args.years)
    else:
        y = date.today().year
        fetch_years = [y, y - 1]

    incoming: list[dict] = []
    if not args.quiet:
        print(f"Fetching years: {fetch_years}")
    for year in fetch_years:
        try:
            batch = fetch_year_curl(year)
            incoming.extend(batch)
            if not args.quiet:
                print(f"  {year}: {len(batch)} draws")
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"  {year}: FAILED — {e}", file=sys.stderr)

    if not incoming and not existing:
        raise SystemExit("No data fetched and no existing file. Check network or API.")

    merged = merge_draws(existing, incoming)
    doc = build_document(merged, args.years)

    new_dates = [d["date"] for d in doc["draws"] if d["date"] not in before_dates]
    new_dates.sort(reverse=True)

    if not args.quiet:
        print_summary(args.output, len(existing), doc, new_dates)

    if args.dry_run:
        if not args.quiet:
            print("\n(dry-run — file not written)")
        return

    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"\nSaved: {args.output}")
        print("Run ticket: python3 hybrid_draw.py --morning --pool 50000")


if __name__ == "__main__":
    main()
