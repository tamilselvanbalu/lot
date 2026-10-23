# EuroMillions hybrid ticket tool

One workflow: **update history → random pool → filter → score → consensus line**.

No AI, no extra scripts — just the hybrid stack below.

Data: `euromillions_last_5_years.json` (~10 years; filename kept for compatibility).

---

## Daily play

```bash
cd /Users/tamilselvan.balu/code/lot
python3 hybrid_draw.py --morning --pool 50000
# If consensus always looks similar, use fusion #1 or random from top lines:
python3 hybrid_draw.py --morning --pool 50000 --ticket-from top
python3 hybrid_draw.py --morning --pool 50000 --flat-stars
python3 hybrid_draw.py --morning --pool 50000 --exclude-stars 2,10
python3 hybrid_draw.py --morning --pool 50000 --flat-stars --exclude-stars 2,10 --ticket-from random
```

Use the printed **Main** and **Stars**. Run **once** per draw day. `--ticket-from consensus` repeats similar balls (24, 06, 10) because history scoring is fixed until new results land.

Larger pool (slower, often slightly different consensus):

```bash
python3 hybrid_draw.py --morning --pool 500000
```

---

## What it does (short)

1. `update_results.py` merges latest draws from the API.
2. Generates `--pool` fair random tickets (like the machine).
3. Keeps tickets that match historical shape (sum, odd/even, high/low, ≥1 cold main).
4. Scores survivors against all draws in the JSON.
5. Takes top `--lines` lines and builds one **consensus** 5+2 from the most repeated numbers.

---

## Commands

### `hybrid_draw.py`

| Flag | Meaning |
|------|---------|
| `--morning` | Update JSON, then fusion; print consensus ticket |
| `--pool N` | Random tickets to generate (default 20000) |
| `--lines N` | Top lines kept (default 10) |
| `--lookback 0` | Score using all draws (default) |
| `--lookback 100` | Score using last 100 draws only |
| `--seed 42` | Repeatable random pool (omit for fresh picks) |
| `--no-explain` | Ticket only, no “why” section |
| `--no-save` | Do not write `fusion_predictions.json` |
| `--no-position-filter` | Skip sorted-slot bands (main pos 1–5, star pos 1–2) |
| `--position-band PCT` | How much history must fall inside each slot band (default **70**) |
| `--ticket-from` | `consensus` (default), `top` (#1 fusion line), or `random` (from top-N) |
| `--random-from N` | With `--ticket-from random`, pick from top N lines (default 5) |
| `--flat-stars` | Equal star scores (more star variety; stops 02/10 dominating) |
| `--exclude-stars LIST` | Reject tickets with these stars, e.g. `2,10` |
| `-o file.json` | Output path |

**Position filter (on by default):** After sorting mains (smallest→largest) and stars, each value must sit in that slot’s band from your JSON.

| `--position-band` | Meaning (example) |
|-------------------|-------------------|
| `70` (default) | 15th–85th percentile — tighter |
| `90` | 5th–95th percentile — wider, fewer rejects |
| `50` | 25th–75th percentile — very tight |

```bash
python3 hybrid_draw.py --morning --pool 500000 --position-band 90
python3 hybrid_draw.py --morning --pool 500000 --position-band 50
```

Bands are saved in `fusion_predictions.json` → `position_limits`, `position_band`, `sorted_position_ranges`.

```bash
python3 hybrid_draw.py --morning --pool 50000
python3 hybrid_draw.py --morning --no-explain
python3 hybrid_draw.py --pool 50000              # full report, no auto-update
python3 hybrid_draw.py --morning --no-save
```

### `update_results.py`

```bash
python3 update_results.py                    # merge latest (current + previous year)
python3 update_results.py --full             # refetch full window
python3 update_results.py --full --years 10  # rebuild ~10 years
python3 update_results.py --dry-run
```

### `analyze_seasonal.py` (pattern mining — optional)

ML-grade data prep (numpy/pandas/scikit-learn): builds a feature matrix per draw, auto-tests thousands of month/half-month/star hypotheses, and reports only patterns that survive statistical correction.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 analyze_seasonal.py                    # full report + seasonal_analysis.json
python3 analyze_seasonal.py --month 7          # July deep-dive (default)
python3 analyze_seasonal.py --shape-key cluster --clusters 8
python3 analyze_seasonal.py --dump-csv         # also writes seasonal_features.csv
```

**Note:** Surviving patterns measure deviation from chance, not predictive power. Expect few hits after Bonferroni correction.

### `hybrid_seasonal_draw.py` (fusion + seasonal bias)

Same random pool and filters as `hybrid_draw.py`, plus a soft month/half bonus from `seasonal_analysis.json`. Defaults: **long-due filter** (≥2 mains absent ≥15 draws) + **dead random pick** from that pool. `--morning` prints **ticket only**.

```bash
# Daily play — long-due + dead random (recommended)
python3 hybrid_seasonal_draw.py --morning --pool 50000 --no-position-filter

# Stricter overdue: 3 mains absent 20+ draws
python3 hybrid_seasonal_draw.py --morning --pool 50000 --long-due-gap 20 --long-due-min 3

# Pure random (no long-due filter)
python3 hybrid_seasonal_draw.py --morning --pool 50000 --ticket-from pool --no-position-filter
```

Output: `seasonal_fusion_predictions.json`. Flags: `--ticket-from long-due|pool|random|top`, `--long-due-gap`, `--long-due-min`, `--no-position-filter`, `--verbose`.

**Horoscope align (Sagittarius + Moola + live sky):** birth lucky numbers merged with **today’s Vedic transit** for Ireland (Moon nakshatra, weekday lord, Jupiter/Ketu signs via Skyfield + Lahiri). Extra weight when birth∩today numbers are also overdue/cold. Does not change jackpot odds.

```bash
# Needs venv (skyfield): python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python hybrid_seasonal_draw.py --morning --pool 50000 --no-position-filter \
  --ticket-from top --horoscope sagittarius-moola

# Static birth numbers only (no live planets)
.venv/bin/python hybrid_seasonal_draw.py --morning --pool 50000 --horoscope sagittarius-moola \
  --no-live-horoscope --ticket-from top --no-position-filter
```

Flags: `--horoscope-place ireland`, `--no-live-horoscope`, `--horoscope-min-mains`, `--horoscope-min-stars`, `--horoscope-weight`.

---

## Project layout

| File | Role |
|------|------|
| `hybrid_draw.py` | CLI — standard fusion draw |
| `hybrid_seasonal_draw.py` | CLI — fusion + seasonal_analysis bias |
| `horoscope_align.py` | Sagittarius/Moola + live transit merge |
| `vedic_transit.py` | Skyfield Lahiri Moon nakshatra / day lord |
| `seasonal_scoring.py` | Seasonal fit scoring (imported) |
| `update_results.py` | Fetch / merge draws |
| `generate_predictions.py` | Shared rules, consensus, tiers (imported) |
| `prediction_engine.py` | Enhanced scoring (imported) |
| `simulate_draw.py` | Fair random draw (imported) |
| `analyze_tier_mix.py` | Walk-forward hot/mild/cold mix check |
| `analyze_seasonal.py` | Month/half-month/run-length/star pattern mining |
| `seasonal_analysis.json` | Output from `analyze_seasonal.py` |
| `euromillions_last_5_years.json` | History |
| `fusion_predictions.json` | Last `hybrid_draw.py` output |
| `seasonal_fusion_predictions.json` | Last `hybrid_seasonal_draw.py` output |

---

## Requirements

- Python 3.10+
- `curl` (for `update_results.py`)
- Core scripts (`hybrid_draw.py`, etc.): **no pip packages**
- `analyze_seasonal.py`: `pip install -r requirements.txt` (numpy, pandas, scikit-learn) — use a venv:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

---

## Important

This does **not** improve jackpot odds. Each valid ticket is **1 in 139,838,160**. Play responsibly.
