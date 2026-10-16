"""Load seasonal_analysis.json and score tickets against current month/half patterns."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from generate_predictions import LOW_CUTOFF, STAR_MAX, STAR_MIN

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def play_calendar(when: date | None = None) -> tuple[int, str, str]:
    d = when or date.today()
    half = "H1" if d.day <= 15 else "H2"
    return d.month, half, MONTH_NAMES[d.month]


def load_seasonal_analysis(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python3 analyze_seasonal.py -o {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def line_shape_features(main: list[int], stars: list[int]) -> dict:
    nums = sorted(main)
    stars = sorted(stars)
    odd = sum(1 for n in nums if n % 2)
    low = sum(1 for n in nums if n <= LOW_CUTOFF)
    feats: dict = {
        "sum": sum(nums),
        "odd_count": odd,
        "low_count": low,
        "parity_sig": f"{odd}O{5 - odd}E",
        "highlow_sig": f"{low}L{5 - low}H",
        "size_sig": f"{low}L{5 - low}H_{odd}O{5 - odd}E",
        "first_digit_single": nums[0] <= 9,
        "star_pair": f"{stars[0]:02d}-{stars[1]:02d}",
    }
    for s in range(STAR_MIN, STAR_MAX + 1):
        feats[f"star_{s}"] = int(s in stars)
    return feats


def _feature_matches(feats: dict, feature: str, value: str) -> bool:
    if feature.startswith("star_") and feature != "star_pair":
        try:
            star_num = int(feature.split("_", 1)[1])
        except (IndexError, ValueError):
            return False
        want = value.lower() in ("1", "true", "yes")
        return bool(feats.get(f"star_{star_num}", 0)) == want
    if feature not in feats:
        return False
    actual = feats[feature]
    if isinstance(actual, bool):
        return actual == (value.lower() in ("true", "1", "yes"))
    return str(actual) == value


def build_seasonal_context(analysis: dict, month: int, half: str) -> dict:
    profile = next((p for p in analysis.get("monthly_profiles", []) if p["month"] == month), None)
    survivors: list[dict] = []
    for row in analysis.get("top_survivors_bh", []):
        seg_type = row.get("segment_type", "")
        seg = str(row.get("segment", ""))
        if seg_type == "month" and seg == str(month):
            survivors.append(row)
        elif seg_type == "half" and seg == half:
            survivors.append(row)
        elif seg_type == "month_half" and seg == f"{month}_{half}":
            survivors.append(row)

    star_hints = [
        x for x in analysis.get("star_seasonality_top", []) if x.get("month") == month
    ][:4]

    mi = {r["feature"]: r for r in analysis.get("mutual_info", [])}

    return {
        "month": month,
        "half": half,
        "month_name": MONTH_NAMES[month],
        "profile": profile,
        "survivors": survivors,
        "star_hints": star_hints,
        "mutual_info": mi,
        "hypotheses_tested": analysis.get("hypotheses_tested", 0),
        "survivors_bh_total": analysis.get("survivors_bh", 0),
    }


def score_seasonal_fit(
    main: list[int],
    stars: list[int],
    ctx: dict,
    weight: float = 1.0,
) -> tuple[float, dict]:
    """Bonus points for matching this month/half seasonal profile (soft, not a filter)."""
    feats = line_shape_features(main, stars)
    pts = 0.0
    breakdown: dict = {}

    profile = ctx.get("profile")
    if profile:
        mean_sum = profile.get("mean_sum", 127)
        half_width = max(12.0, abs(mean_sum - 127) + 8)
        dist = abs(feats["sum"] - mean_sum) / half_width
        sum_pts = max(0.0, 10.0 - dist * 10.0)
        pts += sum_pts
        breakdown["seasonal_sum"] = round(sum_pts, 2)

        preferred_low = 2 if profile.get("pct_2low", 0.33) >= 0.33 else 3
        if feats["low_count"] == preferred_low:
            pts += 5.0
            breakdown["seasonal_low"] = 5.0

        want_first_single = profile.get("pct_first_single", 0.5) >= 0.5
        if feats["first_digit_single"] == want_first_single:
            pts += 4.0
            breakdown["seasonal_first_single"] = 4.0

        preferred_odd = 3 if profile.get("pct_3odd", 0.33) >= 0.33 else 2
        if feats["odd_count"] == preferred_odd:
            pts += 3.0
            breakdown["seasonal_parity"] = 3.0

    matches = 0
    for surv in ctx.get("survivors", []):
        if _feature_matches(feats, surv["feature"], str(surv["value"])):
            matches += 1
    surv_pts = min(matches * 4.0, 20.0)
    pts += surv_pts
    breakdown["seasonal_survivor_matches"] = matches
    breakdown["seasonal_survivors"] = round(surv_pts, 2)

    for hint in ctx.get("star_hints", []):
        star_num = hint.get("star")
        if star_num and feats.get(f"star_{star_num}"):
            pts += 2.0
            breakdown.setdefault("seasonal_star_hints", []).append(star_num)

    total = round(pts * weight, 2)
    breakdown["seasonal_total"] = total
    return total, breakdown


def parse_play_date(raw: str | None) -> date | None:
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%d").date()
