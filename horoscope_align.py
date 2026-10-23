"""
Sagittarius + Moola alignment with optional live Vedic transit overlay.

Birth-profile numbers stay fixed; when live=True they merge with today's
Moon nakshatra / weekday / planet positions for the play place (Ireland).

EuroMillions odds are unchanged — soft preference only.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from vedic_transit import (
    compute_sky,
    judge_for_moola_sagittarius,
    live_number_sets,
    resolve_place,
)

# Sagittarius (Jupiter): 3, 9; Western lucky extras 12, 21
# Moola / Mula (Ketu): primary 7; potency 19; common compounds
# Stars must be 1–12 — 3, 7, 9, 12 sit in both systems.
PROFILES: dict[str, dict[str, Any]] = {
    "sagittarius-moola": {
        "label": "Sagittarius + Moola (Ketu)",
        "mains": (3, 7, 9, 12, 16, 19, 21, 25, 27, 30, 34, 37, 39, 43, 48),
        "stars": (3, 7, 9, 12),
        "core_mains": (3, 7, 9, 12, 19, 21),
        "notes": (
            "Sagittarius/Jupiter favour 3 & 9; Moola/Ketu favour 7 "
            "(also 19 potency). Compounds kept in 1–50 range."
        ),
        "birth_theme": "moola_sagittarius",
    },
}

DEFAULT_PROFILE = "sagittarius-moola"


def list_profiles() -> list[str]:
    return sorted(PROFILES)


def get_profile(name: str | None) -> dict[str, Any] | None:
    """Static birth profile only (no live sky)."""
    if not name:
        return None
    key = name.strip().lower().replace("_", "-")
    if key not in PROFILES:
        known = ", ".join(list_profiles())
        raise SystemExit(f"Unknown --horoscope '{name}'. Choose: {known}")
    return dict(PROFILES[key])


def resolve_horoscope_profile(
    name: str | None,
    play_when: date | None = None,
    *,
    live: bool = True,
    place: str = "ireland",
) -> dict[str, Any] | None:
    """
    Birth profile, optionally merged with today's transit numbers for `place`.
    """
    base = get_profile(name)
    if base is None:
        return None

    profile = dict(base)
    profile["mains"] = list(base["mains"])
    profile["stars"] = list(base["stars"])
    profile["core_mains"] = list(base.get("core_mains") or ())
    profile["live"] = False
    profile["transit"] = None
    profile["judgment"] = None

    if not live:
        profile["label"] = f"{base['label']} (static)"
        return profile

    resolve_place(place)  # validate
    sky = compute_sky(play_when or date.today(), place_name=place)
    judgment = judge_for_moola_sagittarius(sky)
    live_nums = live_number_sets(sky, judgment)

    birth_mains = set(profile["mains"])
    birth_stars = set(profile["stars"])
    live_mains = set(live_nums["mains"])
    live_stars = set(live_nums["stars"])

    # Prefer intersection for filter core; union for scoring pool
    overlap_mains = sorted(birth_mains & live_mains)
    overlap_stars = sorted(birth_stars & live_stars)
    merged_mains = sorted(birth_mains | live_mains)
    merged_stars = sorted(birth_stars | live_stars)

    # Scoring uses full merge; filter prefers today's emphasis ∩ birth when possible
    filter_mains = overlap_mains if len(overlap_mains) >= 3 else merged_mains
    filter_stars = overlap_stars if len(overlap_stars) >= 1 else merged_stars

    profile["mains"] = filter_mains
    profile["stars"] = filter_stars
    profile["score_mains"] = merged_mains
    profile["score_stars"] = merged_stars
    profile["live_mains"] = sorted(live_mains)
    profile["live_stars"] = sorted(live_stars)
    profile["overlap_mains"] = overlap_mains
    profile["overlap_stars"] = overlap_stars
    profile["core_mains"] = sorted(
        set(profile["core_mains"]) | set(overlap_mains[:6])
    )
    profile["live"] = True
    profile["transit"] = sky.as_dict()
    profile["judgment"] = judgment
    profile["live_sources"] = live_nums["sources"]
    tone = judgment.get("tone", "neutral")
    nak = sky.moon_nakshatra["name"]
    profile["label"] = (
        f"{base['label']} + live {place.title()} "
        f"(Moon in {nak}, {tone})"
    )
    return profile


def line_meets_horoscope(
    main: list[int],
    stars: list[int],
    profile: dict[str, Any],
    min_mains: int = 2,
    min_stars: int = 1,
) -> bool:
    mains = set(profile["mains"])
    star_set = set(profile["stars"])
    return (
        sum(1 for n in main if n in mains) >= min_mains
        and sum(1 for s in stars if s in star_set) >= min_stars
    )


def score_horoscope_fit(
    main: list[int],
    stars: list[int],
    profile: dict[str, Any],
    main_gaps: dict[int, int] | None = None,
    main_tiers: dict[int, str] | None = None,
    long_due_gap: int = 15,
    weight: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Soft bonus: birth + live hits; extra for overlap / overdue / cold."""
    score_mains = set(profile.get("score_mains") or profile["mains"])
    score_stars = set(profile.get("score_stars") or profile["stars"])
    live_mains = set(profile.get("live_mains") or ())
    live_stars = set(profile.get("live_stars") or ())
    overlap_mains = set(profile.get("overlap_mains") or ())
    overlap_stars = set(profile.get("overlap_stars") or ())
    core = set(profile.get("core_mains") or ())
    main_gaps = main_gaps or {}
    main_tiers = main_tiers or {}

    hit_mains = sorted(n for n in main if n in score_mains)
    hit_stars = sorted(s for s in stars if s in score_stars)
    live_main_hits = [n for n in hit_mains if n in live_mains]
    live_star_hits = [s for s in hit_stars if s in live_stars]
    overlap_main_hits = [n for n in hit_mains if n in overlap_mains]
    overlap_star_hits = [s for s in hit_stars if s in overlap_stars]
    core_hits = [n for n in hit_mains if n in core]
    overdue = [
        {"number": n, "gap": main_gaps.get(n, 0)}
        for n in hit_mains
        if main_gaps.get(n, 0) >= long_due_gap
    ]
    cold_hits = [n for n in hit_mains if main_tiers.get(n) == "cold"]

    pts = 0.0
    pts += 3.0 * len(hit_mains)
    pts += 5.0 * len(hit_stars)
    pts += 4.0 * len(live_main_hits)  # today's sky
    pts += 5.0 * len(live_star_hits)
    pts += 6.0 * len(overlap_main_hits)  # birth ∩ today
    pts += 7.0 * len(overlap_star_hits)
    pts += 3.0 * len(core_hits)
    pts += 5.0 * len(overdue)
    pts += 2.5 * len(cold_hits)

    judgment = profile.get("judgment") or {}
    tone = judgment.get("tone")
    if tone == "favourable":
        pts *= 1.15
    elif tone == "cautious":
        pts *= 0.9

    total = round(pts * weight, 2)
    breakdown = {
        "profile": profile.get("label", ""),
        "horoscope_total": total,
        "aligned_mains": hit_mains,
        "aligned_stars": hit_stars,
        "live_mains": live_main_hits,
        "live_stars": live_star_hits,
        "overlap_mains": overlap_main_hits,
        "overlap_stars": overlap_star_hits,
        "core_mains": core_hits,
        "overdue_aligned": overdue,
        "cold_aligned": cold_hits,
        "live": bool(profile.get("live")),
        "judgment_tone": tone,
        "weight": weight,
    }
    return total, breakdown
