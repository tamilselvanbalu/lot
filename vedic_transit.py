"""
Live Vedic sky snapshot for a play date/place (default: Ireland / Dublin).

Uses Skyfield (JPL DE421) + Lahiri ayanamsa to compute:
  - Moon nakshatra + pada + Vimshottari lord
  - weekday (vara) lord
  - sidereal signs for Moon, Sun, Jupiter, Ketu
  - soft "judgment" relative to a Sagittarius / Moola birth profile

This is traditional preference only — it does not change lottery odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Dublin — EuroMillions / Ireland default
PLACES: dict[str, dict[str, Any]] = {
    "ireland": {
        "label": "Ireland (Dublin)",
        "lat": 53.3498,
        "lon": -6.2603,
        "tz": "Europe/Dublin",
    },
}

NAKSHATRAS: list[dict[str, Any]] = [
    {"name": "Ashwini", "lord": "Ketu", "number": 7},
    {"name": "Bharani", "lord": "Venus", "number": 6},
    {"name": "Krittika", "lord": "Sun", "number": 1},
    {"name": "Rohini", "lord": "Moon", "number": 2},
    {"name": "Mrigashira", "lord": "Mars", "number": 9},
    {"name": "Ardra", "lord": "Rahu", "number": 4},
    {"name": "Punarvasu", "lord": "Jupiter", "number": 3},
    {"name": "Pushya", "lord": "Saturn", "number": 8},
    {"name": "Ashlesha", "lord": "Mercury", "number": 5},
    {"name": "Magha", "lord": "Ketu", "number": 7},
    {"name": "Purva Phalguni", "lord": "Venus", "number": 6},
    {"name": "Uttara Phalguni", "lord": "Sun", "number": 1},
    {"name": "Hasta", "lord": "Moon", "number": 2},
    {"name": "Chitra", "lord": "Mars", "number": 9},
    {"name": "Swati", "lord": "Rahu", "number": 4},
    {"name": "Vishakha", "lord": "Jupiter", "number": 3},
    {"name": "Anuradha", "lord": "Saturn", "number": 8},
    {"name": "Jyeshtha", "lord": "Mercury", "number": 5},
    {"name": "Moola", "lord": "Ketu", "number": 7},
    {"name": "Purva Ashadha", "lord": "Venus", "number": 6},
    {"name": "Uttara Ashadha", "lord": "Sun", "number": 1},
    {"name": "Shravana", "lord": "Moon", "number": 2},
    {"name": "Dhanishta", "lord": "Mars", "number": 9},
    {"name": "Shatabhisha", "lord": "Rahu", "number": 4},
    {"name": "Purva Bhadrapada", "lord": "Jupiter", "number": 3},
    {"name": "Uttara Bhadrapada", "lord": "Saturn", "number": 8},
    {"name": "Revati", "lord": "Mercury", "number": 5},
]

RASHIS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Planetary numbers → EuroMillions candidates (1–50 / 1–12)
PLANET_DIGITS = {
    "Sun": 1,
    "Moon": 2,
    "Jupiter": 3,
    "Rahu": 4,
    "Mercury": 5,
    "Venus": 6,
    "Ketu": 7,
    "Saturn": 8,
    "Mars": 9,
}

WEEKDAY_LORD = {
    0: ("Moon", 2),      # Monday
    1: ("Mars", 9),      # Tuesday
    2: ("Mercury", 5),   # Wednesday
    3: ("Jupiter", 3),   # Thursday
    4: ("Venus", 6),     # Friday
    5: ("Saturn", 8),    # Saturday
    6: ("Sun", 1),       # Sunday
}

# Nakshatra lords friendly / neutral / challenging to Moola (Ketu) — soft judgment only
MOOLA_FRIENDLY_LORDS = {"Ketu", "Jupiter", "Mars", "Sun"}
MOOLA_NEUTRAL_LORDS = {"Mercury", "Venus", "Moon"}
MOOLA_TOUGH_LORDS = {"Saturn", "Rahu"}

NAK_SPAN = 360.0 / 27.0  # 13°20'


def list_places() -> list[str]:
    return sorted(PLACES)


def resolve_place(name: str | None) -> dict[str, Any]:
    key = (name or "ireland").strip().lower()
    if key not in PLACES:
        raise SystemExit(f"Unknown --horoscope-place '{name}'. Choose: {', '.join(list_places())}")
    return PLACES[key]


def julian_day(dt_utc: datetime) -> float:
    """Julian Day (UTC) — Meeus-style."""
    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day + (
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    ) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def lahiri_ayanamsa_deg(jd: float) -> float:
    """
    Lahiri (Chitrapaksha) ayanamsa approximation.
    Anchored near modern values (~24.2° in mid-2020s).
    """
    t = (jd - 2451545.0) / 36525.0  # centuries from J2000.0
    # Linear Lahiri-style drift (~50.29"/yr) from epoch value
    return 23.853582 + 1.39797148 * t


def _digit_sum_mod9(n: int) -> int:
    s = sum(int(c) for c in str(n))
    while s > 9:
        s = sum(int(c) for c in str(s))
    return s or 9


def numbers_from_digit(digit: int) -> tuple[list[int], list[int]]:
    """Expand a planetary digit into main (1–50) and star (1–12) candidates."""
    d = ((int(digit) - 1) % 9) + 1
    series = [d + 9 * k for k in range(0, 6) if 1 <= d + 9 * k <= 50]
    extras = [n for n in range(1, 51) if _digit_sum_mod9(n) == d and n not in series]
    mains = sorted(
        set(series + [n for n in extras if n in (16, 19, 25, 27, 28, 34, 36, 37, 45, 46, 48, 49)])
    )
    stars = sorted({s for s in series if 1 <= s <= 12} | ({d} if d <= 12 else set()))
    return mains, stars


@lru_cache(maxsize=1)
def _load_ephemeris():
    from skyfield.api import Loader

    lot_dir = Path(__file__).parent
    load = Loader(str(lot_dir / ".skyfield"))
    return load.timescale(), load("de421.bsp")


def _ecliptic_lon_deg(eph, ts, body_name: str, t) -> float:
    from skyfield.framelib import ecliptic_frame

    earth = eph["earth"]
    body = eph[body_name]
    _, lon, _ = earth.at(t).observe(body).apparent().frame_latlon(ecliptic_frame)
    return float(lon.degrees) % 360.0


def mean_node_longitude_deg(jd: float) -> float:
    """Mean lunar ascending node (Rahu) — Meeus approximation, degrees."""
    t = (jd - 2451545.0) / 36525.0
    # Meeus Astronomical Algorithms
    omega = (
        125.04452
        - 1934.136261 * t
        + 0.0020708 * t * t
        + t * t * t / 450000.0
    )
    return omega % 360.0


def nakshatra_from_sidereal(lon: float) -> dict[str, Any]:
    lon = lon % 360.0
    idx = int(lon // NAK_SPAN) % 27
    within = lon - idx * NAK_SPAN
    pada = int(within // (NAK_SPAN / 4.0)) + 1
    row = NAKSHATRAS[idx]
    return {
        "index": idx + 1,
        "name": row["name"],
        "lord": row["lord"],
        "planet_number": row["number"],
        "pada": pada,
        "longitude": round(lon, 4),
        "degrees_in_nakshatra": round(within, 4),
    }


def rashi_from_sidereal(lon: float) -> dict[str, Any]:
    lon = lon % 360.0
    idx = int(lon // 30) % 12
    return {
        "name": RASHIS[idx],
        "index": idx + 1,
        "longitude": round(lon, 4),
        "degrees_in_sign": round(lon % 30.0, 4),
    }


@dataclass
class SkySnapshot:
    when_local: datetime
    place: dict[str, Any]
    ayanamsa: float
    moon_nakshatra: dict[str, Any]
    moon_rashi: dict[str, Any]
    sun_rashi: dict[str, Any]
    jupiter_rashi: dict[str, Any]
    ketu_rashi: dict[str, Any]
    day_lord: str
    day_number: int
    bodies_sidereal: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "when_local": self.when_local.isoformat(timespec="minutes"),
            "place": self.place["label"],
            "timezone": self.place["tz"],
            "ayanamsa_lahiri": round(self.ayanamsa, 4),
            "moon_nakshatra": self.moon_nakshatra,
            "moon_rashi": self.moon_rashi,
            "sun_rashi": self.sun_rashi,
            "jupiter_rashi": self.jupiter_rashi,
            "ketu_rashi": self.ketu_rashi,
            "day_lord": self.day_lord,
            "day_number": self.day_number,
            "bodies_sidereal": {k: round(v, 4) for k, v in self.bodies_sidereal.items()},
        }


def local_play_datetime(play_when: date, place: dict[str, Any], hour: int = 19) -> datetime:
    """EuroMillions evening context in local place time (default 19:00)."""
    tz = ZoneInfo(place["tz"])
    return datetime.combine(play_when, time(hour, 0), tzinfo=tz)


def compute_sky(play_when: date | None = None, place_name: str = "ireland", hour: int = 19) -> SkySnapshot:
    place = resolve_place(place_name)
    when_local = local_play_datetime(play_when or date.today(), place, hour=hour)
    when_utc = when_local.astimezone(timezone.utc)

    ts, eph = _load_ephemeris()
    t = ts.from_datetime(when_utc)
    jd = julian_day(when_utc.replace(tzinfo=None))
    aya = lahiri_ayanamsa_deg(jd)

    moon_trop = _ecliptic_lon_deg(eph, ts, "moon", t)
    sun_trop = _ecliptic_lon_deg(eph, ts, "sun", t)
    jup_trop = _ecliptic_lon_deg(eph, ts, "jupiter barycenter", t)
    rahu = mean_node_longitude_deg(jd)
    # Convert Rahu tropical mean node → sidereal; Ketu opposite
    rahu_sid = (rahu - aya) % 360.0
    ketu_sid = (rahu_sid + 180.0) % 360.0

    moon_sid = (moon_trop - aya) % 360.0
    sun_sid = (sun_trop - aya) % 360.0
    jup_sid = (jup_trop - aya) % 360.0

    weekday = when_local.weekday()  # Mon=0
    day_lord, day_number = WEEKDAY_LORD[weekday]

    return SkySnapshot(
        when_local=when_local,
        place=place,
        ayanamsa=aya,
        moon_nakshatra=nakshatra_from_sidereal(moon_sid),
        moon_rashi=rashi_from_sidereal(moon_sid),
        sun_rashi=rashi_from_sidereal(sun_sid),
        jupiter_rashi=rashi_from_sidereal(jup_sid),
        ketu_rashi=rashi_from_sidereal(ketu_sid),
        day_lord=day_lord,
        day_number=day_number,
        bodies_sidereal={
            "Moon": moon_sid,
            "Sun": sun_sid,
            "Jupiter": jup_sid,
            "Rahu": rahu_sid,
            "Ketu": ketu_sid,
        },
    )


def judge_for_moola_sagittarius(sky: SkySnapshot) -> dict[str, Any]:
    """Soft daily judgment vs Sagittarius (Jupiter) + Moola (Ketu) birth theme."""
    nak = sky.moon_nakshatra
    lord = nak["lord"]
    notes: list[str] = []
    score = 0  # -2 .. +3 style

    if nak["name"] == "Moola":
        notes.append("Moon transits Moola — strongest match to birth star")
        score += 3
    elif lord == "Ketu":
        notes.append(f"Moon in {nak['name']} (Ketu lord) — supportive for Moola natives")
        score += 2
    elif lord == "Jupiter":
        notes.append(f"Moon in {nak['name']} (Jupiter lord) — supportive for Sagittarius")
        score += 2
    elif lord in MOOLA_FRIENDLY_LORDS:
        notes.append(f"Moon lord {lord} is generally workable for Moola/Sagittarius")
        score += 1
    elif lord in MOOLA_TOUGH_LORDS:
        notes.append(f"Moon lord {lord} is traditionally more demanding for Moola")
        score -= 1
    else:
        notes.append(f"Moon in {nak['name']} ({lord}) — neutral for Moola/Sagittarius")

    if sky.moon_rashi["name"] == "Sagittarius":
        notes.append("Moon in sidereal Sagittarius — reinforces birth rashi theme")
        score += 1
    if sky.jupiter_rashi["name"] == "Sagittarius":
        notes.append("Jupiter in Sagittarius — classic boost for Dhanu themes")
        score += 1
    if sky.ketu_rashi["name"] == "Sagittarius":
        notes.append("Ketu in Sagittarius — highlights Moola/Dhanu axis")
        score += 1
    if sky.day_lord in ("Jupiter", "Ketu"):
        notes.append(f"Weekday lord {sky.day_lord} favours Sagittarius/Moola numbers")
        score += 1
    elif sky.day_lord == "Saturn":
        notes.append("Saturday (Saturn) — slower, more cautious tone")
        score -= 1

    if score >= 3:
        tone = "favourable"
    elif score >= 1:
        tone = "mixed-positive"
    elif score >= 0:
        tone = "neutral"
    else:
        tone = "cautious"

    return {
        "tone": tone,
        "score": score,
        "notes": notes,
        "focus_planets": sorted(
            {
                "Ketu",
                "Jupiter",
                nak["lord"],
                sky.day_lord,
            }
        ),
    }


def live_number_sets(sky: SkySnapshot, judgment: dict[str, Any]) -> dict[str, Any]:
    """Today's planetary digits → mains/stars, weighted by judgment focus."""
    focus = judgment.get("focus_planets") or []
    mains: set[int] = set()
    stars: set[int] = set()
    sources: dict[str, dict[str, list[int]]] = {}

    def add(label: str, digit: int, weight: str = "live") -> None:
        m, s = numbers_from_digit(digit)
        sources[label] = {"digit": digit, "mains": m, "stars": s, "weight": weight}
        mains.update(m)
        stars.update(s)

    add(f"moon_nakshatra:{sky.moon_nakshatra['name']}", sky.moon_nakshatra["planet_number"], "primary")
    add(f"weekday:{sky.day_lord}", sky.day_number, "primary")
    for planet in focus:
        digit = PLANET_DIGITS.get(planet)
        if digit:
            add(f"focus:{planet}", digit, "focus")

    # Mild boost digits when tone is favourable: reinforce Jupiter+Ketu
    if judgment.get("tone") in ("favourable", "mixed-positive"):
        add("tone:Jupiter", 3, "tone")
        add("tone:Ketu", 7, "tone")

    return {
        "mains": sorted(mains),
        "stars": sorted(s for s in stars if 1 <= s <= 12),
        "sources": sources,
    }
