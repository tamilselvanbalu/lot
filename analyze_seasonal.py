#!/usr/bin/env python3
"""
Seasonal / temporal pattern mining with ML-grade data preparation.

Builds a pandas feature matrix, auto-enumerates segment x feature hypotheses,
runs significance tests (with multiple-comparison correction), and reports
month / half-month / run-length / star seasonality.

DISCLAIMER: EuroMillions draws are independent. Surviving patterns are rare;
this tool measures evidence, not predictive edge.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from generate_predictions import LOW_CUTOFF, STAR_MAX, STAR_MIN, load_draws

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def consecutive_pairs(nums: list[int]) -> int:
    s = sorted(nums)
    return sum(1 for i in range(1, len(s)) if s[i] - s[i - 1] == 1)


def decade_bucket(n: int) -> str:
    if n <= 9:
        return "0-9"
    if n <= 19:
        return "10-19"
    if n <= 29:
        return "20-29"
    if n <= 39:
        return "30-39"
    return "40-49"


def build_dataframe(draws: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for d in draws:
        nums = sorted(d["numbers"])
        stars = sorted(d["stars"])
        dt = pd.Timestamp(d["date"])
        odd = sum(1 for n in nums if n % 2)
        low = sum(1 for n in nums if n <= LOW_CUTOFF)
        row: dict = {
            "date": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "half": "H1" if dt.day <= 15 else "H2",
            "weekday": dt.day_name()[:3],
            "week_of_month": (dt.day - 1) // 7 + 1,
            "day_of_year": dt.dayofyear,
            "n1": nums[0],
            "n2": nums[1],
            "n3": nums[2],
            "n4": nums[3],
            "n5": nums[4],
            "s1": stars[0],
            "s2": stars[1],
            "sum": sum(nums),
            "mean": sum(nums) / 5,
            "std": float(np.std(nums)),
            "odd_count": odd,
            "even_count": 5 - odd,
            "low_count": low,
            "high_count": 5 - low,
            "min": nums[0],
            "max": nums[-1],
            "spread": nums[-1] - nums[0],
            "first_digit_single": nums[0] <= 9,
            "consecutive_pairs": consecutive_pairs(nums),
            "parity_sig": f"{odd}O{5-odd}E",
            "highlow_sig": f"{low}L{5-low}H",
            "size_sig": f"{low}L{5-low}H_{odd}O{5-odd}E",
            "star_pair": f"{stars[0]:02d}-{stars[1]:02d}",
        }
        for dec in ("0-9", "10-19", "20-29", "30-39", "40-49"):
            row[f"dec_{dec.replace('-', '_')}"] = sum(1 for n in nums if decade_bucket(n) == dec)
        for n in range(1, 51):
            row[f"main_{n}"] = int(n in nums)
        for s in range(STAR_MIN, STAR_MAX + 1):
            row[f"star_{s}"] = int(s in stars)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["sum_roll5"] = df["sum"].rolling(5, min_periods=1).mean()
    return df


def segment_masks(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    masks: list[tuple[str, str, pd.Series]] = []
    for m in range(1, 13):
        masks.append(("month", str(m), df["month"] == m))
    for h in ("H1", "H2"):
        masks.append(("half", h, df["half"] == h))
    for m in range(1, 13):
        for h in ("H1", "H2"):
            masks.append(("month_half", f"{m}_{h}", (df["month"] == m) & (df["half"] == h)))
    for y in sorted(df["year"].unique()):
        masks.append(("year", str(y), df["year"] == y))
    return masks


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = np.zeros(n, dtype=bool)
    last = -1
    for i, (p, t) in enumerate(zip(ranked, thresh)):
        if p <= t:
            last = i
    if last >= 0:
        passed[order[: last + 1]] = True
    return passed


def chi2_cell_pvalue(observed: int, expected: float, n_trials: int) -> float:
    if expected <= 0 or n_trials <= 0:
        return 1.0
    # Poisson approx for single-cell excess/deficit
    if observed >= expected:
        # upper tail via normal approx
        z = (observed - expected) / math.sqrt(expected)
        return 0.5 * math.erfc(z / math.sqrt(2))
    z = (expected - observed) / math.sqrt(expected)
    return 0.5 * math.erfc(z / math.sqrt(2))


def enumerate_hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    """Auto segment x feature-value hypotheses."""
    cat_features = [
        "parity_sig",
        "highlow_sig",
        "size_sig",
        "first_digit_single",
        "star_pair",
    ]
    for s in range(STAR_MIN, STAR_MAX + 1):
        cat_features.append(f"star_{s}")

    records: list[dict] = []
    n_total = len(df)

    for seg_type, seg_key, mask in segment_masks(df):
        sub = df[mask]
        n_seg = len(sub)
        if n_seg < 8:
            continue
        for feat in cat_features:
            if feat not in df.columns:
                continue
            if feat == "first_digit_single":
                values = [True, False]
            else:
                values = df[feat].dropna().unique().tolist()
            for val in values:
                obs = int((sub[feat] == val).sum())
                global_rate = (df[feat] == val).mean()
                expected = n_seg * global_rate
                p = chi2_cell_pvalue(obs, expected, n_seg)
                records.append(
                    {
                        "segment_type": seg_type,
                        "segment": seg_key,
                        "feature": feat,
                        "value": str(val),
                        "n_segment": n_seg,
                        "observed": obs,
                        "expected": round(expected, 2),
                        "rate_segment": round(obs / n_seg, 4) if n_seg else 0,
                        "rate_global": round(global_rate, 4),
                        "p_value": p,
                    }
                )

    # Numeric buckets (qcut) x segment mean shift
    num_cols = ["sum", "min", "max", "spread", "odd_count", "low_count"]
    for col in num_cols:
        try:
            df["_bucket"] = pd.qcut(df[col], 4, duplicates="drop")
        except ValueError:
            continue
        for seg_type, seg_key, mask in segment_masks(df):
            sub = df[mask]
            if len(sub) < 8:
                continue
            for bucket in df["_bucket"].cat.categories:
                obs = int((sub["_bucket"] == bucket).sum())
                global_rate = (df["_bucket"] == bucket).mean()
                expected = len(sub) * global_rate
                p = chi2_cell_pvalue(obs, expected, len(sub))
                records.append(
                    {
                        "segment_type": seg_type,
                        "segment": seg_key,
                        "feature": col,
                        "value": str(bucket),
                        "n_segment": len(sub),
                        "observed": obs,
                        "expected": round(expected, 2),
                        "rate_segment": round(obs / len(sub), 4),
                        "rate_global": round(global_rate, 4),
                        "p_value": p,
                    }
                )
        df.drop(columns=["_bucket"], inplace=True, errors="ignore")

    hyp = pd.DataFrame(records)
    if hyp.empty:
        return hyp
    hyp["p_bonferroni"] = np.minimum(hyp["p_value"] * len(hyp), 1.0)
    bh = benjamini_hochberg(hyp["p_value"].to_numpy(), q=0.05)
    hyp["significant_bh"] = bh
    hyp["significant_bonferroni"] = hyp["p_bonferroni"] < 0.05
    hyp.sort_values("p_value", inplace=True)
    return hyp.reset_index(drop=True)


def mutual_info_ranking(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["sum", "min", "max", "spread", "odd_count", "low_count", "consecutive_pairs"]
    X = df[num_cols].fillna(0).to_numpy()
    y_month = df["month"].to_numpy()
    y_half = df["half"].map({"H1": 0, "H2": 1}).to_numpy()
    mi_m = mutual_info_classif(X, y_month, random_state=42)
    mi_h = mutual_info_classif(X, y_half, random_state=42)
    return pd.DataFrame(
        {
            "feature": num_cols,
            "mi_month": np.round(mi_m, 4),
            "mi_half": np.round(mi_h, 4),
        }
    ).sort_values("mi_month", ascending=False)


def shape_clusters(df: pd.DataFrame, k: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = ["sum", "odd_count", "low_count", "min", "max", "spread", "consecutive_pairs"]
    X = StandardScaler().fit_transform(df[num_cols].fillna(0))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    df = df.copy()
    df["shape_cluster"] = labels
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca1"] = coords[:, 0]
    df["pca2"] = coords[:, 1]

    concentration = []
    for c in range(k):
        sub = df[df["shape_cluster"] == c]
        for m in range(1, 13):
            n_m = (sub["month"] == m).sum()
            exp = len(sub) * (df["month"] == m).mean()
            concentration.append(
                {
                    "cluster": c,
                    "month": m,
                    "count": int(n_m),
                    "expected": round(exp, 1),
                    "n_cluster": len(sub),
                }
            )
    return df, pd.DataFrame(concentration)


def monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month").agg(
        draws=("date", "count"),
        mean_sum=("sum", "mean"),
        mean_min=("min", "mean"),
        mean_max=("max", "mean"),
        pct_first_single=("first_digit_single", "mean"),
        pct_2low=("low_count", lambda s: (s == 2).mean()),
        pct_3odd=("odd_count", lambda s: (s == 3).mean()),
    )
    return g.round(3)


def half_month_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "half"])
        .agg(draws=("date", "count"), mean_sum=("sum", "mean"), pct_2low=("low_count", lambda s: (s == 2).mean()))
        .round(3)
    )


def july_by_year(df: pd.DataFrame, month: int) -> pd.DataFrame:
    sub = df[df["month"] == month]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby("year")
        .agg(
            draws=("date", "count"),
            mean_sum=("sum", "mean"),
            top_parity=("parity_sig", lambda s: s.mode().iloc[0] if len(s) else ""),
            top_highlow=("highlow_sig", lambda s: s.mode().iloc[0] if len(s) else ""),
            pct_first_single=("first_digit_single", "mean"),
        )
        .round(3)
    )


def run_lengths(series: pd.Series) -> list[int]:
    runs: list[int] = []
    if series.empty:
        return runs
    cur = 1
    prev = series.iloc[0]
    for v in series.iloc[1:]:
        if v == prev:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
            prev = v
    runs.append(cur)
    return runs


def run_length_test(df: pd.DataFrame, shape_key: str, monte_carlo: int, seed: int) -> dict:
    col_map = {
        "parity": "parity_sig",
        "highlow": "highlow_sig",
        "size": "size_sig",
        "cluster": "shape_cluster",
    }
    col = col_map.get(shape_key, shape_key)
    if col not in df.columns:
        col = "parity_sig"
    ordered = df.sort_values("date")[col]
    obs_runs = run_lengths(ordered)
    obs_max = max(obs_runs) if obs_runs else 0
    obs_mean = float(np.mean(obs_runs)) if obs_runs else 0.0

    rng = np.random.default_rng(seed)
    sim_max = []
    sim_mean = []
    vals = ordered.to_numpy()
    for _ in range(monte_carlo):
        shuffled = rng.permutation(vals)
        runs = run_lengths(pd.Series(shuffled))
        sim_max.append(max(runs) if runs else 0)
        sim_mean.append(float(np.mean(runs)) if runs else 0.0)

    sim_max = np.array(sim_max)
    p_max = float((sim_max >= obs_max).mean())
    return {
        "shape_key": shape_key,
        "column": col,
        "observed_max_run": obs_max,
        "observed_mean_run": round(obs_mean, 2),
        "shuffle_max_p95": float(np.percentile(sim_max, 95)),
        "shuffle_max_p99": float(np.percentile(sim_max, 99)),
        "p_value_max_run": p_max,
        "monte_carlo": monte_carlo,
    }


def star_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_star = {s: df[f"star_{s}"].mean() for s in range(STAR_MIN, STAR_MAX + 1)}
    for m in range(1, 13):
        sub = df[df["month"] == m]
        if len(sub) < 5:
            continue
        for s in range(STAR_MIN, STAR_MAX + 1):
            obs = int(sub[f"star_{s}"].sum())
            exp = len(sub) * global_star[s] * 2  # two stars per draw
            p = chi2_cell_pvalue(obs, exp, len(sub) * 2)
            rows.append(
                {
                    "month": m,
                    "star": s,
                    "observed": obs,
                    "expected": round(exp, 1),
                    "p_value": p,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    return out


def print_report(
    df: pd.DataFrame,
    hyp: pd.DataFrame,
    mi: pd.DataFrame,
    monthly: pd.DataFrame,
    half_prof: pd.DataFrame,
    focus_year: pd.DataFrame,
    run_stat: dict,
    stars: pd.DataFrame,
    focus_month: int,
    surviving: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("SEASONAL PATTERN ANALYSIS (ML-grade data prep)")
    print("=" * 72)
    print(
        "\nDISCLAIMER: Draws are independent. Patterns here measure deviation from "
        "random chance — not predictive edge. Expect few survivors after correction.\n"
    )
    print(f"Draws analyzed: {len(df)}  |  {df['date'].min().date()} → {df['date'].max().date()}\n")

    print("--- Mutual information (numeric features vs month / half) ---")
    print(mi.to_string(index=False))
    print()

    print("--- Monthly profiles ---")
    print(monthly.to_string())
    print()

    name = MONTH_NAMES[focus_month]
    print(f"--- {name} (month {focus_month}) by year ---")
    if focus_year.empty:
        print("  (no data)")
    else:
        print(focus_year.to_string())
    print()

    print("--- Half-month (H1=days 1-15, H2=16-end) sample ---")
    print(half_prof.head(24).to_string())
    print()

    print("--- Run-length persistence ---")
    print(f"  Shape: {run_stat['shape_key']} ({run_stat['column']})")
    print(f"  Observed max run: {run_stat['observed_max_run']} draws")
    print(f"  Shuffle p95 max:  {run_stat['shuffle_max_p95']:.0f}  p99: {run_stat['shuffle_max_p99']:.0f}")
    print(f"  p-value (max run vs shuffle): {run_stat['p_value_max_run']:.4f}")
    print()

    print("--- Star seasonality (top deviations, raw p) ---")
    if stars.empty:
        print("  (none)")
    else:
        top = stars.nsmallest(8, "p_value")
        print(top.to_string(index=False))
    print()

    print(f"--- Hypothesis tests: {len(hyp)} auto-generated ---")
    survivors_bh = hyp[hyp["significant_bh"]] if not hyp.empty else hyp
    survivors_bonf = hyp[hyp["significant_bonferroni"]] if not hyp.empty else hyp
    print(f"  Survive Benjamini-Hochberg (FDR 5%): {len(survivors_bh)}")
    print(f"  Survive Bonferroni:                  {len(survivors_bonf)}")
    if surviving.empty:
        print("\n  No patterns survive both corrections (expected for random draws).")
    else:
        print("\n  Top surviving (BH):")
        print(surviving.head(15).to_string(index=False))
    print("\n" + "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal pattern mining with numpy/pandas/sklearn")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "euromillions_last_5_years.json",
    )
    parser.add_argument("-o", type=Path, default=Path(__file__).parent / "seasonal_analysis.json")
    parser.add_argument("--month", type=int, default=7, help="Focus month for deep-dive (default July=7)")
    parser.add_argument(
        "--shape-key",
        choices=("parity", "highlow", "size", "cluster"),
        default="highlow",
        help="Shape for run-length test",
    )
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--monte-carlo", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--dump-csv", action="store_true", help="Write seasonal_features.csv")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write JSON only; no report on stdout",
    )
    args = parser.parse_args()

    draws = load_draws(args.input)
    df = build_dataframe(draws)
    df, cluster_conc = shape_clusters(df, k=args.clusters)

    mi = mutual_info_ranking(df)
    monthly = monthly_profile(df)
    half_prof = half_month_profile(df)
    focus_year = july_by_year(df, args.month)
    run_stat = run_length_test(df, args.shape_key, args.monte_carlo, args.seed)
    stars = star_seasonality(df)
    hyp = enumerate_hypotheses(df)
    if not hyp.empty:
        hyp["significant_bh"] = benjamini_hochberg(hyp["p_value"].to_numpy(), q=args.fdr)
    surviving = hyp[hyp["significant_bh"]].head(20) if not hyp.empty else hyp

    if not args.quiet:
        print_report(
            df, hyp, mi, monthly, half_prof, focus_year, run_stat, stars, args.month, surviving
        )

    out = {
        "draws": len(df),
        "mutual_info": mi.to_dict(orient="records"),
        "monthly_profiles": monthly.reset_index().to_dict(orient="records"),
        "focus_month": args.month,
        "focus_month_by_year": focus_year.reset_index().to_dict(orient="records") if not focus_year.empty else [],
        "run_length": run_stat,
        "star_seasonality_top": stars.nsmallest(20, "p_value").to_dict(orient="records") if not stars.empty else [],
        "hypotheses_tested": len(hyp),
        "survivors_bh": int(hyp["significant_bh"].sum()) if not hyp.empty else 0,
        "survivors_bonferroni": int(hyp["significant_bonferroni"].sum()) if not hyp.empty else 0,
        "top_survivors_bh": surviving.to_dict(orient="records"),
        "cluster_concentration": cluster_conc.to_dict(orient="records"),
    }
    args.o.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"Saved: {args.o}")

    if args.dump_csv:
        csv_path = args.o.parent / "seasonal_features.csv"
        df.to_csv(csv_path, index=False)
        if not args.quiet:
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
