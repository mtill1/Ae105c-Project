import os
import sys
import pickle
import argparse
import glob
from datetime import datetime
from pathlib import Path

_PC_ROOT = Path(__file__).resolve().parent.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy

from core import YEAR, load_kernels
from optimization import load_composition_map, two_level_optimize


def _load_science_scores(csv_path):
    df = pd.read_csv(csv_path)
    scores = {}
    for _, row in df.iterrows():
        name = str(row["Name_DecRadius"]).split("(")[0].strip()
        parts = name.split()
        if parts and parts[0].replace(".", "").isdigit():
            name = " ".join(parts[1:])
        scores[name.upper()] = float(row["Total_WeightedScore"])
    return scores


def _flyby_epoch(res):
    if res.get("et_flyby") is not None:
        try:
            return float(res["et_flyby"])
        except (TypeError, ValueError):
            pass
    if res.get("et_mars") is not None:
        try:
            return float(res["et_mars"])
        except (TypeError, ValueError):
            pass
    return float("nan")


def _build_rows(results, asteroid_list, comp_map, science_scores):
    rows = []
    for rank, (i, j, k, res) in enumerate(results, 1):
        n1 = asteroid_list[i]["NAME"].upper()
        n2 = asteroid_list[j]["NAME"].upper()
        n3 = asteroid_list[k]["NAME"].upper()
        c1 = comp_map.get(n1, "Unknown")
        c2 = comp_map.get(n2, "Unknown")
        c3 = comp_map.get(n3, "Unknown")

        s1 = science_scores.get(n1, np.nan)
        s2 = science_scores.get(n2, np.nan)
        s3 = science_scores.get(n3, np.nan)
        sci_sum = np.nansum([s1, s2, s3])

        dv_launch = float(np.linalg.norm(res["delta_v_launch"])) if len(res["delta_v_launch"]) > 0 else np.nan
        dv_after = float(res["delta_v_total"])
        dv_total_ref = dv_launch + dv_after if np.isfinite(dv_launch) else np.nan

        et_launch = float(res.get("et_launch", np.nan))
        et_arrive_1 = float(res.get("et_arrive_1", np.nan))
        et_stay_1 = float(res.get("et_stay_1", np.nan))
        et_arrive_2 = float(res.get("et_arrive_2", np.nan))
        et_stay_2 = float(res.get("et_stay_2", np.nan))
        et_arrive_3 = float(res.get("et_arrive_3", np.nan))
        mission_years = float((et_arrive_3 - et_launch) / YEAR) if np.isfinite(et_arrive_3) and np.isfinite(et_launch) else np.nan
        arch = res.get("architecture", "direct")
        flyby_alt = res.get("flyby_altitude_km", np.nan)

        if arch == "mars":
            assist_type = "mars_flyby"
        elif arch == "moon":
            assist_type = "moon_flyby"
        elif arch == "earth":
            assist_type = "earth_gravity_assist"
        else:
            assist_type = "none"

        is_feasible = bool(res.get("feasible", np.isfinite(dv_launch) and dv_after < 1e3))
        fail_reason = str(res.get("fail_reason", "")).strip()
        if is_feasible and not fail_reason:
            fail_reason = "none"
        elif (not is_feasible) and not fail_reason:
            fail_reason = "unspecified_infeasible"

        et_fb = _flyby_epoch(res)
        flyby_body = str(res.get("flyby_body", ""))
        flyby_utc = spiceypy.et2utc(et_fb, "C", 3) if np.isfinite(et_fb) else ""
        if arch == "mars":
            leg1 = "electric_low_thrust_post_mars_flyby_to_a1"
        elif arch == "moon":
            leg1 = "electric_low_thrust_post_moon_flyby_to_a1"
        else:
            leg1 = "electric_low_thrust_earth_to_a1"

        rows.append(
            {
                "rank": rank,
                "a1": n1,
                "a2": n2,
                "a3": n3,
                "comp_a1": c1,
                "comp_a2": c2,
                "comp_a3": c3,
                "science_a1": s1,
                "science_a2": s2,
                "science_a3": s3,
                "science_sum": sci_sum,
                "architecture": arch,
                "assist_type": assist_type,
                "flyby_altitude_km": flyby_alt,
                "flyby_et": et_fb,
                "flyby_body_naif_id": flyby_body,
                "flyby_utc": flyby_utc,
                "trajectory_model": "low_thrust_sims_flanagan",
                "propulsion_profile": "launch_impulsive_then_all_electric",
                "leg0_launch_transfer": "earth_departure_impulsive_reference_only_not_inter_asteroid",
                "leg1_transfer": leg1,
                "leg2_transfer": "electric_low_thrust_a1_to_a2",
                "leg3_transfer": "electric_low_thrust_a2_to_a3",
                "dv_launch_km_s": dv_launch,
                "dv_after_launch_km_s": dv_after,
                "dv_total_reference_km_s": dv_total_ref,
                "lt_leg0_dv_km_s": float(res.get("lt_leg0_dv", np.nan)),
                "lt_leg1_dv_km_s": float(res.get("lt_leg1_dv", np.nan)),
                "lt_leg2_dv_km_s": float(res.get("lt_leg2_dv", np.nan)),
                "lt_final_mass_kg": float(res.get("lt_final_mass_kg", np.nan)),
                "lambert_leg0_required_dv_km_s": float(res.get("lambert_leg0_dv_km_s", np.nan)),
                "lambert_leg1_required_dv_km_s": float(res.get("lambert_leg1_dv_km_s", np.nan)),
                "lambert_leg2_required_dv_km_s": float(res.get("lambert_leg2_dv_km_s", np.nan)),
                "feasible": is_feasible,
                "fail_reason": fail_reason,
                "mission_years": mission_years,
                "launch_et": et_launch,
                "arrive_a1_et": et_arrive_1,
                "leave_a1_et": et_stay_1,
                "arrive_a2_et": et_arrive_2,
                "leave_a2_et": et_stay_2,
                "arrive_a3_et": et_arrive_3,
            }
        )
    return pd.DataFrame(rows)


def _plot_top_summary(df, out_path, title):
    if df.empty:
        return
    top = df.iloc[0]
    labels = ["Launch dv", "Post-launch dv\n(optimization objective)"]
    vals = [top["dv_launch_km_s"], top["dv_after_launch_km_s"]]
    colors = ["#4e79a7", "#f28e2b"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("km/s")
    ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _coverage_bounds_for_bsp(bsp_path):
    ids = spiceypy.spkobj(bsp_path)
    n_ids = spiceypy.card(ids)
    if n_ids < 1:
        return None
    body_id = ids[0]
    cover = spiceypy.cell_double(20000)
    spiceypy.spkcov(bsp_path, int(body_id), cover)
    n = spiceypy.wncard(cover)
    if n < 1:
        return None
    starts = []
    ends = []
    for i in range(n):
        left, right = spiceypy.wnfetd(cover, i)
        starts.append(float(left))
        ends.append(float(right))
    return min(starts), max(ends)


def _collect_bounds_by_body_id(bsp_dir):
    bounds_by_id = {}
    for bsp in glob.glob(os.path.join(bsp_dir, "*.bsp")):
        try:
            ids = spiceypy.spkobj(bsp)
            n_ids = spiceypy.card(ids)
            if n_ids < 1:
                continue
            body_id = int(ids[0])
        except Exception:
            continue
        b = _coverage_bounds_for_bsp(bsp)
        if b is None:
            continue
        start_et, end_et = b
        if body_id in bounds_by_id:
            old_s, old_e = bounds_by_id[body_id]
            bounds_by_id[body_id] = (min(old_s, start_et), max(old_e, end_et))
        else:
            bounds_by_id[body_id] = (start_et, end_et)
    return bounds_by_id


def _apply_coverage_filters(asteroid_list, launch_min_et, launch_max_et, mission_cap_years, coverage_horizon_years):
    effective_years = mission_cap_years if mission_cap_years > 0 else coverage_horizon_years
    mission_cap_sec = effective_years * YEAR
    needed_end = launch_max_et + mission_cap_sec
    bsp_dir = "NOTABLE_ASTEROID_BSPs"
    bounds_by_id = _collect_bounds_by_body_id(bsp_dir)
    if not bounds_by_id:
        return asteroid_list, launch_max_et

    # Keep asteroids that can support the entire launch window + mission horizon.
    kept = []
    max_end = -np.inf
    for a in asteroid_list:
        body_id = int(a["ID"])
        if body_id not in bounds_by_id:
            continue
        start_et, end_et = bounds_by_id[body_id]
        max_end = max(max_end, end_et)
        if start_et <= launch_min_et and end_et >= needed_end:
            kept.append(a)

    # If strict filter wipes out the pool, clip launch_max to what kernels can support.
    if len(kept) < 3 and np.isfinite(max_end):
        launch_max_clipped = max(launch_min_et, max_end - mission_cap_sec)
        needed_end2 = launch_max_clipped + mission_cap_sec
        kept2 = []
        for a in asteroid_list:
            body_id = int(a["ID"])
            if body_id not in bounds_by_id:
                continue
            start_et, end_et = bounds_by_id[body_id]
            if start_et <= launch_min_et and end_et >= needed_end2:
                kept2.append(a)
        return kept2 if len(kept2) >= 3 else asteroid_list, launch_max_clipped

    return kept, launch_max_et


def main():
    parser = argparse.ArgumentParser(description="Run constrained optimization and export results.")
    parser.add_argument("--launch-min", default="Jan 1 12:00:00 UTC 2027")
    parser.add_argument("--launch-max", default="Dec 31 12:00:00 UTC 2035")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--require-diverse-composition", action="store_true", default=False)
    parser.add_argument(
        "--label",
        default="electric_only",
        help="Legacy CLI flag (ignored for filenames). Outputs: results_<timestamp>.{csv,pkl}.",
    )
    parser.add_argument("--architectures", default="mars",
                        help="Comma-separated architectures to consider (e.g. direct,mars,earth).")
    parser.add_argument("--mission-cap-years", type=float, default=30.0,
                        help="Total mission duration cap in years.")
    parser.add_argument("--coverage-horizon-years", type=float, default=14.0,
                        help="Coverage horizon used when mission cap is disabled (<=0).")
    parser.add_argument("--max-leg-years", type=float, default=4.0,
                        help="Upper bound per transfer leg duration in years.")
    parser.add_argument("--max-stay-years", type=float, default=2.0,
                        help="Upper bound per asteroid stay duration in years.")
    parser.add_argument("--force-flyby", action="store_true", default=True,
                        help="Require flyby architectures only (no direct).")
    args = parser.parse_args()
    allowed_architectures = [a.strip() for a in args.architectures.split(",") if a.strip() and a.strip() not in {"moon", "earth"}]
    if args.force_flyby:
        allowed_architectures = [a for a in allowed_architectures if a in {"mars"}]
    if not allowed_architectures:
        allowed_architectures = ["mars"]

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(repo_root)

    os.environ["AE_MISSION_CAP_YEARS"] = str(args.mission_cap_years)
    os.environ["AE_MAX_LEG_YEARS"] = str(args.max_leg_years)
    os.environ["AE_MAX_STAY_YEARS"] = str(args.max_stay_years)
    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    comp_map = load_composition_map("asteroid_tradeoff.csv")
    science_scores = _load_science_scores("asteroid_tradeoff.csv")
    required = {"C", "S", "X/M"} if args.require_diverse_composition else None
    launch_min = args.launch_min
    launch_max = args.launch_max
    launch_min_et = spiceypy.str2et(launch_min)
    launch_max_et = spiceypy.str2et(launch_max)
    asteroid_list, launch_max_et_adj = _apply_coverage_filters(
        asteroid_list, launch_min_et, launch_max_et, args.mission_cap_years, args.coverage_horizon_years
    )
    if len(asteroid_list) < 3:
        raise RuntimeError("Coverage filter left fewer than 3 asteroids; cannot build mission.")
    if launch_max_et_adj < launch_max_et:
        launch_max = spiceypy.et2utc(launch_max_et_adj, "C", 0)
        print(f"Adjusted launch_max to kernel-supported bound: {launch_max}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running constrained optimization (launch impulsive, post-launch electric-only)...")
    res_dv = two_level_optimize(
        asteroid_list,
        0,
        0,
        0,
        launch_min,
        launch_max,
        top_n=args.top_n,
        science_scores=None,
        alpha=1.0,
        comp_map=comp_map if args.require_diverse_composition else None,
        required_compositions=required,
        allowed_architectures=allowed_architectures,
    )

    oap = os.path.join("optimal_asteroid_paths")
    plot_dir = os.path.join(oap, "plots")
    os.makedirs(oap, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = os.path.join(oap, f"results_{timestamp}")
    pkl_path = f"{base}.pkl"
    csv_path = f"{base}.csv"
    with open(pkl_path, "wb") as f:
        pickle.dump(res_dv, f)

    df_dv = _build_rows(res_dv, asteroid_list, comp_map, science_scores)
    df_dv.to_csv(csv_path, index=False)

    # Reference plot: launch impulsive vs post-launch electric objective (rank 1).
    plot_path = os.path.join(plot_dir, f"dv_reference_top1_{timestamp}.png")
    _plot_top_summary(df_dv, plot_path, "Top solution: launch vs post-launch Δv")

    print("\nDone.")
    if df_dv.empty:
        print("No feasible solutions found under current constraints.")
    print(f"Saved PKL: {pkl_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot: {plot_path}")


if __name__ == "__main__":
    main()
