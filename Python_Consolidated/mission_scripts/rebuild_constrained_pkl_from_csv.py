"""Rebuild results_*.pkl + dv bar chart from a constrained results CSV.

If the CSV includes ``flyby_et``, attempts a cheap Lambert + m-rev match to the
stored epochs. Otherwise (typical export), each row is **re-optimized** with
``optimize_times_flyby`` (Mars, m=(0,0,0,0)) over the launch window so the PKL
matches current ``compute_path_with_flyby`` physics and ``main.py plot`` inputs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy

_PC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PC_ROOT.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

from core import DAY, load_kernels
from optimization import (
    compute_path_with_flyby,
    load_composition_map,
    optimize_times,
    optimize_times_flyby,
)

from run_and_export_constrained_results import _build_rows, _load_science_scores, _plot_top_summary


def _name_to_indices(asteroid_list, n1: str, n2: str, n3: str):
    up = {a["NAME"].upper(): k for k, a in enumerate(asteroid_list)}
    for n in (n1, n2, n3):
        if n.upper() not in up:
            raise KeyError(f"Asteroid {n!r} not in loaded BSP list")
    return up[n1.upper()], up[n2.upper()], up[n3.upper()]


def _best_m_for_fixed_flyby(
    a1_id: str,
    a2_id: str,
    a3_id: str,
    et_launch: float,
    et_flyby: float,
    et_arrive_1: float,
    et_stay_1: float,
    et_arrive_2: float,
    et_stay_2: float,
    et_arrive_3: float,
    target_launch: float,
    target_after: float,
):
    best = None
    best_err = np.inf
    for m0 in (0, 1, 2):
        for m1 in (0, 1, 2):
            for m2 in (0, 1, 2):
                for m3 in (0, 1, 2):
                    res = compute_path_with_flyby(
                        a1_id,
                        a2_id,
                        a3_id,
                        et_launch,
                        et_flyby,
                        et_arrive_1,
                        et_stay_1,
                        et_arrive_2,
                        et_stay_2,
                        et_arrive_3,
                        "mars",
                        m0,
                        m1,
                        m2,
                        m3,
                    )
                    if not res.get("feasible"):
                        continue
                    dv_l = float(np.linalg.norm(res["delta_v_launch"]))
                    dv_a = float(res["delta_v_total"])
                    err = abs(dv_l - target_launch) + abs(dv_a - target_after)
                    if err < best_err:
                        best_err = err
                        best = (res, (m0, m1, m2, m3))
    return best


def _merge_from_csv_flyby_et(row, a1_id: str, a2_id: str, a3_id: str):
    et_launch = float(row["launch_et"])
    et_arrive_1 = float(row["arrive_a1_et"])
    et_stay_1 = float(row["leave_a1_et"])
    et_arrive_2 = float(row["arrive_a2_et"])
    et_stay_2 = float(row["leave_a2_et"])
    et_arrive_3 = float(row["arrive_a3_et"])
    et_fb = float(row["flyby_et"])
    tgt_l = float(row["dv_launch_km_s"])
    tgt_a = float(row["dv_after_launch_km_s"])
    hit = _best_m_for_fixed_flyby(
        a1_id,
        a2_id,
        a3_id,
        et_launch,
        et_fb,
        et_arrive_1,
        et_stay_1,
        et_arrive_2,
        et_stay_2,
        et_arrive_3,
        tgt_l,
        tgt_a,
    )
    if hit is None:
        return None
    res, mrevs = hit
    merged = dict(res)
    merged["et_launch"] = et_launch
    merged["et_flyby"] = float(merged.get("et_flyby", et_fb))
    merged["et_arrive_1"] = et_arrive_1
    merged["et_stay_1"] = et_stay_1
    merged["et_arrive_2"] = et_arrive_2
    merged["et_stay_2"] = et_stay_2
    merged["et_arrive_3"] = et_arrive_3
    merged["m_revs"] = mrevs
    merged["flyby_name"] = "mars"
    merged["architecture"] = "mars"
    return merged


def main():
    parser = argparse.ArgumentParser(description="Rebuild PKL + dv plot from constrained CSV.")
    parser.add_argument("--csv", required=True, help="Path to results CSV (repo-relative or absolute).")
    parser.add_argument(
        "--kernels",
        default=os.environ.get("AE105C_SPICE_GENERIC", "generic_kernels"),
        help="SPICE generic kernel directory.",
    )
    parser.add_argument("--launch-min", default="Jan 1 12:00:00 UTC 2027")
    parser.add_argument("--launch-max", default="Dec 31 12:00:00 UTC 2035")
    parser.add_argument(
        "--mission-cap-years",
        type=float,
        default=30.0,
        help="Forwarded to AE_MISSION_CAP_YEARS (match run_and_export_constrained_results).",
    )
    parser.add_argument(
        "--max-leg-years",
        type=float,
        default=4.0,
        help="Forwarded to AE_MAX_LEG_YEARS.",
    )
    parser.add_argument(
        "--max-stay-years",
        type=float,
        default=2.0,
        help="Forwarded to AE_MAX_STAY_YEARS.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Output stamp YYYYMMDD_HHMMSS (default: now).",
    )
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    os.environ["AE_MISSION_CAP_YEARS"] = str(args.mission_cap_years)
    os.environ["AE_MAX_LEG_YEARS"] = str(args.max_leg_years)
    os.environ["AE_MAX_STAY_YEARS"] = str(args.max_stay_years)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO_ROOT, args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df_in = pd.read_csv(csv_path)
    ts = args.timestamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", args.kernels)
    comp_map = load_composition_map("asteroid_tradeoff.csv")
    try:
        science_scores = _load_science_scores(os.path.join(_REPO_ROOT, "asteroid_tradeoff.csv"))
    except Exception:
        science_scores = {}

    launch_range = (spiceypy.str2et(args.launch_min), spiceypy.str2et(args.launch_max))
    has_flyby_et = "flyby_et" in df_in.columns and df_in["flyby_et"].notna().any()

    results = []
    for _, row in df_in.sort_values("rank").iterrows():
        arch = str(row.get("architecture", "")).lower()
        if arch != "mars":
            print(f"skip rank {int(row['rank'])}: architecture {arch!r}")
            continue
        n1, n2, n3 = str(row["a1"]), str(row["a2"]), str(row["a3"])
        try:
            i, j, k = _name_to_indices(asteroid_list, n1, n2, n3)
        except KeyError as e:
            print(f"skip rank {int(row['rank'])}: {e}")
            continue
        a1_id = str(int(asteroid_list[i]["ID"]))
        a2_id = str(int(asteroid_list[j]["ID"]))
        a3_id = str(int(asteroid_list[k]["ID"]))

        if has_flyby_et and np.isfinite(row.get("flyby_et", np.nan)):
            merged = _merge_from_csv_flyby_et(row, a1_id, a2_id, a3_id)
            if merged is None:
                print(f"warn: flyby_et match failed rank {int(row['rank'])}; re-optimizing…")
                merged = None
        else:
            merged = None

        if merged is None:
            print(f"optimize rank {int(row['rank'])} {n1}->{n2}->{n3} …")
            merged = optimize_times_flyby(a1_id, a2_id, a3_id, launch_range, "mars", 0, 0, 0, 0)
            merged["architecture"] = "mars"
            merged["flyby_name"] = "mars"
            merged["m_revs"] = (0, 0, 0, 0)
            mars_ok = bool(merged.get("feasible")) and float(merged.get("delta_v_total", 1e9)) < 999.0
            if not mars_ok:
                print(f"  mars infeasible/penalty → direct electric fallback")
                merged = optimize_times(a1_id, a2_id, a3_id, launch_range, 0, 0, 0)
                merged["architecture"] = "direct"
                merged.pop("flyby_name", None)
                merged["m_revs"] = (0, 0, 0, 0)

        results.append((i, j, k, merged))
        ldv = float(np.linalg.norm(merged.get("delta_v_launch", [])))
        print(
            f"  ok rank {int(row['rank'])} arch={merged.get('architecture')} "
            f"feasible={merged.get('feasible')} "
            f"launch_dv={ldv:.2f} post={float(merged.get('delta_v_total', np.nan)):.2f}"
        )

    if not results:
        raise RuntimeError("No rows exported.")

    oap = os.path.join(_REPO_ROOT, "optimal_asteroid_paths")
    plot_dir = os.path.join(oap, "plots")
    os.makedirs(oap, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = os.path.join(oap, f"results_{ts}")
    pkl_path = f"{base}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    df_out = _build_rows(results, asteroid_list, comp_map, science_scores)
    out_csv = f"{base}.csv"
    df_out.to_csv(out_csv, index=False)

    plot_path = os.path.join(plot_dir, f"dv_reference_top1_{ts}.png")
    _plot_top_summary(df_out, plot_path, "Top solution (rebuilt): launch vs post-launch Δv")

    print(f"Saved PKL: {pkl_path}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
