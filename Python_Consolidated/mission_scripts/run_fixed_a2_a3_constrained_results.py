"""Run constrained optimization with fixed asteroid 2 and 3.

One-time utility that mirrors the constrained workflow but restricts candidate
triplets to:
    A1 -> <fixed_a2> -> <fixed_a3>
with A1 varying across the kernel-covered asteroid pool.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import spiceypy
from tqdm import tqdm

_PC_ROOT = Path(__file__).resolve().parent.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

from core import load_kernels
from optimization import load_composition_map, optimize_best_architecture
from run_and_export_constrained_results import (
    _apply_coverage_filters,
    _build_rows,
    _load_science_scores,
    _plot_top_summary,
)


def _name_to_index(asteroid_list, name):
    target = str(name).upper().strip()
    for idx, a in enumerate(asteroid_list):
        if str(a["NAME"]).upper() == target:
            return idx
    raise ValueError(f"Asteroid not found in loaded kernel list: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run constrained optimization with fixed asteroid 2 and 3."
    )
    parser.add_argument("--a2", default="PSYCHE", help="Fixed asteroid 2 name (default: PSYCHE).")
    parser.add_argument("--a3", default="THEMIS", help="Fixed asteroid 3 name (default: THEMIS).")
    parser.add_argument("--launch-min", default="Jan 1 12:00:00 UTC 2027")
    parser.add_argument("--launch-max", default="Dec 31 12:00:00 UTC 2035")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--architectures",
        default="direct,moon,mars,earth",
        help="Comma-separated architectures to consider (default: direct,moon,mars,earth).",
    )
    parser.add_argument("--mission-cap-years", type=float, default=30.0)
    parser.add_argument("--coverage-horizon-years", type=float, default=14.0)
    parser.add_argument("--max-leg-years", type=float, default=4.0)
    parser.add_argument("--max-stay-years", type=float, default=2.0)
    parser.add_argument(
        "--force-flyby",
        action="store_true",
        default=False,
        help="If set, only flyby architectures are considered (moon/mars/earth).",
    )
    args = parser.parse_args()

    allowed_architectures = [a.strip().lower() for a in args.architectures.split(",") if a.strip()]
    allowed_architectures = [a for a in allowed_architectures if a in {"direct", "moon", "mars", "earth"}]
    if args.force_flyby:
        allowed_architectures = [a for a in allowed_architectures if a in {"moon", "mars", "earth"}]
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

    launch_min_et = spiceypy.str2et(args.launch_min)
    launch_max_et = spiceypy.str2et(args.launch_max)
    asteroid_list, launch_max_et_adj = _apply_coverage_filters(
        asteroid_list, launch_min_et, launch_max_et, args.mission_cap_years, args.coverage_horizon_years
    )
    if len(asteroid_list) < 3:
        raise RuntimeError("Coverage filter left fewer than 3 asteroids; cannot build mission.")
    launch_max = args.launch_max
    if launch_max_et_adj < launch_max_et:
        launch_max = spiceypy.et2utc(launch_max_et_adj, "C", 0)
        print(f"Adjusted launch_max to kernel-supported bound: {launch_max}")

    j = _name_to_index(asteroid_list, args.a2)
    k = _name_to_index(asteroid_list, args.a3)
    if j == k:
        raise ValueError("A2 and A3 must be different asteroids.")

    # Candidate set: all valid A1 choices with fixed A2/A3.
    tasks = []
    for i in range(len(asteroid_list)):
        if len({asteroid_list[i]["ID"], asteroid_list[j]["ID"], asteroid_list[k]["ID"]}) < 3:
            continue
        tasks.append((i, j, k))
    if not tasks:
        raise RuntimeError("No valid A1 candidates remain after fixed A2/A3 and coverage filters.")

    launch_window = [spiceypy.str2et(args.launch_min), spiceypy.str2et(launch_max)]
    print(f"Running fixed-target constrained optimization for A2={args.a2.upper()}, A3={args.a3.upper()}...")
    print(f"Pass 1: Coarse evaluation ({len(tasks)} valid triplets)...")

    coarse = []
    for i, j, k in tqdm(tasks, desc="Coarse"):
        ids = [str(int(asteroid_list[x]["ID"])) for x in [i, j, k]]
        best_dv, best_arch = optimize_best_architecture(
            *ids, launch_window, 0, 0, 0, quick=True, allowed_architectures=allowed_architectures
        )
        coarse.append((i, j, k, best_dv, best_arch))

    coarse.sort(key=lambda x: x[3])
    top = coarse[: min(args.top_n, len(coarse))]

    print(f"\nPass 2: Fine optimization on top {len(top)} candidates...")
    print(f"  Architectures in top {len(top)}: "
          + str({a: sum(1 for t in top if t[4] == a) for a in ['direct', 'moon', 'mars', 'earth']}))

    results = []
    for i, j, k, _, _arch in tqdm(top, desc="Fine"):
        ids = [str(int(asteroid_list[x]["ID"])) for x in [i, j, k]]
        result, best_arch = optimize_best_architecture(
            *ids, launch_window, 0, 0, 0, quick=False, allowed_architectures=allowed_architectures
        )
        result["architecture"] = best_arch
        results.append((i, j, k, result))

    results.sort(key=lambda x: (0 if np.isfinite(x[3].get("delta_v_total", np.inf)) and x[3].get("delta_v_total", np.inf) < 1e3 else 1,
                                x[3].get("delta_v_total", np.inf)))

    print(f"\n{'='*90}\nTOP 10 PATHS (fixed A2/A3)\n{'='*90}")
    for rank, (i, j, k, res) in enumerate(results[:10], 1):
        n1, n2, n3 = asteroid_list[i]["NAME"], asteroid_list[j]["NAME"], asteroid_list[k]["NAME"]
        dv = float(res.get("delta_v_total", np.nan))
        ldv = float(np.linalg.norm(res["delta_v_launch"])) if len(res.get("delta_v_launch", [])) > 0 else float("nan")
        arch = res.get("architecture", "direct")
        ldv_txt = f"{ldv:.2f}" if np.isfinite(ldv) else "n/a"
        print(f"  #{rank}: {n1} -> {n2} -> {n3}  |  dv={dv:.2f}  launch_dv={ldv_txt}  [{arch.upper()}]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"a2_{args.a2.lower()}_a3_{args.a3.lower()}".replace(" ", "_")
    oap = os.path.join("optimal_asteroid_paths")
    plot_dir = os.path.join(oap, "plots")
    os.makedirs(oap, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = os.path.join(oap, f"results_{timestamp}_{slug}")
    pkl_path = f"{base}.pkl"
    csv_path = f"{base}.csv"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    df = _build_rows(results, asteroid_list, comp_map, science_scores)
    df.to_csv(csv_path, index=False)
    plot_path = os.path.join(plot_dir, f"dv_reference_top1_{timestamp}_{slug}.png")
    _plot_top_summary(df, plot_path, f"Top solution (fixed A2={args.a2.upper()}, A3={args.a3.upper()})")

    print("\nDone.")
    print(f"Saved PKL: {pkl_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot: {plot_path}")


if __name__ == "__main__":
    main()
