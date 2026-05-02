"""Encode synoptic flightpath MP4 for one packaged rank folder (CSV + solution_row)."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PC_ROOT.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

from core import load_kernels
from visualization import flightpath_animation


def _kernels_candidates(cli: str) -> list:
    out = []
    if cli:
        out.append(cli)
    env = os.environ.get("AE105C_GENERIC_KERNELS")
    if env:
        out.append(env)
    out.extend(
        [
            str(_REPO_ROOT / "generic_kernels"),
            str(Path.home() / "Documents" / "ae105" / "generic_kernels"),
            "/Users/rebnoob/Documents/ae105/generic_kernels",
        ]
    )
    seen = set()
    uniq = []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _resolve_kernels(cli: str) -> str:
    tls = ("lsk", "naif0012.tls")
    for p in _kernels_candidates(cli):
        if os.path.isfile(os.path.join(p, *tls)):
            return p
    tried = ", ".join(_kernels_candidates(cli))
    raise FileNotFoundError(
        "Could not find naif0012.tls under any kernel path. "
        f"Tried: {tried}. Set AE105C_GENERIC_KERNELS or pass --kernels."
    )


def _vec_from_burns(burns: pd.DataFrame, event: str) -> np.ndarray:
    r = burns.loc[burns["event"] == event].iloc[0]
    return np.array(
        [float(r["dv_x_km_s"]), float(r["dv_y_km_s"]), float(r["dv_z_km_s"])],
        dtype=float,
    )


def _pdv_from_package(folder: str) -> dict:
    row = pd.read_csv(os.path.join(folder, "solution_row.csv")).iloc[0]
    burns = pd.read_csv(os.path.join(folder, "burn_events.csv"))
    fa = row.get("flyby_altitude_km")
    pdv = {
        "architecture": str(row["architecture"]),
        "flyby_altitude_km": float(fa) if pd.notna(fa) else None,
        "delta_v_launch": _vec_from_burns(burns, "earth_launch"),
        "delta_v_total": float(row["dv_after_launch_km_s"]),
        "et_launch": float(row["launch_et"]),
        "et_arrive_1": float(row["arrive_a1_et"]),
        "et_stay_1": float(row["leave_a1_et"]),
        "et_arrive_2": float(row["arrive_a2_et"]),
        "et_stay_2": float(row["leave_a2_et"]),
        "et_arrive_3": float(row["arrive_a3_et"]),
        "delta_v_A1_leave": _vec_from_burns(burns, "a1_depart"),
        "delta_v_A2_leave": _vec_from_burns(burns, "a2_depart"),
    }
    if "mission_years" in row and pd.notna(row.get("mission_years")):
        pdv["mission_years"] = float(row["mission_years"])
    return pdv


def _index_from_name(asteroid_list, name: str) -> int:
    upper = str(name).upper()
    for i, ast in enumerate(asteroid_list):
        if str(ast["NAME"]).upper() == upper:
            return i
    raise ValueError(f"Could not find asteroid '{name}' in loaded kernel list.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        default=str(
            _REPO_ROOT
            / "optimal_asteroid_paths/reports/top_general_solution_packages/rank_01_amphitrite_aurora_eunomia"
        ),
        help="Rank package folder with time_position_velocity.csv and solution_row.csv.",
    )
    parser.add_argument(
        "--kernels",
        default="",
        help="Generic SPICE kernels root (lsk/, spk/, pck/). If empty, auto-detect.",
    )
    parser.add_argument("--duration", type=float, default=18.0, help="Video length in seconds.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path (default: <folder>/visualization_flightpath.mp4).",
    )
    args = parser.parse_args()
    folder = os.path.abspath(args.folder)
    out = args.out or os.path.join(folder, "visualization_flightpath.mp4")

    os.chdir(_REPO_ROOT)
    bsp = os.path.join(_REPO_ROOT, "NOTABLE_ASTEROID_BSPs")
    gk = _resolve_kernels(str(args.kernels or "").strip())
    asteroid_list = load_kernels(bsp, gk)

    row = pd.read_csv(os.path.join(folder, "solution_row.csv")).iloc[0]
    pdv = _pdv_from_package(folder)

    flightpath_animation(
        pdv,
        asteroid_list,
        _index_from_name(asteroid_list, row["a1"]),
        _index_from_name(asteroid_list, row["a2"]),
        _index_from_name(asteroid_list, row["a3"]),
        t_duration=float(args.duration),
        output_video_name=out,
        package_folder=folder,
    )


if __name__ == "__main__":
    main()
