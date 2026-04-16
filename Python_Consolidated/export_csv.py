"""
export_csv.py — Export optimization result pickle files to CSV.

Usage:
    python export_csv.py <path_to_pkl>
    python export_csv.py ../optimal_asteroid_paths/pkl/results_science_priority.pkl
"""

import os
import sys
import csv
import pickle
import numpy as np
import spiceypy

sys.path.insert(0, ".")
from core import load_kernels, YEAR


def load_results(pkl_path, bsp_folder, generic_path):
    """Load a results pkl and resolve asteroid names."""
    asteroid_list = load_kernels(bsp_folder, generic_path)

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    rows = []
    for rank, entry in enumerate(results):
        i, j, k, d = entry
        a1 = asteroid_list[i]["NAME"]
        a2 = asteroid_list[j]["NAME"]
        a3 = asteroid_list[k]["NAME"]
        dur = (d["et_arrive_3"] - d["et_launch"]) / YEAR

        row = {
            "rank": rank + 1,
            "a1": a1, "a2": a2, "a3": a3,
            "dv": d["delta_v_total"],
            "science": d.get("science_sum", 0),
            "score": d.get("combined_score", d["delta_v_total"]),
            "arch": d.get("architecture", "direct"),
            "duration": dur,
            # Per-maneuver magnitudes
            "dv_launch": float(np.linalg.norm(d["delta_v_launch"])),
            "dv_A1_arrive": float(np.linalg.norm(d["delta_v_A1_arrive"])),
            "dv_A1_leave": float(np.linalg.norm(d["delta_v_A1_leave"])),
            "dv_A2_arrive": float(np.linalg.norm(d["delta_v_A2_arrive"])),
            "dv_A2_leave": float(np.linalg.norm(d["delta_v_A2_leave"])),
            "dv_A3_arrive": float(np.linalg.norm(d["delta_v_A3_arrive"])),
        }
        # Flyby dv (scalar or zero)
        if "delta_v_flyby" in d:
            flyby_val = d["delta_v_flyby"]
            row["dv_flyby"] = float(flyby_val) if np.isscalar(flyby_val) else float(np.linalg.norm(flyby_val))
        else:
            row["dv_flyby"] = 0.0

        # Dates for each event
        for key, col in [("et_launch", "date_launch"), ("et_arrive_1", "date_arrive_A1"),
                         ("et_stay_1", "date_leave_A1"), ("et_arrive_2", "date_arrive_A2"),
                         ("et_stay_2", "date_leave_A2"), ("et_arrive_3", "date_arrive_A3")]:
            if key in d:
                row[col] = spiceypy.et2utc(d[key], "C", 0)[:20].strip()
        if "et_flyby" in d:
            row["date_flyby"] = spiceypy.et2utc(d["et_flyby"], "C", 0)[:20].strip()

        rows.append(row)
    return rows


def export_csv(rows, csv_path):
    """Write rows to a CSV file."""
    columns = [
        "rank", "a1", "a2", "a3", "dv", "science", "score", "arch", "duration",
        "dv_launch", "dv_flyby", "dv_A1_arrive", "dv_A1_leave",
        "dv_A2_arrive", "dv_A2_leave", "dv_A3_arrive",
        "date_launch", "date_flyby", "date_arrive_A1", "date_leave_A1",
        "date_arrive_A2", "date_leave_A2", "date_arrive_A3",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = dict(r)
            for key in ["dv", "science", "score", "duration",
                        "dv_launch", "dv_flyby", "dv_A1_arrive", "dv_A1_leave",
                        "dv_A2_arrive", "dv_A2_leave", "dv_A3_arrive"]:
                if key in out and isinstance(out[key], float):
                    out[key] = round(out[key], 4)
            writer.writerow(out)
    print(f"  Saved → {csv_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_csv.py <path_to_pkl>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bsp_folder = os.path.join(repo_root, "NOTABLE_ASTEROID_BSPs")
    generic_path = "/Users/rebnoob/Documents/ae105/generic_kernels"

    print(f"Loading {pkl_path}...")
    rows = load_results(pkl_path, bsp_folder, generic_path)
    print(f"  {len(rows)} paths loaded")

    csv_path = pkl_path.replace(".pkl", ".csv")
    export_csv(rows, csv_path)

    if sys.platform == "darwin":
        os.system(f'open "{csv_path}"')


if __name__ == "__main__":
    main()
