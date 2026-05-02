import os
import sys
from datetime import datetime
from pathlib import Path

_PC_ROOT = Path(__file__).resolve().parent.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import DAY, get_state, load_kernels
from results_artifacts import latest_results_csv


def _load_results(csv_path):
    df = pd.read_csv(csv_path)
    if "rank" in df.columns:
        df = df.sort_values("rank").reset_index(drop=True)
    return df


def _format_top10_markdown(df_top):
    cols = [
        "rank", "a1", "a2", "a3", "architecture", "assist_type",
        "dv_launch_km_s", "dv_after_launch_km_s", "mission_years", "science_sum"
    ]
    lines = []
    lines.append("# Top 10 Mission Results")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df_top.iterrows():
        row = []
        for c in cols:
            v = r[c] if c in r else ""
            if isinstance(v, float):
                if c in ("dv_launch_km_s", "dv_after_launch_km_s", "mission_years", "science_sum"):
                    row.append(f"{v:.4f}")
                else:
                    row.append(f"{v}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_best_details(best_row):
    lines = []
    lines.append("")
    lines.append("## Selected Best Mission (Rank 1)")
    lines.append("")
    for k, v in best_row.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


def _plot_orbits(best_row, asteroid_list, out_path):
    a1 = str(best_row["a1"]).upper()
    a2 = str(best_row["a2"]).upper()
    a3 = str(best_row["a3"]).upper()

    ids = {}
    for ast in asteroid_list:
        ids[str(ast["NAME"]).upper()] = str(int(ast["ID"]))
    a1_id = ids[a1]
    a2_id = ids[a2]
    a3_id = ids[a3]

    et0 = float(best_row["launch_et"])
    etf = float(best_row["arrive_a3_et"])
    t = np.arange(et0, etf + 2 * DAY, 5 * DAY)

    earth = np.array([get_state("399", ti)[0] for ti in t])
    p1 = np.array([get_state(a1_id, ti)[0] for ti in t])
    p2 = np.array([get_state(a2_id, ti)[0] for ti in t])
    p3 = np.array([get_state(a3_id, ti)[0] for ti in t])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(earth[:, 0], earth[:, 1], earth[:, 2], color="cyan", label="Earth")
    if str(best_row.get("architecture", "")) == "mars":
        mars = np.array([get_state("4", ti)[0] for ti in t])
        ax.plot(mars[:, 0], mars[:, 1], mars[:, 2], color="indianred", linestyle=":", label="Mars")
    elif str(best_row.get("architecture", "")) == "moon":
        moon = np.array([get_state("301", ti)[0] for ti in t])
        ax.plot(moon[:, 0], moon[:, 1], moon[:, 2], color="silver", linestyle=":", label="Moon")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color="red", label=a1)
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color="green", label=a2)
    ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="orange", label=a3)

    ax.scatter(*earth[0], color="cyan", s=30)
    ax.scatter(*p1[-1], color="red", s=30)
    ax.scatter(*p2[-1], color="green", s=30)
    ax.scatter(*p3[-1], color="orange", s=30)

    ax.set_title("Mission Orbit Context (Earth + Top-1 Asteroids)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    repo = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(repo)

    results_csv = latest_results_csv(repo)
    df = _load_results(results_csv)
    top10 = df.head(10).copy()
    best = top10.iloc[0].to_dict()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(repo, "optimal_asteroid_paths", "reports")
    plot_dir = os.path.join(repo, "optimal_asteroid_paths", "plots")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"top10_mission_report_{ts}.md")
    best_csv_path = os.path.join(report_dir, f"best_mission_full_row_{ts}.csv")
    orbit_plot_path = os.path.join(plot_dir, f"top1_orbit_context_{ts}.png")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_format_top10_markdown(top10))
        f.write(_format_best_details(best))

    pd.DataFrame([best]).to_csv(best_csv_path, index=False)

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    _plot_orbits(best, asteroid_list, orbit_plot_path)

    print(f"Report: {report_path}")
    print(f"Best row CSV: {best_csv_path}")
    print(f"Orbit plot: {orbit_plot_path}")


if __name__ == "__main__":
    main()
