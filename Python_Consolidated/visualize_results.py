"""
visualize_results.py — Visualize optimization result pickle files.

Usage:
    python visualize_results.py <path_to_pkl>
    python visualize_results.py ../optimal_asteroid_paths/pkl/results_science_priority.pkl
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        launch = spiceypy.et2utc(d["et_launch"], "C", 0)[:11].strip()

        rows.append({
            "rank": rank + 1,
            "a1": a1, "a2": a2, "a3": a3,
            "label": f"{a1} → {a2} → {a3}",
            "dv": d["delta_v_total"],
            "science": d.get("science_sum", 0),
            "score": d.get("combined_score", d["delta_v_total"]),
            "arch": d.get("architecture", "direct"),
            "duration": dur,
            "launch": launch,
        })
    return rows


def plot_all(rows, title="Optimization Results"):
    """Generate a 4-panel figure from result rows."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Colors by architecture
    arch_colors = {"moon": "#4A90D9", "mars": "#D94A4A", "direct": "#5BBD72"}
    colors = [arch_colors.get(r["arch"], "#888888") for r in rows]

    # --- Panel 1: Horizontal bar chart of combined score (top 25) ---
    ax1 = fig.add_subplot(2, 2, 1)
    top = rows[:25]
    y_pos = np.arange(len(top))
    bars = ax1.barh(y_pos, [r["score"] for r in top],
                    color=[arch_colors.get(r["arch"], "#888") for r in top],
                    edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"#{r['rank']} {r['label']}" for r in top], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Combined Score (lower = better)")
    ax1.set_title("Top 25 Paths by Combined Score", fontsize=11, fontweight="bold")
    for bar, r in zip(bars, top):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{r['score']:.2f}", va="center", fontsize=7, color="#333")

    # --- Panel 2: Delta-v vs Science scatter ---
    ax2 = fig.add_subplot(2, 2, 2)
    has_science = any(r["science"] > 0 for r in rows)
    if has_science:
        for arch, color, marker in [("moon", "#4A90D9", "o"), ("mars", "#D94A4A", "^"), ("direct", "#5BBD72", "s")]:
            subset = [r for r in rows if r["arch"] == arch]
            if subset:
                ax2.scatter([r["dv"] for r in subset], [r["science"] for r in subset],
                            c=color, marker=marker, s=80, alpha=0.8, edgecolors="white",
                            linewidth=0.5, label=f"{arch.title()} flyby", zorder=3)
        for r in rows[:5]:
            ax2.annotate(f"#{r['rank']}", (r["dv"], r["science"]),
                         fontsize=7, fontweight="bold", ha="center",
                         xytext=(0, 8), textcoords="offset points")
        ax2.set_xlabel("Total Δv (km/s)")
        ax2.set_ylabel("Science Sum")
        ax2.set_title("Δv vs Science Tradeoff", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.annotate("← ideal\n(low Δv, high science)", xy=(0.02, 0.98),
                     xycoords="axes fraction", fontsize=8, color="#666",
                     va="top", style="italic")
    else:
        ax2.barh(np.arange(min(25, len(rows))),
                 [r["dv"] for r in rows[:25]],
                 color=colors[:25], edgecolor="white", linewidth=0.5)
        ax2.set_yticks(np.arange(min(25, len(rows))))
        ax2.set_yticklabels([f"#{r['rank']}" for r in rows[:25]], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel("Total Δv (km/s)")
        ax2.set_title("Delta-v Ranking", fontsize=11, fontweight="bold")

    # --- Panel 3: Duration vs Delta-v bubble chart ---
    ax3 = fig.add_subplot(2, 2, 3)
    dvs = [r["dv"] for r in rows]
    durs = [r["duration"] for r in rows]
    scores = [r["score"] for r in rows]
    max_score = max(scores)
    sizes = [200 * (max_score - s + 0.5) / max_score for s in scores]
    scatter = ax3.scatter(durs, dvs, s=sizes, c=scores, cmap="RdYlGn_r",
                          alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
    cb = plt.colorbar(scatter, ax=ax3, shrink=0.8)
    cb.set_label("Combined Score", fontsize=9)
    for r in rows[:5]:
        ax3.annotate(f"#{r['rank']}", (r["duration"], r["dv"]),
                     fontsize=7, fontweight="bold", ha="center",
                     xytext=(0, 8), textcoords="offset points")
    ax3.set_xlabel("Mission Duration (years)")
    ax3.set_ylabel("Total Δv (km/s)")
    ax3.set_title("Duration vs Δv (bubble size = quality)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.annotate("← ideal\n(short, low Δv)", xy=(0.02, 0.02),
                 xycoords="axes fraction", fontsize=8, color="#666", style="italic")

    # --- Panel 4: Asteroid frequency ---
    ax4 = fig.add_subplot(2, 2, 4)
    from collections import Counter
    counts = Counter()
    for r in rows:
        counts[r["a1"]] += 1
        counts[r["a2"]] += 1
        counts[r["a3"]] += 1
    top_ast = counts.most_common(15)
    names = [t[0] for t in top_ast]
    freqs = [t[1] for t in top_ast]
    bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax4.barh(np.arange(len(names)), freqs, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
    ax4.set_yticks(np.arange(len(names)))
    ax4.set_yticklabels(names, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel("Appearances in Top 50")
    ax4.set_title("Most Frequent Asteroids", fontsize=11, fontweight="bold")
    for bar, freq in zip(bars, freqs):
        ax4.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                 str(freq), va="center", fontsize=9, fontweight="bold")

    # --- Architecture legend at bottom ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A90D9", markersize=10, label="Moon flyby"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#D94A4A", markersize=10, label="Mars flyby"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#5BBD72", markersize=10, label="Direct"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <path_to_pkl>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bsp_folder = os.path.join(repo_root, "NOTABLE_ASTEROID_BSPs")
    generic_path = "/Users/rebnoob/Documents/ae105/generic_kernels"

    print(f"Loading {pkl_path}...")
    rows = load_results(pkl_path, bsp_folder, generic_path)
    print(f"  {len(rows)} paths loaded")

    name = os.path.splitext(os.path.basename(pkl_path))[0]
    title = name.replace("_", " ").replace("results ", "").title()

    fig = plot_all(rows, title=title)

    out_path = pkl_path.replace(".pkl", "_viz.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out_path}")

    if sys.platform == "darwin":
        os.system(f'open "{out_path}"')


if __name__ == "__main__":
    main()
