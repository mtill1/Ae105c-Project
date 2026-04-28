"""Visualize hybrid LT results: bar chart comparing CC vs EE final mass per triplet."""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot_hybrid(results, out_path, top_n=20):
    valid = [r for r in results if '_error' not in r]
    valid.sort(key=lambda r: -r['m_best_kg'])
    top = valid[:top_n]

    labels = [' → '.join(n[:8] for n in r['_names']) for r in top]
    m_cc = [r['m_baseline_CC_kg'] for r in top]
    m_best = [r['m_best_kg'] for r in top]
    archs = [r['best_arch'] for r in top]
    imp_dv = [r['_dv_total_impulsive'] for r in top]

    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    fig.patch.set_facecolor('#0a0a1f')
    fig.suptitle('Hybrid Low-Thrust vs All-Chemical — Top 20 by Delivered Mass',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # --- Panel 1: side-by-side bar chart ---
    ax = axes[0]
    ax.set_facecolor('#0a0a1f')
    y = np.arange(len(top))
    h = 0.35
    bars_cc   = ax.barh(y + h/2, m_cc,   h, color='#D94A4A', label='All Chemical (CC)',
                         edgecolor='white', linewidth=0.5)
    bars_best = ax.barh(y - h/2, m_best, h, color='#5BBD72', label='Best Hybrid (EE/CE/EC)',
                         edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Final Delivered Mass (kg)', color='white', fontsize=11)
    ax.set_title('Chemical vs Hybrid: Delivered Mass\n(from 1500 kg launch mass)',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.legend(facecolor='#1a1a3a', edgecolor='#666', labelcolor='white', fontsize=10)

    for bar, m, arch in zip(bars_best, m_best, archs):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{m:.0f} kg [{arch}]', va='center', fontsize=7, color='#5BBD72',
                fontweight='bold')
    for bar, m in zip(bars_cc, m_cc):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{m:.0f} kg', va='center', fontsize=7, color='#D94A4A')

    # --- Panel 2: improvement factor ---
    ax2 = axes[1]
    ax2.set_facecolor('#0a0a1f')
    factors = [b / max(c, 1) for b, c in zip(m_best, m_cc)]
    colors = ['#FFD54F' if f > 3 else '#5BBD72' if f > 2 else '#4FC3F7' for f in factors]
    bars_f = ax2.barh(y, factors, 0.6, color=colors, edgecolor='white', linewidth=0.5)
    ax2.axvline(1.0, color='#D94A4A', linestyle='--', lw=1.5, alpha=0.7, label='Break-even')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8, color='white')
    ax2.invert_yaxis()
    ax2.set_xlabel('Mass Improvement Factor (hybrid / chemical)', color='white', fontsize=11)
    ax2.set_title('How much more payload does electric propulsion deliver?',
                  color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    for s in ax2.spines.values(): s.set_edgecolor('#666')

    for bar, f, dv in zip(bars_f, factors, imp_dv):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{f:.1f}× (Δv={dv:.1f})', va='center', fontsize=8, color='white')

    ax2.legend(facecolor='#1a1a3a', edgecolor='#666', labelcolor='white', fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl = os.path.join(repo, 'optimal_asteroid_paths/pkl/results_hybrid_lt.pkl')
    with open(pkl, 'rb') as f:
        results = pickle.load(f)

    out = os.path.join(repo, 'Renders/Asteroid_Plots/hybrid_lt_comparison.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_hybrid(results, out)


if __name__ == '__main__':
    main()
