"""Thrust-vs-time chart for the v2 PARTHENOPE → PSYCHE → THEMIS LT-chain
(direct, no flyby).

3 LT legs (Earth→PARTHENOPE, PARTHENOPE→PSYCHE, PSYCHE→THEMIS) plus the two
asteroid stays in between. No Mars phase marker (this trajectory doesn't use
a flyby).
"""
import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spiceypy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))

from core import load_kernels, YEAR

PKL = 'optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl'
OUT = 'Renders/ppt_lt_chain_thrust_vs_time_v2.png'


def main():
    load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)

    triplet = data['best_ordering']
    v = data['verified']
    cfg = data['config']
    eps = v['epochs']

    et_launch = eps['et_launch']
    et_a1_arr = eps['et_a1_arr']; et_a1_dep = eps['et_a1_dep']
    et_a2_arr = eps['et_a2_arr']; et_a2_dep = eps['et_a2_dep']
    et_a3_arr = eps['et_a3_arr']
    mission_yr = (et_a3_arr - et_launch) / YEAR

    thrust_max_N = cfg['thrust_N']

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    # =====================================================================
    # Top: thrust magnitude
    # =====================================================================
    ax = axes[0]
    leg_colors = ['#3498db', '#9b59b6', '#16a085']
    leg_labels_plot = []

    for li, L in enumerate(v['verified_legs']):
        tp = L['thrust_profile']
        nseg = len(tp['thrust_magnitude_N'])
        seg_dt = tp['segment_dt_yr']
        leg_start_yr = (L['et_start'] - et_launch) / YEAR
        # Step plot
        t_steps, F_steps = [], []
        for k in range(nseg):
            t0 = leg_start_yr + k*seg_dt
            t1 = leg_start_yr + (k+1)*seg_dt
            F = tp['thrust_magnitude_N'][k] * 1000  # mN
            t_steps += [t0, t1]
            F_steps += [F, F]
        # Cleaner labels
        nice = (L['label'].replace('A1', triplet[0])
                          .replace('A2', triplet[1])
                          .replace('A3', triplet[2]))
        ax.fill_between(t_steps, 0, F_steps, color=leg_colors[li],
                         alpha=0.55, label=nice)
        ax.plot(t_steps, F_steps, color=leg_colors[li], linewidth=1.2)
        leg_labels_plot.append(nice)

    # Stay periods: zero thrust (visualized as gray bands)
    for et_s, et_e, asteroid in [(et_a1_arr, et_a1_dep, triplet[0]),
                                    (et_a2_arr, et_a2_dep, triplet[1])]:
        t_s_yr = (et_s - et_launch) / YEAR
        t_e_yr = (et_e - et_launch) / YEAR
        ax.axvspan(t_s_yr, t_e_yr, color='#95a5a6', alpha=0.18,
                    zorder=0)
        ax.text((t_s_yr + t_e_yr)/2, thrust_max_N*1000*0.05,
                 f'Stay\n{asteroid}', ha='center', va='bottom',
                 fontsize=8, color='#555', style='italic')

    ax.axhline(thrust_max_N * 1000, color='red', linestyle='--', linewidth=1,
                label=f'Engine max {thrust_max_N*1000:.0f} mN')
    ax.set_ylabel('Thrust magnitude (mN)', fontsize=12)
    ax.set_title(
        f'PARTHENOPE [S] → PSYCHE [X/M] → THEMIS [C]   direct, all-LT (no flyby)\n'
        f'Total post-launch Δv: {v["post_launch_dv_kms_full"]:.3f} km/s   '
        f'(launch Δv {v["launch_dv_kms"]:.3f} km/s impulsive, ≤ 7 cap, '
        f'excluded from objective)\n'
        f'Final mass: {v["m_final_kg_full"]:.0f} kg / 1500 kg LT-start ('
        f'{100*v["m_final_kg_full"]/1500:.1f}%)   |   '
        f'Mission: {mission_yr:.2f} yr   |   Engine: Isp 3100 s, ≤0.30 N',
        fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, thrust_max_N * 1000 * 1.18)

    # Phase markers (no Mars flyby!)
    phase_markers = [
        (0,                                   'Launch'),
        ((et_a1_arr - et_launch)/YEAR,        f'Arr.\n{triplet[0]}'),
        ((et_a1_dep - et_launch)/YEAR,        f'Dep.\n{triplet[0]}'),
        ((et_a2_arr - et_launch)/YEAR,        f'Arr.\n{triplet[1]}'),
        ((et_a2_dep - et_launch)/YEAR,        f'Dep.\n{triplet[1]}'),
        ((et_a3_arr - et_launch)/YEAR,        f'Arr.\n{triplet[2]}'),
    ]
    for t, label in phase_markers:
        ax.axvline(t, color='#34495e', linestyle=':', linewidth=0.7, alpha=0.7)
        ax.annotate(label, xy=(t, thrust_max_N*1000*1.05),
                     xytext=(t, thrust_max_N*1000*1.10), ha='center',
                     fontsize=8, color='#34495e')

    # =====================================================================
    # Middle: spacecraft mass
    # =====================================================================
    ax2 = axes[1]
    masses = [(0, 1500.0)]
    for L in v['verified_legs']:
        leg_end_yr = (L['et_end'] - et_launch) / YEAR
        masses.append((leg_end_yr, L['m_out_kg']))
    mt = [m[0] for m in masses]; mv = [m[1] for m in masses]
    ax2.step(mt, mv, where='post', linewidth=2, color='#2980b9')
    ax2.fill_between(mt, 0, mv, step='post', color='#3498db', alpha=0.3)
    ax2.set_ylabel('Spacecraft mass (kg)', fontsize=11)
    ax2.set_ylim(0, 1500 * 1.1)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Annotate masses at events
    for t, m in masses:
        ax2.annotate(f'{m:.0f} kg', xy=(t, m), xytext=(t, m+50),
                      ha='center', fontsize=8, color='#2c3e50')

    # =====================================================================
    # Bottom: cumulative integrated Δv
    # =====================================================================
    ax3 = axes[2]
    cum_t = [0]; cum_dv = [0]
    running = 0.0
    for L in v['verified_legs']:
        # within each leg, accumulate dv segment by segment
        tp = L['thrust_profile']
        nseg = len(tp['thrust_magnitude_N'])
        seg_dt_yr = tp['segment_dt_yr']
        leg_start_yr = (L['et_start'] - et_launch) / YEAR
        # Use saved per-segment thrust magnitude * dt to estimate per-segment dv
        # (this gives an approximate cumulative; the leg-end values match exact)
        # Actual per-segment integrated dv ≈ thrust_N × dt / m (using mean m).
        # For visualization, just lerp leg total dv across the leg time.
        leg_end_yr = (L['et_end'] - et_launch) / YEAR
        for k in range(1, nseg+1):
            frac = k / nseg
            cum_t.append(leg_start_yr + frac * (leg_end_yr - leg_start_yr))
            cum_dv.append(running + frac * L['dv_integral_kms'])
        running += L['dv_integral_kms']
        # Also stay periods (no dv added)
        # the next leg's start is at et_dep; already handled by leg_start_yr next time

    ax3.plot(cum_t, cum_dv, color='#c0392b', linewidth=2)
    ax3.fill_between(cum_t, 0, cum_dv, color='#e74c3c', alpha=0.3)
    ax3.set_ylabel('Cumulative\nintegrated Δv (km/s)', fontsize=11)
    ax3.set_xlabel('Mission elapsed time (years from launch)', fontsize=12)
    ax3.set_ylim(0, max(cum_dv) * 1.1)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.text(cum_t[-1], cum_dv[-1], f'  {cum_dv[-1]:.2f} km/s',
              fontsize=10, color='#922b21', va='center')

    plt.tight_layout()
    os.makedirs('Renders', exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'Saved: {OUT}')
    plt.close(fig)


if __name__ == '__main__':
    main()
