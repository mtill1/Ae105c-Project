"""2D top-down GIF for PARTHENOPE → PSYCHE → THEMIS LT-chain (direct, no flyby).

Same Sims-Flanagan trajectory as the 3D version, just projected onto the
ecliptic XY plane. Easier to read for distance/orbit-shape inspection.
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
from matplotlib.animation import FuncAnimation, PillowWriter
import spiceypy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))

from core import (load_kernels, get_state, solve_lambert,
                   propagate_two_body, MU_SUN, DAY, YEAR,
                   get_id_from_asteroid_name)
from plot_ppt_lt_chain_gif import integrate_lt_leg

_DEFAULT_PKL = 'optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl'
_DEFAULT_OUT = 'Renders/ppt_lt_chain_trajectory_2d.gif'
PKL = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PKL
OUT = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_OUT

N_FRAMES = 240
FPS      = 24


def main():
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)

    triplet = data.get('best_ordering') or data.get('ordering')
    v = data['verified']
    cfg = data['config']
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n))) for n in triplet]
    solver_m_revs = v.get('m_revs', [0, 0, 0])

    eps = v['epochs']
    et_launch = eps['et_launch']
    et_a1_arr = eps['et_a1_arr']; et_a1_dep = eps['et_a1_dep']
    et_a2_arr = eps['et_a2_arr']; et_a2_dep = eps['et_a2_dep']
    et_a3_arr = eps['et_a3_arr']

    print(f'Trajectory: {" → ".join(triplet)} (direct, all-LT)')
    print(f'Mission: {(et_a3_arr-et_launch)/YEAR:.2f} yr   '
          f'Post-launch Δv: {v["post_launch_dv_kms_full"]:.3f} km/s')

    # Build trajectory using the same logic as the 3D version
    legs_lt_data = v['verified_legs']
    legs = []

    # Leg 1: Earth → A1 (LT)
    L = legs_lt_data[0]
    earth_r, _ = get_state('399', L['et_start'])
    a1_arr_r, _ = get_state(a_ids[0], L['et_end'])
    V1, _, _ = solve_lambert(earth_r, a1_arr_r, (L['et_end']-L['et_start'])/DAY,
                              int(solver_m_revs[0]), MU_SUN)
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    rs, ts, _ = integrate_lt_leg(earth_r, V1, L['m_in_kg'], throttles,
                                   L['et_end']-L['et_start'],
                                   cfg['thrust_N'], cfg['isp_elec_s'],
                                   len(throttles))
    legs.append({'r': rs, 't_abs': ts + L['et_start']})

    # Stay at PARTHENOPE
    n_stay = 30
    ts_stay = np.linspace(et_a1_arr, et_a1_dep, n_stay)
    rs_stay = np.array([get_state(a_ids[0], t)[0] for t in ts_stay])
    legs.append({'r': rs_stay, 't_abs': ts_stay})

    # Leg 2: PARTHENOPE → PSYCHE (LT)
    L = legs_lt_data[1]
    a1_lv_r, a1_lv_v = get_state(a_ids[0], L['et_start'])
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    rs, ts, _ = integrate_lt_leg(a1_lv_r, a1_lv_v, L['m_in_kg'], throttles,
                                   L['et_end']-L['et_start'],
                                   cfg['thrust_N'], cfg['isp_elec_s'],
                                   len(throttles))
    legs.append({'r': rs, 't_abs': ts + L['et_start']})

    # Stay at PSYCHE
    ts_stay = np.linspace(et_a2_arr, et_a2_dep, n_stay)
    rs_stay = np.array([get_state(a_ids[1], t)[0] for t in ts_stay])
    legs.append({'r': rs_stay, 't_abs': ts_stay})

    # Leg 3: PSYCHE → THEMIS (LT)
    L = legs_lt_data[2]
    a2_lv_r, a2_lv_v = get_state(a_ids[1], L['et_start'])
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    rs, ts, _ = integrate_lt_leg(a2_lv_r, a2_lv_v, L['m_in_kg'], throttles,
                                   L['et_end']-L['et_start'],
                                   cfg['thrust_N'], cfg['isp_elec_s'],
                                   len(throttles))
    legs.append({'r': rs, 't_abs': ts + L['et_start']})

    all_t = np.concatenate([L['t_abs'] for L in legs])
    all_r = np.concatenate([L['r'] for L in legs])

    bodies = [
        ('Earth',     '399',  '#3498db'),
        ('Mars',      '4',    '#e74c3c'),
        (triplet[0],  a_ids[0], '#9b59b6'),
        (triplet[1],  a_ids[1], '#95a5a6'),
        (triplet[2],  a_ids[2], '#16a085'),
    ]
    sample_ts = np.linspace(et_launch, et_a3_arr, 400)
    body_orbits = {n: np.array([get_state(bid, t)[0] for t in sample_ts])
                    for n, bid, _ in bodies}

    frame_ts = np.linspace(et_launch, et_a3_arr, N_FRAMES)

    AU = 1.496e8

    fig, ax = plt.subplots(figsize=(11, 11), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    # Sun
    ax.scatter([0],[0], color='#fff200', s=400,
                edgecolor='#ffae00', linewidth=2, zorder=10)
    ax.text(0.15*AU, -0.25*AU, 'Sun', color='#fff200', fontsize=10,
             fontweight='bold')

    # Body orbits (faint)
    for name, _, color in bodies:
        pts = body_orbits[name]
        ax.plot(pts[:,0], pts[:,1], color=color, alpha=0.20, linewidth=1.0)

    # Spacecraft trajectory faint full path
    ax.plot(all_r[:,0], all_r[:,1], color='#b3b3b3', alpha=0.20, linewidth=0.8)

    R = 3.6 * AU
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect('equal')

    # Orbit reference rings (1, 2, 3 AU)
    theta = np.linspace(0, 2*np.pi, 200)
    for r_au in [1, 2, 3]:
        ax.plot(r_au*AU*np.cos(theta), r_au*AU*np.sin(theta),
                 color='#444', linewidth=0.4, linestyle=':', alpha=0.6)
        ax.text(r_au*AU + 0.05*AU, 0.02*AU, f'{r_au} AU',
                 color='#666', fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#333')

    body_dots = {}; body_labels = {}
    for name, _, color in bodies:
        size = 12 if name in ('Earth', 'Mars') else 8
        dot, = ax.plot([], [], 'o', color=color, markersize=size,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=5)
        body_dots[name] = dot
        body_labels[name] = ax.text(0,0, '', color=color, fontsize=9,
                                      fontweight='bold', zorder=6)

    sc_line, = ax.plot([], [], '-', color='#00d4ff', linewidth=2.5, alpha=0.95, zorder=7)
    sc_dot,  = ax.plot([], [], 'o', color='#00d4ff', markersize=8,
                        markeredgecolor='white', markeredgewidth=0.7, zorder=8)

    ax.set_title(
        f'PARTHENOPE [S] → PSYCHE [X/M] → THEMIS [C]   direct, all-LT (no flyby)\n'
        f'Launch dv: {v["launch_dv_kms"]:.2f} km/s (impulsive ≤7 cap)   '
        f'Post-launch Δv: {v["post_launch_dv_kms_full"]:.2f} km/s   '
        f'Final mass: {v["m_final_kg_full"]:.0f}/1500 kg   '
        f'Mission: {(et_a3_arr-et_launch)/YEAR:.1f} yr',
        color='white', fontsize=11, pad=15)

    date_box = ax.text(0.02, 0.04, '', transform=ax.transAxes,
                        color='#ffe066', fontsize=12, fontweight='bold',
                        family='monospace')
    leg_box  = ax.text(0.02, 0.08, '', transform=ax.transAxes,
                        color='#cccccc', fontsize=10, family='monospace')

    phase_etps = [et_launch, et_a1_arr, et_a1_dep, et_a2_arr, et_a2_dep, et_a3_arr]
    phase_names = ['Launch', f'Arrive {triplet[0]}', f'Depart {triplet[0]}',
                    f'Arrive {triplet[1]}', f'Depart {triplet[1]}',
                    f'Arrive {triplet[2]}']

    def current_phase(et):
        for i, ep in enumerate(phase_etps):
            if et < ep:
                return phase_names[i-1] if i > 0 else 'Pre-launch'
        return 'Mission complete'

    def init():
        for d_ in body_dots.values():
            d_.set_data([], [])
        sc_line.set_data([], [])
        sc_dot.set_data([], [])
        return list(body_dots.values()) + [sc_line, sc_dot]

    def animate(i):
        et = frame_ts[i]
        for name, bid, _color in bodies:
            r = get_state(bid, et)[0]
            body_dots[name].set_data([r[0]], [r[1]])
            body_labels[name].set_position((r[0] + 0.05*AU, r[1] + 0.05*AU))
            body_labels[name].set_text(name)
        mask = all_t <= et
        if mask.any():
            sc_line.set_data(all_r[mask,0], all_r[mask,1])
            j = max(1, min(np.searchsorted(all_t, et), len(all_t)-1))
            t0, t1 = all_t[j-1], all_t[j]
            x0, x1 = all_r[j-1], all_r[j]
            f = (et-t0)/(t1-t0) if t1 > t0 else 0
            r_sc = x0 + f*(x1-x0)
            sc_dot.set_data([r_sc[0]], [r_sc[1]])
        date_box.set_text(spiceypy.et2utc(et, 'C', 0))
        leg_box.set_text(f'phase: {current_phase(et)}')
        return (list(body_dots.values()) + list(body_labels.values())
                + [sc_line, sc_dot, date_box, leg_box])

    print(f'Rendering {N_FRAMES} frames -> {OUT}')
    ani = FuncAnimation(fig, animate, init_func=init, frames=N_FRAMES,
                         interval=1000/FPS, blit=False)
    os.makedirs('Renders', exist_ok=True)
    ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=110)
    print(f'Saved: {OUT}')
    plt.close(fig)


if __name__ == '__main__':
    main()
