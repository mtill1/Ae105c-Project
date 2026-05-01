"""3D GIF for PARTHENOPE → PSYCHE → THEMIS LT-chain (direct, no flyby).

Integrates the Sims-Flanagan throttle profile through each leg so the rendered
spacecraft path actually shows the curvature/spiraling of low-thrust flight,
not just a Lambert arc.
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

PKL = 'optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl'
OUT = 'Renders/ppt_lt_chain_trajectory.gif'

N_FRAMES = 240
FPS      = 24


def integrate_lt_leg(r0, v0, m0, throttles, tof_sec, thrust_N, isp_s, nseg,
                       samples_per_half=4):
    """Sims-Flanagan forward integration. Each segment = half-coast, kick,
    half-coast. Returns (r_array, t_array, m_final).

    Total time advanced per segment = dt_seg (sum of two half-coasts of
    dt_seg/2 each). Total leg time = nseg × dt_seg = tof_sec exactly.
    """
    G0 = 9.80665
    dt_seg = tof_sec / nseg
    half_coast = dt_seg / 2.0
    sub_dt = half_coast / samples_per_half

    r = np.array(r0, float).copy()
    v = np.array(v0, float).copy()
    m = float(m0)
    t = 0.0
    out_r = [r.copy()]
    out_t = [0.0]

    for i in range(nseg):
        u = np.asarray(throttles[i])

        # Half-coast 1
        for _ in range(samples_per_half):
            r, v = propagate_two_body(r, v, sub_dt, MU_SUN)
            t += sub_dt
            out_r.append(r.copy()); out_t.append(t)

        # Impulsive kick at segment midpoint
        dv_max_kms = thrust_N * dt_seg / m / 1e3
        dv_vec = u * dv_max_kms
        v = v + dv_vec
        m = m * np.exp(-np.linalg.norm(dv_vec) * 1e3 / (isp_s * G0))

        # Half-coast 2
        for _ in range(samples_per_half):
            r, v = propagate_two_body(r, v, sub_dt, MU_SUN)
            t += sub_dt
            out_r.append(r.copy()); out_t.append(t)

    return np.array(out_r), np.array(out_t), m


def main():
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)

    triplet = data['best_ordering']
    v = data['verified']
    cfg = data['config']
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n))) for n in triplet]

    eps = v['epochs']
    et_launch = eps['et_launch']
    et_a1_arr = eps['et_a1_arr']; et_a1_dep = eps['et_a1_dep']
    et_a2_arr = eps['et_a2_arr']; et_a2_dep = eps['et_a2_dep']
    et_a3_arr = eps['et_a3_arr']

    print(f'Trajectory: {" → ".join(triplet)} (direct, all-LT)')
    print(f'Launch: {spiceypy.et2utc(et_launch,"C",0)}')
    print(f'Final:  {spiceypy.et2utc(et_a3_arr,"C",0)}')
    print(f'Duration: {(et_a3_arr-et_launch)/YEAR:.2f} yr')
    print(f'Post-launch Δv: {v["post_launch_dv_kms_full"]:.3f} km/s')

    # Build trajectory by integrating each LT leg with its actual throttles
    # plus the stay-at-asteroid coast periods.
    legs = []
    leg_event_etps = [et_launch]   # start of each segment
    leg_labels = []

    legs_lt_data = v['verified_legs']
    # m_revs from the solver — required to reconstruct the correct Lambert
    # initial velocity at Earth. Solver picks multi-rev branches; the GIF
    # must use the SAME branch the solver used or the spacecraft will
    # follow a completely different trajectory (and "teleport" past the
    # asteroid).
    solver_m_revs = v.get('m_revs', [0, 0, 0])

    # Leg 1: Earth → A1 (LT)
    L = legs_lt_data[0]
    et_start, et_end = L['et_start'], L['et_end']
    earth_r, earth_v = get_state('399', et_start)
    a1_arr_r, _ = get_state(a_ids[0], et_end)
    # Use the solver's chosen Lambert revolution count for the launch leg
    V1, V2, _ = solve_lambert(earth_r, a1_arr_r, (et_end-et_start)/DAY,
                                int(solver_m_revs[0]), MU_SUN)
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    rs, ts, m_end = integrate_lt_leg(
        earth_r, V1, L['m_in_kg'], throttles, et_end-et_start,
        cfg['thrust_N'], cfg['isp_elec_s'], len(throttles))
    legs.append({'r': rs, 't_abs': ts + et_start, 'label': L['label']})
    leg_event_etps.append(et_end); leg_labels.append('Arrive PARTHENOPE')

    # Stay at PARTHENOPE
    n_stay = 30
    ts_stay = np.linspace(et_a1_arr, et_a1_dep, n_stay)
    rs_stay = np.array([get_state(a_ids[0], t)[0] for t in ts_stay])
    legs.append({'r': rs_stay, 't_abs': ts_stay, 'label': 'Stay PARTHENOPE'})
    leg_event_etps.append(et_a1_dep); leg_labels.append('Depart PARTHENOPE')

    # Leg 2: PARTHENOPE → PSYCHE (LT)
    L = legs_lt_data[1]
    et_start, et_end = L['et_start'], L['et_end']
    a1_lv_r, a1_lv_v = get_state(a_ids[0], et_start)
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    # Initial v: spacecraft co-orbits with PARTHENOPE
    rs, ts, m_end = integrate_lt_leg(
        a1_lv_r, a1_lv_v, L['m_in_kg'], throttles, et_end-et_start,
        cfg['thrust_N'], cfg['isp_elec_s'], len(throttles))
    legs.append({'r': rs, 't_abs': ts + et_start, 'label': L['label']})
    leg_event_etps.append(et_end); leg_labels.append('Arrive PSYCHE')

    # Stay at PSYCHE
    ts_stay = np.linspace(et_a2_arr, et_a2_dep, n_stay)
    rs_stay = np.array([get_state(a_ids[1], t)[0] for t in ts_stay])
    legs.append({'r': rs_stay, 't_abs': ts_stay, 'label': 'Stay PSYCHE'})
    leg_event_etps.append(et_a2_dep); leg_labels.append('Depart PSYCHE')

    # Leg 3: PSYCHE → THEMIS (LT)
    L = legs_lt_data[2]
    et_start, et_end = L['et_start'], L['et_end']
    a2_lv_r, a2_lv_v = get_state(a_ids[1], et_start)
    throttles = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    rs, ts, m_end = integrate_lt_leg(
        a2_lv_r, a2_lv_v, L['m_in_kg'], throttles, et_end-et_start,
        cfg['thrust_N'], cfg['isp_elec_s'], len(throttles))
    legs.append({'r': rs, 't_abs': ts + et_start, 'label': L['label']})
    leg_event_etps.append(et_end); leg_labels.append('Arrive THEMIS')

    all_t = np.concatenate([L['t_abs'] for L in legs])
    all_r = np.concatenate([L['r'] for L in legs])
    print(f'Trajectory sampled at {len(all_t)} points across {len(legs)} legs')

    bodies = [
        ('Earth',     '399',  '#3498db'),
        ('Mars',      '4',    '#e74c3c'),
        (triplet[0],  a_ids[0], '#9b59b6'),
        (triplet[1],  a_ids[1], '#95a5a6'),
        (triplet[2],  a_ids[2], '#16a085'),
    ]
    sample_ts = np.linspace(et_launch, et_a3_arr, 300)
    body_orbits = {n: np.array([get_state(bid, t)[0] for t in sample_ts])
                    for n, bid, _ in bodies}

    frame_ts = np.linspace(et_launch, et_a3_arr, N_FRAMES)

    fig = plt.figure(figsize=(13, 9), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
    ax.scatter([0],[0],[0], color='#fff200', s=180,
                edgecolor='#ffae00', linewidth=1)
    for name, _, color in bodies:
        pts = body_orbits[name]
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=0.18, linewidth=0.8)
    ax.plot(all_r[:,0], all_r[:,1], all_r[:,2],
            color='#b3b3b3', alpha=0.20, linewidth=0.8)

    ax.set_xlabel('X (km)', color='#ddd', labelpad=8)
    ax.set_ylabel('Y (km)', color='#ddd', labelpad=8)
    ax.set_zlabel('Z (km)', color='#ddd', labelpad=8)
    ax.tick_params(colors='#888', labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color('#444')
        axis.set_pane_color((0.05, 0.06, 0.08, 1.0))
    AU = 1.496e8
    R = 4.0 * AU
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R*0.4, R*0.4)
    ax.set_box_aspect((1, 1, 0.4))

    body_dots = {}; body_labels = {}
    for name, _, color in bodies:
        size = 9 if name in ('Earth', 'Mars') else 6
        dot, = ax.plot([], [], [], 'o', color=color, markersize=size,
                        markeredgecolor='white', markeredgewidth=0.5)
        body_dots[name] = dot
        body_labels[name] = ax.text(0,0,0, '', color=color, fontsize=9,
                                      fontweight='bold')

    sc_line, = ax.plot([], [], [], '-', color='#00d4ff', linewidth=2.0, alpha=0.95)
    sc_dot,  = ax.plot([], [], [], 'o', color='#00d4ff', markersize=7,
                        markeredgecolor='white', markeredgewidth=0.7)

    ax.text2D(0.02, 0.97,
        f'PARTHENOPE [S] → PSYCHE [X/M] → THEMIS [C]   '
        f'direct, all-LT (no flyby)',
        transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    ax.text2D(0.02, 0.93,
        f'Launch dv: {v["launch_dv_kms"]:.2f} km/s (impulsive, ≤7 cap)   '
        f'Post-launch Δv: {v["post_launch_dv_kms_full"]:.2f} km/s   '
        f'Final mass: {v["m_final_kg_full"]:.0f}/1500 kg   '
        f'Mission: {(et_a3_arr-et_launch)/YEAR:.1f} yr',
        transform=ax.transAxes, color='#9be7ff', fontsize=10)
    date_box = ax.text2D(0.02, 0.05, '', transform=ax.transAxes,
                          color='#ffe066', fontsize=12, fontweight='bold',
                          family='monospace')
    leg_box  = ax.text2D(0.02, 0.10, '', transform=ax.transAxes,
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
            d_.set_data([], []); d_.set_3d_properties([])
        sc_line.set_data([], []); sc_line.set_3d_properties([])
        sc_dot.set_data([], []); sc_dot.set_3d_properties([])
        return list(body_dots.values()) + [sc_line, sc_dot]

    def animate(i):
        et = frame_ts[i]
        for name, bid, color in bodies:
            r = get_state(bid, et)[0]
            body_dots[name].set_data([r[0]], [r[1]])
            body_dots[name].set_3d_properties([r[2]])
            body_labels[name].set_position((r[0], r[1]))
            body_labels[name].set_3d_properties(r[2] + 0.05*AU, zdir='z')
            body_labels[name].set_text(' ' + name)
        mask = all_t <= et
        if mask.any():
            sc_line.set_data(all_r[mask,0], all_r[mask,1])
            sc_line.set_3d_properties(all_r[mask,2])
            j = max(1, min(np.searchsorted(all_t, et), len(all_t)-1))
            t0, t1 = all_t[j-1], all_t[j]
            x0, x1 = all_r[j-1], all_r[j]
            f = (et-t0)/(t1-t0) if t1 > t0 else 0
            r_sc = x0 + f*(x1-x0)
            sc_dot.set_data([r_sc[0]], [r_sc[1]])
            sc_dot.set_3d_properties([r_sc[2]])
        date_box.set_text(spiceypy.et2utc(et, 'C', 0))
        leg_box.set_text(f'phase: {current_phase(et)}')
        ax.view_init(elev=30, azim=-60 + 60*i/N_FRAMES)
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
