"""Render the PARTHENOPE → PSYCHE → THEMIS Earth+Mars-GA chain as a 3D GIF."""
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

PKL = 'optimal_asteroid_paths/pkl/parthenope_psyche_themis_earth_mars_chain.pkl'
OUT = 'Renders/parthenope_psyche_themis_earth_mars_chain.gif'

N_FRAMES = 240
FPS      = 24


def main():
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)

    triplet = data['triplet']
    fb1_name = data['flyby1']
    fb2_name = data['flyby2']
    b = data['best']

    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n)))
              for n in triplet]

    et_launch = b['et_launch']
    et_fb1    = b['et_fb1']
    et_fb2    = b['et_fb2']
    et_arr_1  = b['et_arrive_1']
    et_stay_1 = b['et_stay_1']
    et_arr_2  = b['et_arrive_2']
    et_stay_2 = b['et_stay_2']
    et_arr_3  = b['et_arrive_3']
    m_revs    = b['m_revs']

    print(f'Triplet: {" → ".join(triplet)}')
    print(f'Architecture: {fb1_name.title()} GA + {fb2_name.title()} GA')
    print(f'Mission: {spiceypy.et2utc(et_launch,"C",0)} → '
          f'{spiceypy.et2utc(et_arr_3,"C",0)}')
    print(f'Duration: {(et_arr_3-et_launch)/YEAR:.2f} yr')
    print(f'Total Δv: {b["delta_v_total"]:.3f} km/s')

    # ---- Build trajectory by sampling each leg ----
    # 7 legs total: Earth→FB1, FB1→FB2, FB2→A1, stay A1, A1→A2, stay A2, A2→A3
    # The "stay" legs are co-orbital with the asteroid.
    SAMPLES = 80

    fb1_id = '399' if fb1_name == 'earth' else ('4' if fb1_name == 'mars' else '301')
    fb2_id = '399' if fb2_name == 'earth' else ('4' if fb2_name == 'mars' else '301')

    leg_specs = [
        ('Earth → ' + fb1_name.title(),
          '399',  et_launch,  fb1_id, et_fb1, m_revs[0]),
        (fb1_name.title() + ' → ' + fb2_name.title(),
          fb1_id, et_fb1,     fb2_id, et_fb2, m_revs[1]),
        (fb2_name.title() + ' → ' + triplet[0],
          fb2_id, et_fb2,     a_ids[0], et_arr_1, m_revs[2]),
        (f'Stay at {triplet[0]}',
          a_ids[0], et_arr_1, a_ids[0], et_stay_1, None),
        (f'{triplet[0]} → {triplet[1]}',
          a_ids[0], et_stay_1, a_ids[1], et_arr_2,  m_revs[3]),
        (f'Stay at {triplet[1]}',
          a_ids[1], et_arr_2, a_ids[1], et_stay_2, None),
        (f'{triplet[1]} → {triplet[2]}',
          a_ids[1], et_stay_2, a_ids[2], et_arr_3,  m_revs[4]),
    ]

    legs = []
    for label, b0, et0, b1, et1, mrev in leg_specs:
        ts = np.linspace(et0, et1, SAMPLES)
        if mrev is None:
            xs = np.array([get_state(b0, t)[0] for t in ts])
        else:
            r0, _  = get_state(b0, et0)
            r1, _  = get_state(b1, et1)
            tof = et1 - et0
            # Special-case Earth GA leg: search multi-rev branches with v_inf>=3
            if b0 == '399' and b1 == '399':
                _, earth_v0 = get_state('399', et0)
                best = None
                for m_try in (0, 1, -1, 2, -2):
                    V1, V2, ef = solve_lambert(r0, r1, tof/DAY, m_try, MU_SUN)
                    if ef != 1: continue
                    ldv = np.linalg.norm(V1 - earth_v0)
                    if ldv < 3.0: continue
                    if best is None or ldv < best[0]:
                        best = (ldv, V1)
                V1 = best[1] if best else None
            else:
                V1, V2, ef = solve_lambert(r0, r1, tof/DAY, mrev, MU_SUN)
                if ef != 1: V1 = None
            if V1 is None:
                xs = np.array([r0 + (r1-r0)*((t-et0)/tof) for t in ts])
            else:
                xs = []
                for t in ts:
                    rt, _ = propagate_two_body(r0, V1, t-et0, MU_SUN)
                    xs.append(rt)
                xs = np.array(xs)
        legs.append({'label': label, 'ts': ts, 'xs': xs, 'mode': mrev})

    all_ts = np.concatenate([L['ts'] for L in legs])
    all_xs = np.concatenate([L['xs'] for L in legs])
    print(f'Trajectory: {len(legs)} legs, {len(all_ts)} sample points')

    # ---- Bodies to display ----
    bodies = [
        ('Earth',     '399',  '#3498db'),
        ('Mars',      '4',    '#e74c3c'),
        (triplet[0],  a_ids[0], '#9b59b6'),  # PARTHENOPE (S)
        (triplet[1],  a_ids[1], '#95a5a6'),  # PSYCHE (X/M)
        (triplet[2],  a_ids[2], '#16a085'),  # THEMIS (C)
    ]
    sample_ts = np.linspace(et_launch, et_arr_3, 300)
    body_orbits = {n: np.array([get_state(bid, t)[0] for t in sample_ts])
                    for n, bid, _ in bodies}

    frame_ts = np.linspace(et_launch, et_arr_3, N_FRAMES)

    # ---- Figure ----
    fig = plt.figure(figsize=(13, 9), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
    ax.scatter([0],[0],[0], color='#fff200', s=180,
                edgecolor='#ffae00', linewidth=1)

    for name, _, color in bodies:
        pts = body_orbits[name]
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=0.18, linewidth=0.8)

    ax.plot(all_xs[:,0], all_xs[:,1], all_xs[:,2],
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
        f'PARTHENOPE [S]  →  PSYCHE [X/M]  →  THEMIS [C]   '
        f'via {fb1_name.upper()} GA + {fb2_name.upper()} GA',
        transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    ax.text2D(0.02, 0.93,
        f'Total Δv: {b["delta_v_total"]:.2f} km/s   |   '
        f'Mission: {(et_arr_3-et_launch)/YEAR:.1f} yr   |   '
        f'Earth GA: 0.74 km/s powered, Mars GA: ballistic',
        transform=ax.transAxes, color='#9be7ff', fontsize=10)
    date_box = ax.text2D(0.02, 0.05, '', transform=ax.transAxes,
                          color='#ffe066', fontsize=12, fontweight='bold',
                          family='monospace')
    leg_box  = ax.text2D(0.02, 0.10, '', transform=ax.transAxes,
                          color='#cccccc', fontsize=10, family='monospace')

    phase_etps = [et_launch, et_fb1, et_fb2, et_arr_1, et_stay_1,
                   et_arr_2, et_stay_2, et_arr_3]
    phase_names = ['Launch',
                    f'{fb1_name.title()} GA flyby',
                    f'{fb2_name.title()} GA flyby',
                    f'Arrive {triplet[0]}', f'Depart {triplet[0]}',
                    f'Arrive {triplet[1]}', f'Depart {triplet[1]}',
                    f'Arrive {triplet[2]}']

    def current_phase(et):
        for i, ep in enumerate(phase_etps):
            if et < ep:
                return phase_names[i-1] if i > 0 else 'Pre-launch'
        return 'Mission complete'

    def init():
        for d in body_dots.values():
            d.set_data([], []); d.set_3d_properties([])
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
        mask = all_ts <= et
        if mask.any():
            sc_line.set_data(all_xs[mask,0], all_xs[mask,1])
            sc_line.set_3d_properties(all_xs[mask,2])
            j = max(1, min(np.searchsorted(all_ts, et), len(all_ts)-1))
            t0,t1 = all_ts[j-1], all_ts[j]
            x0,x1 = all_xs[j-1], all_xs[j]
            f = (et-t0)/(t1-t0) if t1>t0 else 0
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
