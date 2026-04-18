"""Animated 2D GIFs for the top paths (generalized for any flyby body)."""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_kernels, two_body_sim, solve_lambert, get_state, MU_SUN, DAY, YEAR

AU = 149_597_870.7

FLYBY_META = {
    '399': {'name': 'Earth', 'color': '#4FC3F7'},
    '301': {'name': 'Moon',  'color': '#BDBDBD'},
    '4':   {'name': 'Mars',  'color': '#FF8A65'},
}


def _best_lambert_min_launch(r0, v0, r1, tof_days, max_m=2):
    best = (None, -1, np.inf)
    for m in range(max_m + 1):
        V1, _, ef = solve_lambert(r0, r1, tof_days, m, MU_SUN)
        if ef == 1:
            ldv = np.linalg.norm(V1 - v0)
            if ldv < best[2]:
                best = (V1, 1, ldv)
    return best[0] if best[0] is not None else np.zeros(3)


def build_legs(res, a_ids, flyby_id):
    r_e, v_e   = get_state('399',    res['et_launch'])
    r_fb, v_fb = get_state(flyby_id, res['et_flyby'])
    r_a1_arr,_ = get_state(a_ids[0], res['et_arrive_1'])
    r_a1_lv,_  = get_state(a_ids[0], res['et_stay_1'])
    r_a2_arr,_ = get_state(a_ids[1], res['et_arrive_2'])
    r_a2_lv,_  = get_state(a_ids[1], res['et_stay_2'])
    r_a3_arr,_ = get_state(a_ids[2], res['et_arrive_3'])

    legs_raw = [
        ('399', flyby_id, res['et_launch'],   res['et_flyby'],    r_e,     v_e,     r_fb),
        (flyby_id, a_ids[0], res['et_flyby'], res['et_arrive_1'], r_fb,    v_fb,    r_a1_arr),
        (a_ids[0], a_ids[1], res['et_stay_1'],res['et_arrive_2'], r_a1_lv, None,    r_a2_arr),
        (a_ids[1], a_ids[2], res['et_stay_2'],res['et_arrive_3'], r_a2_lv, None,    r_a3_arr),
    ]
    stay_segments = [
        (res['et_arrive_1'], res['et_stay_1'], a_ids[0]),
        (res['et_arrive_2'], res['et_stay_2'], a_ids[1]),
    ]

    xs, ys, ets = [], [], []
    for li, (b1, b2, et0, etf, r0, v0, rf) in enumerate(legs_raw):
        tof_days = (etf - et0) / DAY
        # For Earth GA leg 0, pick the m that minimizes launch dv.
        if li == 0 and str(flyby_id) == '399':
            v_dep = _best_lambert_min_launch(r0, v0, rf, tof_days)
        else:
            v_dep, _, _ = solve_lambert(r0, rf, tof_days, 0, MU_SUN)
        X, T = two_body_sim(etf - et0, np.concatenate([r0, v_dep]), MU_SUN)
        for pt, t in zip(X, T):
            xs.append(pt[0]); ys.append(pt[1]); ets.append(et0 + float(t))

        if li in (1, 2):
            st0, stf, b_id = stay_segments[li - 1]
            for tt in np.linspace(st0, stf, 20):
                r, _ = get_state(b_id, float(tt))
                xs.append(r[0]); ys.append(r[1]); ets.append(float(tt))

    return np.array(xs), np.array(ys), np.array(ets)


def body_orbit_xy(body_id, et0, etf, n=300):
    ts = np.linspace(et0, etf, n)
    pts = np.array([get_state(str(body_id), float(t))[0] for t in ts])
    return pts[:, 0] / AU, pts[:, 1] / AU


def make_gif(res, a_names, a_ids, title, out_path, n_frames=120):
    flyby_id = str(res.get('flyby_body', '4'))
    fb_name  = FLYBY_META.get(flyby_id, {}).get('name', flyby_id)
    fb_color = FLYBY_META.get(flyby_id, {}).get('color', '#BBBBBB')

    sc_x, sc_y, sc_et = build_legs(res, a_ids, flyby_id)
    idxs = np.linspace(0, len(sc_x) - 1, n_frames).astype(int)
    et0, etf = res['et_launch'], res['et_arrive_3']

    e_x, e_y   = body_orbit_xy('399', et0, etf)
    a1_x, a1_y = body_orbit_xy(a_ids[0], et0, etf)
    a2_x, a2_y = body_orbit_xy(a_ids[1], et0, etf)
    a3_x, a3_y = body_orbit_xy(a_ids[2], et0, etf)
    if flyby_id != '399':
        fb_x, fb_y = body_orbit_xy(flyby_id, et0, etf)
    else:
        fb_x, fb_y = e_x, e_y

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    all_x = np.concatenate([e_x, fb_x, a1_x, a2_x, a3_x, sc_x / AU])
    all_y = np.concatenate([e_y, fb_y, a1_y, a2_y, a3_y, sc_y / AU])
    lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_xlabel('X (AU)', color='white')
    ax.set_ylabel('Y (AU)', color='white')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')

    ax.plot(e_x, e_y, ':', color='#4FC3F7', lw=1, alpha=0.7, label='Earth')
    if flyby_id != '399':
        ax.plot(fb_x, fb_y, ':', color=fb_color, lw=1, alpha=0.7, label=fb_name)
    ax.plot(a1_x, a1_y, ':', color='#D94A4A', lw=0.9, alpha=0.5, label=a_names[0])
    ax.plot(a2_x, a2_y, ':', color='#5BBD72', lw=0.9, alpha=0.5, label=a_names[1])
    ax.plot(a3_x, a3_y, ':', color='#4A90D9', lw=0.9, alpha=0.5, label=a_names[2])
    ax.scatter(0, 0, s=220, c='#FFD54F', edgecolor='#F57F17', zorder=10, label='Sun')
    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=8)

    (traj_line,) = ax.plot([], [], '-', color='white', lw=2.0, zorder=6)
    (sc_dot,)    = ax.plot([], [], 'o', color='white', ms=8, mec='#FFD54F', zorder=12)
    (earth_dot,) = ax.plot([], [], 'o', color='#4FC3F7', ms=8, zorder=11)
    (fb_dot,)    = ax.plot([], [], 'o', color=fb_color, ms=7, zorder=11) \
                   if flyby_id != '399' else (earth_dot,)
    (a1_dot,)    = ax.plot([], [], 'o', color='#D94A4A', ms=7, zorder=11)
    (a2_dot,)    = ax.plot([], [], 'o', color='#5BBD72', ms=7, zorder=11)
    (a3_dot,)    = ax.plot([], [], 'o', color='#4A90D9', ms=7, zorder=11)
    date_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                       fontsize=10, va='top', ha='left', family='monospace',
                       bbox=dict(facecolor='#1a1a3a', edgecolor='#666',
                                 alpha=0.85, boxstyle='round,pad=0.4'))

    def init():
        traj_line.set_data([], [])
        return traj_line, sc_dot, earth_dot, fb_dot, a1_dot, a2_dot, a3_dot, date_txt

    def update(fi):
        cut = idxs[fi]
        traj_line.set_data(sc_x[:cut + 1] / AU, sc_y[:cut + 1] / AU)
        sc_dot.set_data([sc_x[cut] / AU], [sc_y[cut] / AU])
        et = sc_et[cut]
        for dot, bid in [(earth_dot, '399'), (fb_dot, flyby_id),
                         (a1_dot, a_ids[0]), (a2_dot, a_ids[1]), (a3_dot, a_ids[2])]:
            r, _ = get_state(str(bid), float(et))
            dot.set_data([r[0] / AU], [r[1] / AU])
        date = spiceypy.et2utc(float(et), 'C', 0)[:11].strip()
        elapsed = (et - et0) / YEAR
        date_txt.set_text(f"{date}\n+{elapsed:.2f} yr\ndv={res['delta_v_total']:.2f} km/s")
        return traj_line, sc_dot, earth_dot, fb_dot, a1_dot, a2_dot, a3_dot, date_txt

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=60, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=18), dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asteroid_list = load_kernels(os.path.join(repo_root, "NOTABLE_ASTEROID_BSPs"),
                                 "/Users/rebnoob/Documents/ae105/generic_kernels")

    jobs = [
        ("optimal_asteroid_paths/pkl/results_69ast_ega_v2.pkl",
         "Δv-Min #1: {names}  (dv={dv:.2f} km/s, {arch} GA)",
         "Renders/Asteroid_Plots/top_dvmin_ega_v2_2d.gif"),
        ("optimal_asteroid_paths/pkl/results_science_priority_ega_v2.pkl",
         "Science #1: {names}  (dv={dv:.2f}, sci={sci:.1f}, {arch} GA)",
         "Renders/Asteroid_Plots/top_science_ega_v2_2d.gif"),
    ]

    for pkl, tpl, out in jobs:
        with open(os.path.join(repo_root, pkl), 'rb') as f:
            results = pickle.load(f)
        i, j, k, res = results[0]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        ids   = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        arch  = res.get('architecture', 'direct').upper()
        title = tpl.format(names=' → '.join(names),
                           dv=res['delta_v_total'],
                           sci=res.get('science_sum', 0),
                           arch=arch)
        out_full = os.path.join(repo_root, out)
        os.makedirs(os.path.dirname(out_full), exist_ok=True)
        make_gif(res, names, ids, title, out_full)


if __name__ == '__main__':
    main()
