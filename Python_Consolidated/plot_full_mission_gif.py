"""Full mission GIF: Earth launch → EGA loop → Earth flyby → A1 → A2 → A3.

Uniform-time compression: 11.24 yr mission → 30 s at 30 fps (900 frames).
Each frame = ~4.6 days of real time. Stays at asteroids appear as brief pauses.

Uses stored delta_v_launch to reproduce the optimizer's exact trajectory
(avoids Lambert branch ambiguity that would otherwise show a different path).
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_kernels, solve_lambert, two_body_sim, get_state,
                  MU_SUN, DAY, YEAR)

AU = 149_597_870.7


def build_full_mission(res, a_ids, n_per_leg=400):
    """Build (xy, et, segment_tag) for every leg + stay of the mission.

    segment_tag is a label for coloring: 'loop', 'leg2', 'leg3', 'leg4',
    'stay1', 'stay2'.
    """
    et_l   = res['et_launch']
    et_fb  = res['et_flyby']
    et_a1a = res['et_arrive_1']
    et_a1d = res['et_stay_1']
    et_a2a = res['et_arrive_2']
    et_a2d = res['et_stay_2']
    et_a3a = res['et_arrive_3']

    r_e_l, v_e_l = get_state('399', et_l)
    r_e_f, _     = get_state('399', et_fb)
    r_a1a, _     = get_state(a_ids[0], et_a1a)
    r_a1d, _     = get_state(a_ids[0], et_a1d)
    r_a2a, _     = get_state(a_ids[1], et_a2a)
    r_a2d, _     = get_state(a_ids[1], et_a2d)
    r_a3a, _     = get_state(a_ids[2], et_a3a)

    xs, ys, ets, tags = [], [], [], []

    def add_leg(r0, v0, dt, n, tag):
        X, T = two_body_sim(dt, np.concatenate([r0, v0]), MU_SUN, n_steps=n)
        for pt, t in zip(X, T):
            xs.append(pt[0]); ys.append(pt[1])
            ets.append(float(et_here + t)); tags.append(tag)
        return X[-1, 3:6]   # return final velocity (for reference)

    # Leg 1: Earth launch → Earth flyby  (uses stored delta_v_launch)
    et_here = et_l
    v_dep = v_e_l + np.asarray(res['delta_v_launch'])
    add_leg(r_e_l, v_dep, et_fb - et_l, n_per_leg, 'loop')

    # Leg 2: Earth (flyby) → A1.  Standard Lambert m=0.
    et_here = et_fb
    v_dep2, _, _ = solve_lambert(r_e_f, r_a1a, (et_a1a - et_fb) / DAY, 0, MU_SUN)
    add_leg(r_e_f, v_dep2, et_a1a - et_fb, n_per_leg, 'leg2')

    # Stay at A1 — asteroid position over stay duration
    n_stay = max(int(n_per_leg * (et_a1d - et_a1a) / (et_a3a - et_l)), 8)
    for tt in np.linspace(et_a1a, et_a1d, n_stay):
        r, _ = get_state(a_ids[0], float(tt))
        xs.append(r[0]); ys.append(r[1]); ets.append(float(tt)); tags.append('stay1')

    # Leg 3: A1 → A2
    et_here = et_a1d
    v_dep3, _, _ = solve_lambert(r_a1d, r_a2a, (et_a2a - et_a1d) / DAY, 0, MU_SUN)
    add_leg(r_a1d, v_dep3, et_a2a - et_a1d, n_per_leg, 'leg3')

    # Stay at A2
    n_stay = max(int(n_per_leg * (et_a2d - et_a2a) / (et_a3a - et_l)), 8)
    for tt in np.linspace(et_a2a, et_a2d, n_stay):
        r, _ = get_state(a_ids[1], float(tt))
        xs.append(r[0]); ys.append(r[1]); ets.append(float(tt)); tags.append('stay2')

    # Leg 4: A2 → A3
    et_here = et_a2d
    v_dep4, _, _ = solve_lambert(r_a2d, r_a3a, (et_a3a - et_a2d) / DAY, 0, MU_SUN)
    add_leg(r_a2d, v_dep4, et_a3a - et_a2d, n_per_leg, 'leg4')

    return (np.array(xs), np.array(ys), np.array(ets), np.array(tags))


def resample_uniform_time(xs, ys, ets, tags, n):
    """Resample the trajectory so that frame i corresponds to time
    linearly spaced between et_start and et_end. Each frame = equal real time."""
    t_target = np.linspace(ets[0], ets[-1], n)
    x_out = np.interp(t_target, ets, xs)
    y_out = np.interp(t_target, ets, ys)
    # Nearest-neighbor for tags
    idxs = np.searchsorted(ets, t_target)
    idxs = np.clip(idxs, 0, len(tags) - 1)
    tag_out = tags[idxs]
    return x_out, y_out, t_target, tag_out


SEG_COLORS = {
    'loop':  '#FF8A65',   # Earth GA loop — orange
    'leg2':  '#E57373',   # GA → A1
    'stay1': '#FFFFFF',   # at A1
    'leg3':  '#5BBD72',   # A1 → A2
    'stay2': '#FFFFFF',   # at A2
    'leg4':  '#4A90D9',   # A2 → A3
}


def body_orbit_xy(body_id, et0, etf, n=500):
    ts = np.linspace(et0, etf, n)
    pts = np.array([get_state(str(body_id), float(t))[0] for t in ts])
    return pts[:, 0] / AU, pts[:, 1] / AU


def make_gif(res, names, ids, out_path, fps=30, duration_s=30):
    xs, ys, ets, tags = build_full_mission(res, ids)
    n_frames = fps * duration_s
    x_u, y_u, et_u, tag_u = resample_uniform_time(xs, ys, ets, tags, n_frames)

    et0, etf = res['et_launch'], res['et_arrive_3']

    # Reference orbits
    e_x,  e_y  = body_orbit_xy('399',    et0, etf)
    a1_x, a1_y = body_orbit_xy(ids[0],   et0, etf)
    a2_x, a2_y = body_orbit_xy(ids[1],   et0, etf)
    a3_x, a3_y = body_orbit_xy(ids[2],   et0, etf)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    # Static reference orbits
    ax.plot(e_x,  e_y,  ':', color='#4FC3F7', lw=1.2, alpha=0.7, label='Earth')
    ax.plot(a1_x, a1_y, ':', color='#E57373', lw=1.0, alpha=0.5, label=names[0])
    ax.plot(a2_x, a2_y, ':', color='#5BBD72', lw=1.0, alpha=0.5, label=names[1])
    ax.plot(a3_x, a3_y, ':', color='#4A90D9', lw=1.0, alpha=0.5, label=names[2])
    ax.scatter(0, 0, s=260, c='#FFD54F', edgecolor='#F57F17', zorder=10, label='Sun')

    all_x = np.concatenate([e_x, a1_x, a2_x, a3_x, xs / AU])
    all_y = np.concatenate([e_y, a1_y, a2_y, a3_y, ys / AU])
    lim = 1.1 * max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_xlabel('X (AU)', color='white')
    ax.set_ylabel('Y (AU)', color='white')
    ax.set_title(f"Mission: {' → '.join(names)}  (Earth GA)",
                 color='white', fontsize=13, fontweight='bold')

    # Info box (static)
    ldv  = np.linalg.norm(res['delta_v_launch'])
    fbdv = abs(float(res['delta_v_flyby']))
    dur  = (etf - et0) / YEAR
    info = (f"Launch v∞:  {ldv:.2f} km/s\n"
            f"Flyby Δv:   {fbdv:.2f} km/s   (Earth gravity)\n"
            f"Total Δv:   {res['delta_v_total']:.2f} km/s\n"
            f"Duration:   {dur:.2f} yr\n"
            f"30 s video → each sec ≈ {dur/duration_s:.2f} yr")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white',
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9,
                      boxstyle='round,pad=0.5'))

    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=9)

    # Dynamic artists — per-leg colored trail segments + body dots
    trails = {k: ax.plot([], [], '-', color=c, lw=2.2, zorder=7)[0]
              for k, c in SEG_COLORS.items() if k not in ('stay1', 'stay2')}

    (sc_dot,)   = ax.plot([], [], 'o', color='white', ms=9,
                          markeredgecolor='#FFD54F', markeredgewidth=1.5, zorder=12)
    (earth_dot,)= ax.plot([], [], 'o', color='#4FC3F7', ms=8, zorder=11)
    (a1_dot,)   = ax.plot([], [], 'o', color='#E57373', ms=7, zorder=11)
    (a2_dot,)   = ax.plot([], [], 'o', color='#5BBD72', ms=7, zorder=11)
    (a3_dot,)   = ax.plot([], [], 'o', color='#4A90D9', ms=7, zorder=11)

    clock = ax.text(0.98, 0.02, '', transform=ax.transAxes, color='white',
                    fontsize=10, ha='right', va='bottom', family='monospace',
                    bbox=dict(facecolor='#1a1a3a', edgecolor='#666',
                              alpha=0.85, boxstyle='round,pad=0.3'))

    # Pre-compute per-leg masks for efficient drawing
    leg_masks = {k: (tag_u == k) for k in SEG_COLORS}

    def init():
        for tr in trails.values(): tr.set_data([], [])
        return list(trails.values()) + [sc_dot, earth_dot, a1_dot, a2_dot, a3_dot, clock]

    def update(fi):
        cur_tag = tag_u[fi]
        # Draw each leg's trail up to current frame
        for k in trails:
            mask = leg_masks[k] & (np.arange(len(tag_u)) <= fi)
            trails[k].set_data(x_u[mask] / AU, y_u[mask] / AU)

        sc_dot.set_data([x_u[fi] / AU], [y_u[fi] / AU])
        et = et_u[fi]
        for dot, bid in [(earth_dot, '399'), (a1_dot, ids[0]),
                         (a2_dot, ids[1]), (a3_dot, ids[2])]:
            r, _ = get_state(str(bid), float(et))
            dot.set_data([r[0] / AU], [r[1] / AU])

        date = spiceypy.et2utc(float(et), 'C', 0)[:11].strip()
        elapsed = (et - et0) / YEAR
        phase = {'loop': 'Earth GA loop',
                 'leg2': f'→ {names[0]}',
                 'stay1': f'at {names[0]}',
                 'leg3': f'→ {names[1]}',
                 'stay2': f'at {names[1]}',
                 'leg4': f'→ {names[2]}'}.get(cur_tag, '')
        clock.set_text(f"{date}   +{elapsed:5.2f} yr\n{phase}")

        return list(trails.values()) + [sc_dot, earth_dot, a1_dot, a2_dot, a3_dot, clock]

    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=1000 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ast = load_kernels(os.path.join(repo, "NOTABLE_ASTEROID_BSPs"),
                       "/Users/rebnoob/Documents/ae105/generic_kernels")

    with open(os.path.join(repo, "optimal_asteroid_paths/pkl/demo_real_ega.pkl"), 'rb') as f:
        results = pickle.load(f)
    i, j, k, res = results[0]
    names = [ast[x]['NAME'] for x in (i, j, k)]
    ids   = [str(int(ast[x]['ID'])) for x in (i, j, k)]
    out = os.path.join(repo, "Renders/Asteroid_Plots/EGA_Analysis/full_mission_real_ega.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make_gif(res, names, ids, out, fps=30, duration_s=30)


if __name__ == '__main__':
    main()
