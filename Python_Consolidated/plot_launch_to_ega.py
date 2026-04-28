"""Animated heliocentric view: launch → Earth flyby (leg 1 only).

30 seconds at 30 fps = 900 frames. Constant on-screen spacecraft speed
via uniform arc-length resampling.
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


def build_leg1_trajectory(res, n_dense=4000):
    """Build the heliocentric Earth → Earth trajectory for leg 1.

    Uses the stored delta_v_launch vector so we reproduce the EXACT trajectory
    the optimizer selected (re-solving Lambert here would pick a different
    branch under the min-launch-dv criterion, since that criterion alone can
    grab degenerate near-zero-v∞ phasing orbits).
    """
    et_l, et_fb = res['et_launch'], res['et_flyby']
    r_e_l, v_e_l = get_state('399', et_l)
    v_dep = v_e_l + np.asarray(res['delta_v_launch'])  # exact optimizer choice
    x0 = np.concatenate([r_e_l, v_dep])
    # two_body_sim uses pykep's propagate_lagrangian internally (see core.py)
    X, T = two_body_sim(et_fb - et_l, x0, MU_SUN, n_steps=n_dense)
    et_array = et_l + T
    return X[:, :3], et_array, v_dep, v_e_l


def resample_uniform_arclength_with_time(xy_ts, n):
    """Uniform-arc-length resample. Returns (n,2) positions and (n,) matched times."""
    xy = xy_ts[0]
    ts = xy_ts[1]
    d = np.sqrt(np.diff(xy[:, 0])**2 + np.diff(xy[:, 1])**2)
    s = np.concatenate([[0], np.cumsum(d)])
    s_t = np.linspace(0, s[-1], n)
    x = np.interp(s_t, s, xy[:, 0])
    y = np.interp(s_t, s, xy[:, 1])
    t = np.interp(s_t, s, ts)
    return np.column_stack([x, y]), t


def make_video(res, names, ids, out_path, fps=30, duration_s=30):
    X3d, et_array, v_dep, v_e_l = build_leg1_trajectory(res)
    xy = X3d[:, :2] / AU

    n_frames = fps * duration_s
    xy_u, et_u = resample_uniform_arclength_with_time((xy, et_array), n_frames)

    et_l, et_fb = res['et_launch'], res['et_flyby']

    # Earth orbit over the leg-1 window
    ts_orb = np.linspace(et_l, et_fb, 400)
    earth_orb = np.array([get_state('399', float(t))[0] for t in ts_orb]) / AU

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    # Earth orbit
    ax.plot(earth_orb[:, 0], earth_orb[:, 1], ':', color='#4FC3F7', lw=1.2,
            alpha=0.7, label='Earth orbit')

    # Sun
    ax.scatter(0, 0, s=260, c='#FFD54F', edgecolor='#F57F17', zorder=10, label='Sun')

    # Ghosted full spacecraft path
    ax.plot(xy[:, 0], xy[:, 1], '-', color='white', lw=0.6, alpha=0.3, zorder=5)

    # Launch / flyby markers
    ax.scatter(xy[0, 0], xy[0, 1], s=130, c='#4FC3F7', edgecolor='white', zorder=11)
    ax.annotate('Launch', (xy[0, 0], xy[0, 1]), xytext=(8, 8),
                textcoords='offset points', color='white',
                fontsize=10, fontweight='bold')
    ax.scatter(xy[-1, 0], xy[-1, 1], s=130, c='#FF8A65', edgecolor='white', zorder=11)
    ax.annotate('Earth GA', (xy[-1, 0], xy[-1, 1]), xytext=(8, -14),
                textcoords='offset points', color='white',
                fontsize=10, fontweight='bold')

    lim = 1.15 * max(np.abs(xy).max(), np.abs(earth_orb).max())
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_xlabel('X (AU)', color='white')
    ax.set_ylabel('Y (AU)', color='white')
    ax.set_title(f"Launch → Earth GA — leg 1 of {' → '.join(names)}",
                 color='white', fontsize=13, fontweight='bold')

    # Info box
    launch_utc = spiceypy.et2utc(et_l,  'C', 0)[:11].strip()
    flyby_utc  = spiceypy.et2utc(et_fb, 'C', 0)[:11].strip()
    loop_yr    = (et_fb - et_l) / YEAR
    launch_dv  = np.linalg.norm(v_dep - v_e_l)
    info = (f"Launch:    {launch_utc}\n"
            f"Earth GA:  {flyby_utc}\n"
            f"Loop TOF:  {loop_yr:.2f} yr\n"
            f"Launch v∞: {launch_dv:.2f} km/s\n"
            f"Total dv:  {res['delta_v_total']:.2f} km/s")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white',
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9,
                      boxstyle='round,pad=0.5'))

    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=9)

    # Dynamic artists
    (trail,)   = ax.plot([], [], '-', color='white', lw=2.4, zorder=7)
    (sc_dot,)  = ax.plot([], [], 'o', color='white', ms=9,
                          markeredgecolor='#FFD54F', markeredgewidth=1.5, zorder=12)
    (earth,)   = ax.plot([], [], 'o', color='#4FC3F7', ms=9, zorder=11)
    clock = ax.text(0.98, 0.02, '', transform=ax.transAxes, color='white',
                    fontsize=11, ha='right', va='bottom', family='monospace',
                    bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.85,
                              boxstyle='round,pad=0.3'))

    def init():
        trail.set_data([], [])
        sc_dot.set_data([], [])
        earth.set_data([], [])
        clock.set_text('')
        return trail, sc_dot, earth, clock

    def update(fi):
        trail.set_data(xy_u[:fi + 1, 0], xy_u[:fi + 1, 1])
        sc_dot.set_data([xy_u[fi, 0]], [xy_u[fi, 1]])
        r_e, _ = get_state('399', float(et_u[fi]))
        earth.set_data([r_e[0] / AU], [r_e[1] / AU])
        date = spiceypy.et2utc(float(et_u[fi]), 'C', 0)[:11].strip()
        elapsed_yr = (et_u[fi] - et_l) / YEAR
        clock.set_text(f"{date}   +{elapsed_yr:4.2f} yr   (t = {fi/fps:4.1f}s)")
        return trail, sc_dot, earth, clock

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
    out = os.path.join(repo, "Renders/Asteroid_Plots/launch_to_ega_real.gif")
    make_video(res, names, ids, out, fps=30, duration_s=30)


if __name__ == '__main__':
    main()
