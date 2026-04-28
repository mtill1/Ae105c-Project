"""Animated zoom-in video of the Earth gravity assist.

30 seconds at 30 fps = 900 frames. Constant on-screen speed
(uniform arc-length parametrization of the hyperbola).
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_kernels, solve_lambert, two_body_sim, get_state,
                  get_mu, get_radius, MU_SUN, DAY, YEAR)

R_E = 6378.137


def reconstruct_flyby_velocities(res, a_id_1):
    """v_in and v_out relative to Earth at the flyby epoch.

    Uses the stored delta_v_launch vector to forward-propagate — avoids the
    Lambert branch-selection ambiguity that would otherwise let this script
    show a degenerate near-zero-v∞ trajectory different from the optimizer's.
    """
    et_l, et_fb, et_a1 = res['et_launch'], res['et_flyby'], res['et_arrive_1']
    r_e_l, v_e_l = get_state('399', et_l)
    r_e_f, v_e_f = get_state('399', et_fb)
    r_a1, _      = get_state(a_id_1, et_a1)

    # Forward-propagate the optimizer's exact trajectory to get arrival velocity
    v_dep = v_e_l + np.asarray(res['delta_v_launch'])
    X, _ = two_body_sim(et_fb - et_l, np.concatenate([r_e_l, v_dep]), MU_SUN, n_steps=50)
    v_arr_e = X[-1, 3:6]
    v_in_rel = v_arr_e - v_e_f

    v_dep_e, _, _ = solve_lambert(r_e_f, r_a1, (et_a1 - et_fb) / DAY, 0, MU_SUN)
    v_out_rel = v_dep_e - v_e_f
    return v_in_rel, v_out_rel


def hyperbolic_curve(v_in_rel, v_out_rel, mu, r_safe, n_dense=5000):
    v_inf = 0.5 * (np.linalg.norm(v_in_rel) + np.linalg.norm(v_out_rel))
    u_in  = v_in_rel  / max(np.linalg.norm(v_in_rel),  1e-12)
    u_out = v_out_rel / max(np.linalg.norm(v_out_rel), 1e-12)

    cosd = np.clip(np.dot(u_in, u_out), -1.0, 1.0)
    delta = np.arccos(cosd)
    sin_half = max(np.sin(delta / 2), 1e-6)
    e = 1.0 / sin_half
    r_p = mu / v_inf**2 * (e - 1.0)
    burned = False
    if r_p < r_safe:
        r_p = r_safe
        e = 1.0 + r_p * v_inf**2 / mu
        burned = True

    a = -mu / v_inf**2
    nu_inf = np.arccos(-1.0 / e)
    nu_end = 0.95 * nu_inf
    nu = np.linspace(-nu_end, nu_end, n_dense)
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    return np.column_stack([x, y]), r_p, e, v_inf, np.degrees(delta), burned, nu_inf


def resample_uniform_arclength(xy, n):
    """Return n points along xy that are equally spaced in arc length."""
    d = np.sqrt(np.diff(xy[:, 0])**2 + np.diff(xy[:, 1])**2)
    s = np.concatenate([[0], np.cumsum(d)])
    s_target = np.linspace(0, s[-1], n)
    x = np.interp(s_target, s, xy[:, 0])
    y = np.interp(s_target, s, xy[:, 1])
    return np.column_stack([x, y])


def make_video(res, names, ids, out_path, fps=30, duration_s=30):
    mu_earth = float(get_mu('399'))
    r_safe = float(get_radius('399')) + 300.0

    v_in_rel, v_out_rel = reconstruct_flyby_velocities(res, ids[0])
    curve_dense, r_p, e, v_inf, turn_deg, burned, nu_inf = \
        hyperbolic_curve(v_in_rel, v_out_rel, mu_earth, r_safe)

    n_frames = fps * duration_s                       # 900 for 30s @ 30fps
    curve = resample_uniform_arclength(curve_dense, n_frames)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    # Earth + safety ring
    ax.add_patch(Circle((0, 0), R_E, color='#4FC3F7', alpha=0.9, zorder=5,
                        ec='#0288D1', lw=1.5))
    ax.text(0, 0, 'Earth', color='white', ha='center', va='center',
            fontsize=10, fontweight='bold', zorder=6)
    ax.add_patch(Circle((0, 0), r_safe, fill=False, edgecolor='#F44336',
                        linestyle='--', lw=1.2, alpha=0.7, zorder=4))

    # Full trajectory (ghosted)
    ax.plot(curve_dense[:, 0], curve_dense[:, 1], '-', color='white',
            lw=0.8, alpha=0.25, zorder=6)

    # Asymptotes (static)
    L = 8 * r_p
    in_dir  = np.array([np.cos(np.pi - nu_inf), -np.sin(np.pi - nu_inf)])
    out_dir = np.array([np.cos(np.pi - nu_inf),  np.sin(np.pi - nu_inf)])
    ax.plot([L * in_dir[0], 0], [L * in_dir[1], 0], ':', color='#FF8A65', lw=1.2, alpha=0.6)
    ax.plot([0, -L * out_dir[0]], [0, -L * out_dir[1]], ':', color='#5BBD72', lw=1.2, alpha=0.6)

    # Periapsis marker
    ax.scatter([r_p], [0], s=90, c='#FFD54F', edgecolor='white', zorder=9,
               label=f'Periapsis (alt {r_p - R_E:.0f} km)')

    # Axis limits / labels
    lim = max(np.abs(curve_dense[:, 0]).max(), np.abs(curve_dense[:, 1]).max()) * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)  — Earth-centered',          color='white')
    ax.set_ylabel('y (km)  — in flyby plane',          color='white')
    title = (f"Earth Flyby — {' → '.join(names)}  (dv={res['delta_v_total']:.2f} km/s)")
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.grid(True, alpha=0.2, color='white')

    # Static info box
    info = (f"Incoming v∞:  {np.linalg.norm(v_in_rel):.2f} km/s\n"
            f"Outgoing v∞:  {np.linalg.norm(v_out_rel):.2f} km/s\n"
            f"Turn angle:   {turn_deg:.1f}°\n"
            f"Periapsis:    {r_p:.0f} km (alt {r_p - R_E:.0f} km)\n"
            f"Eccentricity: {e:.2f}\n"
            f"Powered Δv:   {abs(float(res['delta_v_flyby'])):.2f} km/s"
            f"{'  *required*' if burned else ''}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white',
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9,
                      boxstyle='round,pad=0.5'))

    # Dynamic artists
    (trail,) = ax.plot([], [], '-', color='white', lw=2.5, zorder=7)
    (sc,)    = ax.plot([], [], 'o', color='white', ms=9,
                       markeredgecolor='#FFD54F', markeredgewidth=1.5, zorder=10)

    # Progress indicator (time along the arc, visual only)
    pct_txt = ax.text(0.98, 0.02, '', transform=ax.transAxes, color='white',
                      fontsize=11, ha='right', va='bottom', family='monospace',
                      bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.85,
                                boxstyle='round,pad=0.3'))

    def init():
        trail.set_data([], [])
        sc.set_data([], [])
        pct_txt.set_text('')
        return trail, sc, pct_txt

    def update(fi):
        trail.set_data(curve[:fi + 1, 0], curve[:fi + 1, 1])
        sc.set_data([curve[fi, 0]], [curve[fi, 1]])
        pct_txt.set_text(f"t = {fi / fps:5.1f}s")
        return trail, sc, pct_txt

    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=9)
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
    out = os.path.join(repo, "Renders/Asteroid_Plots/earth_flyby_zoom_real.gif")
    make_video(res, names, ids, out, fps=30, duration_s=30)


if __name__ == '__main__':
    main()
