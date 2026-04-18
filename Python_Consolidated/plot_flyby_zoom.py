"""Zoom in on the Earth gravity assist: Earth-centered hyperbolic flyby.

Reconstructs v_in and v_out relative to Earth from the stored trajectory,
builds the unpowered hyperbolic trajectory analytically (2-body), and renders
it with Earth to scale, the incoming/outgoing asymptotes, and the powered
burn vector (if any) at periapsis.
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_kernels, solve_lambert, get_state,
                  get_mu, get_radius, MU_SUN, DAY, YEAR)

R_E = 6378.137  # km


def _best_lambert_min_launch(r0, v0, r1, tof_days, max_m=2):
    best = (None, None, np.inf)
    for m in range(-max_m, max_m + 1):
        V1, V2, ef = solve_lambert(r0, r1, tof_days, m, MU_SUN)
        if ef == 1:
            ldv = np.linalg.norm(V1 - v0)
            if ldv < best[2]:
                best = (V1, V2, ldv)
    return best[0], best[1]


def reconstruct_flyby_velocities(res, a_id_1):
    """Return (v_in_rel, v_out_rel, v_earth) at the flyby epoch, in Earth-centered frame."""
    et_l  = res['et_launch']
    et_fb = res['et_flyby']
    et_a1 = res['et_arrive_1']

    r_e_launch, v_e_launch = get_state('399', et_l)
    r_e_fb,     v_e_fb     = get_state('399', et_fb)
    r_a1_arr,   _          = get_state(a_id_1, et_a1)

    # Leg 0: Earth → Earth (EGA loop). Pick branch with min launch dv (matches prod).
    tof0 = (et_fb - et_l) / DAY
    _, v_arr_earth = _best_lambert_min_launch(r_e_launch, v_e_launch, r_e_fb, tof0)
    v_in_rel = v_arr_earth - v_e_fb

    # Leg 1: Earth → A1 (outgoing)
    tof1 = (et_a1 - et_fb) / DAY
    v_dep_earth, _, _ = solve_lambert(r_e_fb, r_a1_arr, tof1, 0, MU_SUN)
    v_out_rel = v_dep_earth - v_e_fb

    return v_in_rel, v_out_rel, v_e_fb


def hyperbolic_flyby_curve(v_in_rel, v_out_rel, mu, r_safe):
    """Return (xy_trajectory, r_p, e, v_inf, turn_angle_deg, needs_burn).

    Coordinate frame: plane of v_in and v_out, origin at Earth, with periapsis
    direction (bisector of -v_in_rel and v_out_rel) on the +x axis.
    """
    v_inf_in  = np.linalg.norm(v_in_rel)
    v_inf_out = np.linalg.norm(v_out_rel)
    v_inf = 0.5 * (v_inf_in + v_inf_out)  # best we can do with an impulsive burn

    u_in  = v_in_rel  / max(v_inf_in, 1e-12)
    u_out = v_out_rel / max(v_inf_out, 1e-12)

    # Turn angle required to rotate v_in_rel into v_out_rel
    cosd = np.clip(np.dot(u_in, u_out), -1.0, 1.0)
    delta = np.arccos(cosd)            # radians, in (0, π)
    # Unpowered-flyby geometry: sin(δ/2) = 1/e  ⇒  r_p = μ/v∞² · (e - 1)
    sin_half = max(np.sin(delta / 2), 1e-6)
    e = 1.0 / sin_half
    r_p = mu / v_inf**2 * (e - 1.0)
    needs_burn = False
    if r_p < r_safe:
        # Actual pass clamped to safe altitude — the remaining velocity mismatch
        # is delivered as a burn at periapsis (what fb_dv reports).
        r_p = r_safe
        e = 1.0 + r_p * v_inf**2 / mu
        needs_burn = True

    a = -mu / v_inf**2  # semi-major axis (negative for hyperbola)

    # Parametrize hyperbola over true anomaly ν, with ν_∞ where r → ∞:
    # cos(ν_∞) = -1/e. Build curve from -ν_end to +ν_end just shy of asymptotes.
    nu_inf = np.arccos(-1.0 / e)
    nu_end = 0.95 * nu_inf             # stop short so the plot has finite extent
    nu = np.linspace(-nu_end, nu_end, 400)
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))   # semilatus rectum form
    x_orbit = r * np.cos(nu)           # periapsis at ν=0 on +x axis
    y_orbit = r * np.sin(nu)

    # Orient so incoming asymptote direction matches u_in projected into 2D.
    # In 2D (x=periapsis bisector, y=perpendicular), asymptote comes in from
    # angle π − ν_∞ (measured from +x axis, rotating toward +y).
    # u_in in helio frame maps to -u_in direction at +∞, i.e., incoming at angle
    # corresponding to ν → -ν_∞ on our curve, which in xy is angle (-ν_∞ - ... )
    # Simpler: build 2D basis from (u_in, u_out) and expand.
    # Construct basis e1 = -(u_in + u_out)/|.| (periapsis), e2 = perpendicular in plane.
    e1 = -(u_in + u_out)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-9:
        e1 = np.cross(u_in, np.array([0, 0, 1])); e1_norm = np.linalg.norm(e1)
    e1 = e1 / e1_norm
    # e2 perpendicular to e1 in plane of u_in,u_out, chosen so that u_out has +y
    plane_n = np.cross(u_in, u_out)
    plane_n = plane_n / max(np.linalg.norm(plane_n), 1e-9)
    e2 = np.cross(plane_n, e1)

    # Promote the 2D orbit curve into 3D using (e1, e2)
    xy3d = np.outer(x_orbit, e1) + np.outer(y_orbit, e2)

    # For the plot, project onto (e1, e2) — we already have that as (x_orbit, y_orbit)
    return (np.column_stack([x_orbit, y_orbit]),
            r_p, e, v_inf, np.degrees(delta), needs_burn,
            (e1, e2))


def plot_flyby(res, names, ids, title, out_path):
    mu_earth = float(get_mu('399'))
    r_safe = float(get_radius('399')) + 300.0  # 300 km min altitude

    v_in_rel, v_out_rel, v_earth = reconstruct_flyby_velocities(res, ids[0])
    curve2d, r_p, e, v_inf, turn_deg, burned, (e1, e2) = \
        hyperbolic_flyby_curve(v_in_rel, v_out_rel, mu_earth, r_safe)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    # Earth (to scale)
    earth = Circle((0, 0), R_E, color='#4FC3F7', alpha=0.9, zorder=5, ec='#0288D1', lw=1.5)
    ax.add_patch(earth)
    ax.text(0, 0, 'Earth', color='white', ha='center', va='center',
            fontsize=10, fontweight='bold', zorder=6)

    # Safe-altitude ring (min-altitude boundary)
    safe = Circle((0, 0), r_safe, fill=False, edgecolor='#F44336',
                  linestyle='--', lw=1.2, alpha=0.7, zorder=4)
    ax.add_patch(safe)

    # Hyperbolic trajectory
    ax.plot(curve2d[:, 0], curve2d[:, 1], '-', color='white', lw=2.5,
            zorder=7, label='Spacecraft hyperbolic flyby')

    # Arrowheads at both ends: incoming (start) and outgoing (end)
    # Incoming: first few points — draw arrow from [5] to [0]
    def arrow(p0, p1, color, label=None):
        ax.annotate('', xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5),
                    zorder=8)
    arrow(curve2d[15], curve2d[0],  '#FF8A65')    # incoming direction
    arrow(curve2d[-15], curve2d[-1],'#5BBD72')    # outgoing direction

    # Periapsis marker (on +x axis)
    ax.scatter([r_p], [0], s=90, c='#FFD54F', edgecolor='white', zorder=9,
               label=f'Periapsis (alt = {r_p - R_E:.0f} km)')

    # Asymptote lines (what a straight-line flyby would look like) — dashed
    # In the (e1, e2) 2D frame, v_in enters at angle π - ν_∞ from +x
    nu_inf = np.arccos(-1.0 / e)
    # Incoming asymptote: from far away, direction = +x rotated by -(π-ν_∞)
    L = 8 * r_p
    in_dir  = np.array([np.cos(np.pi - nu_inf), -np.sin(np.pi - nu_inf)])
    out_dir = np.array([np.cos(np.pi - nu_inf),  np.sin(np.pi - nu_inf)])
    ax.plot([L * in_dir[0], 0], [L * in_dir[1], 0], ':', color='#FF8A65',
            lw=1.2, alpha=0.6)
    ax.plot([0, -L * out_dir[0]], [0, -L * out_dir[1]], ':', color='#5BBD72',
            lw=1.2, alpha=0.6)

    # Labels for in / out
    ax.text(curve2d[0, 0], curve2d[0, 1] - 2500, 'v∞ in', color='#FF8A65',
            fontsize=11, fontweight='bold', ha='center')
    ax.text(curve2d[-1, 0], curve2d[-1, 1] + 2500, 'v∞ out', color='#5BBD72',
            fontsize=11, fontweight='bold', ha='center')

    # Info box
    dur_yr = (res['et_arrive_3'] - res['et_launch']) / YEAR
    info = (f"Incoming v∞: {np.linalg.norm(v_in_rel):.2f} km/s\n"
            f"Outgoing v∞: {np.linalg.norm(v_out_rel):.2f} km/s\n"
            f"Turn angle:  {turn_deg:.1f}°\n"
            f"Periapsis:   {r_p:.0f} km  (alt {r_p - R_E:.0f} km)\n"
            f"Eccentricity:{e:5.2f}\n"
            f"Powered Δv:  {abs(float(res['delta_v_flyby'])):.2f} km/s"
            f"{'  *required*' if burned else ''}\n"
            f"Flyby date:  {spiceypy.et2utc(float(res['et_flyby']), 'C', 0)[:11].strip()}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white',
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9,
                      boxstyle='round,pad=0.5'))

    ax.set_xlabel('x (km)  — Earth-centered ECLIPJ2000', color='white')
    ax.set_ylabel('y (km)  — in flyby plane',           color='white')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='white')

    # Zoom scale: a few Earth radii past periapsis
    lim = max(abs(curve2d[:, 0]).max(), abs(curve2d[:, 1]).max()) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ast = load_kernels(os.path.join(repo, "NOTABLE_ASTEROID_BSPs"),
                       "/Users/rebnoob/Documents/ae105/generic_kernels")

    pkl = os.path.join(repo, "optimal_asteroid_paths/pkl/results_69ast_ega_v2.pkl")
    with open(pkl, 'rb') as f:
        results = pickle.load(f)

    # Find first real EGA
    for rank, (i, j, k, res) in enumerate(results[:15], 1):
        if res.get('architecture') == 'earth':
            names = [ast[x]['NAME'] for x in (i, j, k)]
            ids   = [str(int(ast[x]['ID'])) for x in (i, j, k)]
            title = (f"Earth Gravity Assist — #{rank} {' → '.join(names)}\n"
                     f"dv={res['delta_v_total']:.2f} km/s")
            out = os.path.join(repo, "Renders/Asteroid_Plots/earth_flyby_zoom.png")
            plot_flyby(res, names, ids, title, out)
            break


if __name__ == '__main__':
    main()
