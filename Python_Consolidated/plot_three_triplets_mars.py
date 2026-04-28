"""Plot trajectories for the three Mars-flyby triplets in three_triplets_mars_only.pkl.

Produces one PNG per triplet (top-down view + 3D side panel) and a combined PNG.
"""

import os
import pickle

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core import (load_kernels, get_state, two_body_sim, solve_lambert,
                  DAY, MU_SUN)


PKL_PATH = 'optimal_asteroid_paths/pkl/three_triplets_mars_only.pkl'
OUT_DIR = 'Renders/Asteroid_Plots/Best_DeltaV'

LEG_COLORS = ['#ff5555', '#ffaa00', '#55ff55', '#5599ff']  # E->M, M->A1, A1->A2, A2->A3
LEG_LABELS = ['Earth -> Mars', 'Mars -> A1', 'A1 -> A2', 'A2 -> A3']
ORBIT_COLORS = {'earth': '#00d0ff', 'mars': '#ff7733'}
AST_COLORS = ['#ff3366', '#33ff66', '#3366ff']


def propagate_lambert_leg(r1, v1_planet, r2, v2_planet, et_dep, et_arr, n_pts=200):
    """Re-solve Lambert and propagate the transfer arc for plotting.

    Returns (X_traj, V1_sc, V2_sc) where X_traj is (N,6) state along arc.
    """
    tof = (et_arr - et_dep) / DAY
    V1, V2, ef = solve_lambert(r1, r2, tof, 0, MU_SUN)
    if ef != 1:
        # Fall back to other revolutions
        for m in (1, -1, 2, -2):
            V1, V2, ef = solve_lambert(r1, r2, tof, m, MU_SUN)
            if ef == 1:
                break
    x0 = np.concatenate([r1, V1])
    X, _ = two_body_sim(et_arr - et_dep, x0, MU_SUN, n_steps=n_pts)
    return X, V1, V2


def orbit_samples(body_id, et0, et1, n_pts=300):
    """Sample body position over [et0, et1]."""
    ts = np.linspace(et0, et1, n_pts)
    return np.array([get_state(str(body_id), float(t))[0] for t in ts])


def full_orbit_samples(body_id, et_center, period_days=None, n_pts=400):
    """Sample one full orbit centered on et_center.

    For unknown period, sample 6 years forward (covers main-belt periods).
    """
    if period_days is None:
        period_days = 6 * 365.25
    t_half = period_days * DAY / 2
    ts = np.linspace(et_center - t_half, et_center + t_half, n_pts)
    return np.array([get_state(str(body_id), float(t))[0] for t in ts])


def plot_triplet(triplet, result, asteroid_id_lookup, out_path):
    n1, n2, n3 = triplet
    a_id_1 = str(int(asteroid_id_lookup[n1]))
    a_id_2 = str(int(asteroid_id_lookup[n2]))
    a_id_3 = str(int(asteroid_id_lookup[n3]))

    et_l = result['et_launch']
    et_m = result['et_flyby']
    et_a1 = result['et_arrive_1']
    et_s1 = result['et_stay_1']
    et_a2 = result['et_arrive_2']
    et_s2 = result['et_stay_2']
    et_a3 = result['et_arrive_3']

    # Endpoint states
    r_earth, v_earth = get_state('399', et_l)
    r_mars,  v_mars  = get_state('4',   et_m)
    r_a1_arr, _      = get_state(a_id_1, et_a1)
    r_a1_lv,  _      = get_state(a_id_1, et_s1)
    r_a2_arr, _      = get_state(a_id_2, et_a2)
    r_a2_lv,  _      = get_state(a_id_2, et_s2)
    r_a3_arr, _      = get_state(a_id_3, et_a3)

    # Propagate each leg
    leg_E_M, _, _    = propagate_lambert_leg(r_earth, v_earth, r_mars, v_mars, et_l,  et_m)
    leg_M_A1, _, _   = propagate_lambert_leg(r_mars, v_mars, r_a1_arr, None, et_m,  et_a1)
    leg_A1_A2, _, _  = propagate_lambert_leg(r_a1_lv, None, r_a2_arr, None, et_s1, et_a2)
    leg_A2_A3, _, _  = propagate_lambert_leg(r_a2_lv, None, r_a3_arr, None, et_s2, et_a3)

    # Body orbits over the mission window (so arcs sit on the visible orbit)
    earth_orbit = orbit_samples('399', et_l - 0.5*365.25*DAY, et_a3 + 0.5*365.25*DAY, 400)
    mars_orbit  = orbit_samples('4',   et_l - 0.5*365.25*DAY, et_a3 + 0.5*365.25*DAY, 400)
    a1_orbit    = orbit_samples(a_id_1, et_l, et_a3, 400)
    a2_orbit    = orbit_samples(a_id_2, et_l, et_a3, 400)
    a3_orbit    = orbit_samples(a_id_3, et_l, et_a3, 400)

    legs = [leg_E_M, leg_M_A1, leg_A1_A2, leg_A2_A3]

    # ----- Figure: 2 panels (top-down + 3D) -----
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a14')

    # ---------- Panel 1: top-down (XY plane) ----------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#0a0a14')
    ax1.set_aspect('equal')

    # Sun
    ax1.scatter(0, 0, color='yellow', s=200, marker='*', zorder=5, label='Sun')

    # Orbits (XY)
    ax1.plot(earth_orbit[:, 0], earth_orbit[:, 1], color=ORBIT_COLORS['earth'],
             lw=0.8, alpha=0.7, label='Earth orbit')
    ax1.plot(mars_orbit[:, 0], mars_orbit[:, 1], color=ORBIT_COLORS['mars'],
             lw=0.8, alpha=0.7, label='Mars orbit')
    ax1.plot(a1_orbit[:, 0], a1_orbit[:, 1], color=AST_COLORS[0], lw=0.8, alpha=0.6, label=f'{n1} orbit')
    ax1.plot(a2_orbit[:, 0], a2_orbit[:, 1], color=AST_COLORS[1], lw=0.8, alpha=0.6, label=f'{n2} orbit')
    ax1.plot(a3_orbit[:, 0], a3_orbit[:, 1], color=AST_COLORS[2], lw=0.8, alpha=0.6, label=f'{n3} orbit')

    # Trajectory legs
    for leg, c, lab in zip(legs, LEG_COLORS, LEG_LABELS):
        ax1.plot(leg[:, 0], leg[:, 1], color=c, lw=2.0, label=lab, zorder=4)

    # Waypoints
    waypoints = [
        (r_earth, 'Launch',  ORBIT_COLORS['earth']),
        (r_mars,  'Mars GA', ORBIT_COLORS['mars']),
        (r_a1_arr, n1,        AST_COLORS[0]),
        (r_a2_arr, n2,        AST_COLORS[1]),
        (r_a3_arr, n3,        AST_COLORS[2]),
    ]
    for pos, name, col in waypoints:
        ax1.scatter(pos[0], pos[1], color=col, s=80, edgecolor='white', lw=1.5, zorder=6)
        ax1.annotate(name, (pos[0], pos[1]),
                     textcoords='offset points', xytext=(8, 8),
                     color='white', fontsize=9)

    ax1.set_xlabel('X [km]', color='white')
    ax1.set_ylabel('Y [km]', color='white')
    ax1.tick_params(colors='white')
    for s in ax1.spines.values():
        s.set_color('white')
    ax1.grid(True, color='gray', alpha=0.3, linestyle=':')
    ax1.set_title('Top-down (ECLIPJ2000 X-Y)', color='white')

    # ---------- Panel 2: 3D ----------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('#0a0a14')

    ax2.scatter([0], [0], [0], color='yellow', s=200, marker='*')
    ax2.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
             color=ORBIT_COLORS['earth'], lw=0.8, alpha=0.7)
    ax2.plot(mars_orbit[:, 0], mars_orbit[:, 1], mars_orbit[:, 2],
             color=ORBIT_COLORS['mars'], lw=0.8, alpha=0.7)
    ax2.plot(a1_orbit[:, 0], a1_orbit[:, 1], a1_orbit[:, 2], color=AST_COLORS[0], lw=0.8, alpha=0.6)
    ax2.plot(a2_orbit[:, 0], a2_orbit[:, 1], a2_orbit[:, 2], color=AST_COLORS[1], lw=0.8, alpha=0.6)
    ax2.plot(a3_orbit[:, 0], a3_orbit[:, 1], a3_orbit[:, 2], color=AST_COLORS[2], lw=0.8, alpha=0.6)

    for leg, c in zip(legs, LEG_COLORS):
        ax2.plot(leg[:, 0], leg[:, 1], leg[:, 2], color=c, lw=2.0)

    for pos, _, col in waypoints:
        ax2.scatter([pos[0]], [pos[1]], [pos[2]], color=col, s=60, edgecolor='white', lw=1.0)

    ax2.set_xlabel('X [km]', color='white')
    ax2.set_ylabel('Y [km]', color='white')
    ax2.set_zlabel('Z [km]', color='white')
    ax2.tick_params(colors='white')
    ax2.set_title('3D View', color='white')
    ax2.view_init(elev=25, azim=-55)
    # Hide grid panes
    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis.pane.set_edgecolor('gray')

    # Title with summary
    dv = result['delta_v_total']
    dur_yr = (et_a3 - et_l) / (365.25 * DAY)
    fig.suptitle(
        f'{n1} -> {n2} -> {n3}   (Mars flyby)\n'
        f'$\\Delta V_{{tot}}$ = {dv:.2f} km/s    duration = {dur_yr:.1f} yr    '
        f'launch = {spiceypy.et2utc(et_l, "C", 0)}',
        color='white', fontsize=13)

    # Legend on first axis
    ax1.legend(loc='upper right', fontsize=8, facecolor='#1a1a2a',
               edgecolor='gray', labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  saved -> {out_path}')


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs',
                                 '/Users/rebnoob/Documents/ae105/generic_kernels')
    id_lookup = {a['NAME']: a['ID'] for a in asteroid_list}

    with open(PKL_PATH, 'rb') as f:
        results = pickle.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for triplet, res in results:
        if res is None:
            print(f'skip {triplet} (no result)')
            continue
        slug = '_'.join(t.lower() for t in triplet)
        out_path = os.path.join(OUT_DIR, f'mars_flyby_{slug}.png')
        print(f'plotting {" -> ".join(triplet)} ...')
        plot_triplet(triplet, res, id_lookup, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
