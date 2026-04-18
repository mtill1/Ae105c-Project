"""Plot 2D trajectories for the top paths from result pkls.

Reads `architecture` / `flyby_body` from the result dict so it works for
any GA body (Earth, Moon, Mars, or direct).
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_kernels, two_body_sim, solve_lambert, get_state, MU_SUN, DAY, YEAR

AU = 149_597_870.7  # km

FLYBY_META = {
    '399': {'name': 'Earth',  'color': '#4FC3F7'},
    '301': {'name': 'Moon',   'color': '#BDBDBD'},
    '4':   {'name': 'Mars',   'color': '#FF8A65'},
}


def _best_lambert_min_launch(r0, v0, r1, tof_days, max_m=2):
    """Enumerate m=0..max_m, pick the branch with the smallest launch Δv (v-v0)."""
    best = (None, None, -1, np.inf)
    for m in range(max_m + 1):
        V1, V2, ef = solve_lambert(r0, r1, tof_days, m, MU_SUN)
        if ef == 1:
            ldv = np.linalg.norm(V1 - v0)
            if ldv < best[3]:
                best = (V1, V2, 1, ldv)
    return best[0], best[1], best[2]


def reconstruct_flyby_legs(res, a_id_1, a_id_2, a_id_3, flyby_id):
    """Return list of (X_km, label, color) for the 4 legs of a flyby mission."""
    fb_name = FLYBY_META.get(str(flyby_id), {}).get('name', flyby_id)

    r_e, v_e   = get_state('399',    res['et_launch'])
    r_fb, v_fb = get_state(flyby_id, res['et_flyby'])
    r_a1_arr,_ = get_state(a_id_1,   res['et_arrive_1'])
    r_a1_lv,_  = get_state(a_id_1,   res['et_stay_1'])
    r_a2_arr,_ = get_state(a_id_2,   res['et_arrive_2'])
    r_a2_lv,_  = get_state(a_id_2,   res['et_stay_2'])
    r_a3_arr,_ = get_state(a_id_3,   res['et_arrive_3'])

    # Leg 1: Earth launch → flyby body. For EGA, need m>=1; pick best branch.
    tof1 = (res['et_flyby'] - res['et_launch']) / DAY
    if str(flyby_id) == '399':
        v_dep_1, _, _ = _best_lambert_min_launch(r_e, v_e, r_fb, tof1)
    else:
        v_dep_1, _, _ = solve_lambert(r_e, r_fb, tof1, 0, MU_SUN)
    X1, _ = two_body_sim(res['et_flyby'] - res['et_launch'],
                         np.concatenate([r_e, v_dep_1]), MU_SUN)

    # Leg 2: flyby body → A1
    tof2 = (res['et_arrive_1'] - res['et_flyby']) / DAY
    v_dep_2, _, _ = solve_lambert(r_fb, r_a1_arr, tof2, 0, MU_SUN)
    X2, _ = two_body_sim(res['et_arrive_1'] - res['et_flyby'],
                         np.concatenate([r_fb, v_dep_2]), MU_SUN)

    # Leg 3: A1 → A2
    tof3 = (res['et_arrive_2'] - res['et_stay_1']) / DAY
    v_dep_3, _, _ = solve_lambert(r_a1_lv, r_a2_arr, tof3, 0, MU_SUN)
    X3, _ = two_body_sim(res['et_arrive_2'] - res['et_stay_1'],
                         np.concatenate([r_a1_lv, v_dep_3]), MU_SUN)

    # Leg 4: A2 → A3
    tof4 = (res['et_arrive_3'] - res['et_stay_2']) / DAY
    v_dep_4, _, _ = solve_lambert(r_a2_lv, r_a3_arr, tof4, 0, MU_SUN)
    X4, _ = two_body_sim(res['et_arrive_3'] - res['et_stay_2'],
                         np.concatenate([r_a2_lv, v_dep_4]), MU_SUN)

    return [
        (X1, f'Earth → {fb_name}', '#8B0000'),
        (X2, f'{fb_name} → A1',    '#D94A4A'),
        (X3, 'A1 → A2',            '#5BBD72'),
        (X4, 'A2 → A3',            '#4A90D9'),
    ]


def body_orbit(body_id, et0, etf, n=400):
    ts = np.linspace(et0, etf, n)
    return np.array([get_state(str(body_id), float(t))[0] for t in ts])


def plot_one(res, a_names, a_ids, title, out_path):
    flyby_id = str(res.get('flyby_body', '4'))
    fb_name  = FLYBY_META.get(flyby_id, {}).get('name', flyby_id)
    fb_color = FLYBY_META.get(flyby_id, {}).get('color', '#BBBBBB')

    legs = reconstruct_flyby_legs(res, a_ids[0], a_ids[1], a_ids[2], flyby_id)
    et0, etf = res['et_launch'], res['et_arrive_3']

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_facecolor('#0a0a1f')

    # Reference orbits
    earth_orb = body_orbit('399', et0, etf) / AU
    fb_orb    = body_orbit(flyby_id, et0, etf) / AU if flyby_id != '399' else earth_orb
    a1_orb    = body_orbit(a_ids[0], et0, etf) / AU
    a2_orb    = body_orbit(a_ids[1], et0, etf) / AU
    a3_orb    = body_orbit(a_ids[2], et0, etf) / AU

    ax.plot(earth_orb[:, 0], earth_orb[:, 1], ':', color='#4FC3F7', lw=1,
            label='Earth orbit', alpha=0.7)
    if flyby_id != '399':
        ax.plot(fb_orb[:, 0], fb_orb[:, 1], ':', color=fb_color, lw=1,
                label=f'{fb_name} orbit', alpha=0.7)
    ax.plot(a1_orb[:, 0], a1_orb[:, 1], ':', color='#E0E0E0', lw=0.8, alpha=0.5)
    ax.plot(a2_orb[:, 0], a2_orb[:, 1], ':', color='#E0E0E0', lw=0.8, alpha=0.5)
    ax.plot(a3_orb[:, 0], a3_orb[:, 1], ':', color='#E0E0E0', lw=0.8, alpha=0.5)

    for X, label, color in legs:
        ax.plot(X[:, 0] / AU, X[:, 1] / AU, '-', color=color, lw=2.2, label=label)

    ax.scatter(0, 0, s=250, c='#FFD54F', edgecolor='#F57F17', zorder=10, label='Sun')

    def mark(et, body, color, tag):
        r, _ = get_state(str(body), float(et))
        ax.scatter(r[0]/AU, r[1]/AU, s=120, c=color, edgecolor='white', zorder=11)
        ax.annotate(tag, (r[0]/AU, r[1]/AU), xytext=(8, 8),
                    textcoords='offset points', color='white', fontsize=9, fontweight='bold')

    mark(res['et_launch'],   '399',    '#4FC3F7', 'Launch')
    mark(res['et_flyby'],    flyby_id, fb_color, f'{fb_name} GA')
    mark(res['et_arrive_1'], a_ids[0], '#D94A4A', f'A1: {a_names[0]}')
    mark(res['et_arrive_2'], a_ids[1], '#5BBD72', f'A2: {a_names[1]}')
    mark(res['et_arrive_3'], a_ids[2], '#4A90D9', f'A3: {a_names[2]}')

    dur_yr = (res['et_arrive_3'] - res['et_launch']) / YEAR
    launch_utc = spiceypy.et2utc(res['et_launch'], 'C', 0)[:11].strip()
    arr_utc    = spiceypy.et2utc(res['et_arrive_3'], 'C', 0)[:11].strip()
    info = (f"Total Δv: {res['delta_v_total']:.2f} km/s\n"
            f"Launch Δv: {np.linalg.norm(res['delta_v_launch']):.2f} km/s\n"
            f"{fb_name} flyby Δv: {abs(float(res['delta_v_flyby'])):.2f} km/s\n"
            f"Duration: {dur_yr:.2f} yr\n"
            f"Launch: {launch_utc}\n"
            f"End: {arr_utc}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white', fontsize=9,
            va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9, boxstyle='round,pad=0.5'))

    ax.set_xlabel('X (AU)', color='white')
    ax.set_ylabel('Y (AU)', color='white')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#666')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=8)

    fig.patch.set_facecolor('#0a0a1f')
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asteroid_list = load_kernels(os.path.join(repo_root, "NOTABLE_ASTEROID_BSPs"),
                                 "/Users/rebnoob/Documents/ae105/generic_kernels")

    jobs = [
        ("optimal_asteroid_paths/pkl/results_69ast_ega_v2.pkl",
         "Δv-Min Top Path: {names}  (dv={dv:.2f} km/s, {arch} GA)",
         "Renders/Asteroid_Plots/top_dvmin_ega_v2_2d.png"),
        ("optimal_asteroid_paths/pkl/results_science_priority_ega_v2.pkl",
         "Science Top Path: {names}  (dv={dv:.2f}, sci={sci:.1f}, {arch} GA)",
         "Renders/Asteroid_Plots/top_science_ega_v2_2d.png"),
    ]

    for pkl, title_tpl, out in jobs:
        with open(os.path.join(repo_root, pkl), 'rb') as f:
            results = pickle.load(f)
        i, j, k, res = results[0]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        ids   = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        arch  = res.get('architecture', 'direct').upper()

        title = title_tpl.format(
            names=' → '.join(names),
            dv=res['delta_v_total'],
            sci=res.get('science_sum', 0),
            arch=arch,
        )
        out_full = os.path.join(repo_root, out)
        os.makedirs(os.path.dirname(out_full), exist_ok=True)
        plot_one(res, names, ids, title, out_full)


if __name__ == '__main__':
    main()
