"""Quick test: Earth GA + Mars GA chain for PARTHENOPE → PSYCHE → THEMIS.

Trajectory: Earth launch → Earth flyby (1–3 yr loop) → Mars flyby → A1 → A2 → A3
8-D decision space, 5 Lambert legs, both flybys must satisfy:
  • |v_inf_in| ≈ |v_inf_out|        (ballistic)
  • turn ≤ natural max at safe r_p (geometric)

Earth EGA additional constraints:
  • |v_inf_at_earth| ≥ 3 km/s (else degenerate co-orbital)
  • powered Δv at Earth periapsis ≤ 0.05 km/s (also enforced by ballistic check)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import spiceypy
from scipy.optimize import differential_evolution

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))

from core import (load_kernels, get_id_from_asteroid_name, get_state, get_mu,
                   get_radius, solve_lambert, compute_flyby_dv,
                   audit_flyby_geometry,
                   MU_SUN, DAY, WEEK, MONTH, YEAR, MAX_MISSION_DURATION)
from optimization import FLYBY_BODIES


def compute_path_with_two_flybys(a1_id, a2_id, a3_id,
                                   et_launch, et_fb1, et_fb2,
                                   et_arr1, et_stay1, et_arr2, et_stay2, et_arr3,
                                   fb1_name, fb2_name, m_revs):
    """5-leg Lambert solve with two flybys, both ballistic-only.

    m_revs : tuple of 5 ints — Lambert revolutions per leg (E→FB1, FB1→FB2,
             FB2→A1, A1→A2, A2→A3).
    """
    fb1 = FLYBY_BODIES[fb1_name]; fb2 = FLYBY_BODIES[fb2_name]
    fb1_id, fb2_id = fb1['id'], fb2['id']

    earth_r,  earth_v  = get_state('399',  et_launch)
    fb1_r,    fb1_v    = get_state(fb1_id, et_fb1)
    fb2_r,    fb2_v    = get_state(fb2_id, et_fb2)
    a1_arr_r, a1_arr_v = get_state(a1_id,  et_arr1)
    a1_lv_r,  a1_lv_v  = get_state(a1_id,  et_stay1)
    a2_arr_r, a2_arr_v = get_state(a2_id,  et_arr2)
    a2_lv_r,  a2_lv_v  = get_state(a2_id,  et_stay2)
    a3_arr_r, a3_arr_v = get_state(a3_id,  et_arr3)

    m0, m1, m2, m3, m4 = m_revs

    # Leg 1: Earth → first flyby. For Earth GA, find the multi-rev branch with
    # |v_inf| >= 3 km/s (avoid co-orbital phasing).
    if fb1_name == 'earth':
        MIN_EGA_VINF = 3.0
        best_launch = np.inf; e_lv = fb1_arr = None; ef0 = -1
        for m_try in (0, 1, -1, 2, -2):
            V1, V2, ef = solve_lambert(earth_r, fb1_r,
                                         (et_fb1 - et_launch)/DAY, m_try, MU_SUN)
            if ef != 1: continue
            ldv = np.linalg.norm(V1 - earth_v)
            if ldv < MIN_EGA_VINF: continue
            if ldv < best_launch:
                best_launch = ldv; e_lv, fb1_arr = V1, V2; ef0 = 1
    else:
        e_lv, fb1_arr, ef0 = solve_lambert(earth_r, fb1_r,
                                             (et_fb1 - et_launch)/DAY, m0, MU_SUN)

    # Leg 2: FB1 → FB2
    fb1_dep, fb2_arr, ef1 = solve_lambert(fb1_r, fb2_r,
                                            (et_fb2 - et_fb1)/DAY, m1, MU_SUN)
    # Leg 3: FB2 → A1
    fb2_dep, a1_arr, ef2 = solve_lambert(fb2_r, a1_arr_r,
                                           (et_arr1 - et_fb2)/DAY, m2, MU_SUN)
    # Leg 4: A1 → A2
    a1_lv, a2_arr, ef3 = solve_lambert(a1_lv_r, a2_arr_r,
                                         (et_arr2 - et_stay1)/DAY, m3, MU_SUN)
    # Leg 5: A2 → A3
    a2_lv, a3_arr, ef4 = solve_lambert(a2_lv_r, a3_arr_r,
                                         (et_arr3 - et_stay2)/DAY, m4, MU_SUN)

    if ef0 != 1 or ef1 != 1 or ef2 != 1 or ef3 != 1 or ef4 != 1:
        return {'delta_v_total': 1e3, 'reason': 'lambert_fail'}

    # --- Both flybys: count their actual powered Δv (vis-viva at safe r_p)  ---
    # Geometric feasibility: required turn must fit at safe periapsis.
    # If geometric fails, hard-reject. Otherwise compute the powered burn the
    # spacecraft would need to bridge any |v_inf| mismatch — we ADD this to
    # total Δv and let the optimizer minimize the sum (so it naturally
    # chooses near-ballistic geometries).
    mu_fb1 = get_mu(fb1['mu_body']); R_fb1 = get_radius(fb1['radii_body'])
    safe1  = R_fb1 + fb1['min_alt']
    mu_fb2 = get_mu(fb2['mu_body']); R_fb2 = get_radius(fb2['radii_body'])
    safe2  = R_fb2 + fb2['min_alt']

    def _flyby_cost(v_in_helio, v_out_helio, v_body, mu, safe_r):
        """Returns (powered_dv_kms, geometric_feasible_bool)."""
        v_inf_in  = v_in_helio  - v_body
        v_inf_out = v_out_helio - v_body
        a, b = np.linalg.norm(v_inf_in), np.linalg.norm(v_inf_out)
        cosd = np.clip(np.dot(v_inf_in, v_inf_out) / (a * b), -1, 1)
        delta = np.arccos(cosd)
        # geometric max turn at safe periapsis
        sin_a = min(1.0, 1.0 / (1.0 + safe_r * a**2 / mu))
        sin_b = min(1.0, 1.0 / (1.0 + safe_r * b**2 / mu))
        delta_max = np.arcsin(sin_a) + np.arcsin(sin_b)
        if delta > delta_max + 1e-6:
            return 1e3, False
        # powered Δv at safe-r periapsis (vis-viva speed change)
        v_p_in  = (a**2 + 2*mu/safe_r) ** 0.5
        v_p_out = (b**2 + 2*mu/safe_r) ** 0.5
        return abs(v_p_out - v_p_in), True

    dv_fb1, ok1 = _flyby_cost(fb1_arr, fb1_dep, fb1_v, mu_fb1, safe1)
    dv_fb2, ok2 = _flyby_cost(fb2_arr, fb2_dep, fb2_v, mu_fb2, safe2)
    if not ok1 or not ok2:
        return {'delta_v_total': 1e3, 'reason': 'geometric_infeasible'}

    # Per-burn dv components (heliocentric vector mismatches)
    dv_launch = e_lv - earth_v
    dv_A1_arr = a1_arr - a1_arr_v
    dv_A1_lv  = a1_lv  - a1_lv_v
    dv_A2_arr = a2_arr - a2_arr_v
    dv_A2_lv  = a2_lv  - a2_lv_v
    dv_A3_arr = a3_arr - a3_arr_v

    dv_total = (np.linalg.norm(dv_launch) + abs(dv_fb1) + abs(dv_fb2)
                + np.linalg.norm(dv_A1_arr) + np.linalg.norm(dv_A1_lv)
                + np.linalg.norm(dv_A2_arr) + np.linalg.norm(dv_A2_lv)
                + np.linalg.norm(dv_A3_arr))

    return {
        'delta_v_total': float(dv_total),
        'delta_v_launch':    dv_launch,
        'delta_v_fb1':       float(dv_fb1),
        'delta_v_fb2':       float(dv_fb2),
        'delta_v_A1_arrive': dv_A1_arr,
        'delta_v_A1_leave':  dv_A1_lv,
        'delta_v_A2_arrive': dv_A2_arr,
        'delta_v_A2_leave':  dv_A2_lv,
        'delta_v_A3_arrive': dv_A3_arr,
        'flyby1': fb1_name, 'flyby2': fb2_name,
    }


def _unpack(input_vec, launch_range, fb1_name='earth', fb2_name='mars'):
    """Unpack 8-element vector (in years) → 8 absolute SPICE epochs."""
    s = YEAR * input_vec
    et_launch = s[0] + launch_range[0]
    et_fb1    = et_launch + s[1]
    et_fb2    = et_fb1    + s[2]
    et_arr1   = et_fb2    + s[3]
    et_stay1  = et_arr1   + s[4]
    et_arr2   = et_stay1  + s[5]
    et_stay2  = et_arr2   + s[6]
    et_arr3   = et_stay2  + s[7]
    return (et_launch, et_fb1, et_fb2, et_arr1, et_stay1,
             et_arr2, et_stay2, et_arr3)


def score(input_vec, a1_id, a2_id, a3_id, launch_range, m_revs,
           fb1_name='earth', fb2_name='mars'):
    ets = _unpack(input_vec, launch_range, fb1_name, fb2_name)
    if ets[7] - ets[0] > MAX_MISSION_DURATION:
        return 1e3
    out = compute_path_with_two_flybys(a1_id, a2_id, a3_id, *ets,
                                         fb1_name, fb2_name, m_revs)
    return out['delta_v_total']


def main():
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n)))
              for n in ('PARTHENOPE','PSYCHE','THEMIS')]
    launch_range = [spiceypy.str2et('Jan 1 12:00:00 UTC 2027'),
                    spiceypy.str2et('Dec 31 12:00:00 UTC 2035')]

    fb1_name = 'earth'; fb2_name = 'mars'
    fb1 = FLYBY_BODIES[fb1_name]; fb2 = FLYBY_BODIES[fb2_name]
    bounds = list(zip(
        np.array([0,
                   fb1['tof_min'], fb2['tof_min'],
                   2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0],
                   fb1['tof_max'], fb2['tof_max'],
                   5*YEAR, YEAR, 5*YEAR, YEAR, 5*YEAR]) / YEAR))

    print(f'PARTHENOPE → PSYCHE → THEMIS  via {fb1_name.upper()} GA + {fb2_name.upper()} GA')
    print(f'  bounds (years):')
    for i, (lo, hi) in enumerate(bounds):
        print(f'    x[{i}] : [{lo:6.2f}, {hi:6.2f}]')

    # Modest m-revs sweep (5 legs × 2 = 32 combos is too many; restrict)
    # We only allow 0 or 1 rev on legs 1, 2, 3. Legs 4, 5 stay at 0.
    m_combos = [(m0, m1, m2, 0, 0)
                for m0 in (0, 1, -1)
                for m1 in (0, 1)
                for m2 in (0, 1)]
    seeds = (42, 137, 314, 808)

    print(f'  m-revs combos: {len(m_combos)}, seeds: {len(seeds)}')
    print(f'  total DE runs: {len(m_combos) * len(seeds)}')
    print()
    print(f'{"m_revs":18s}  {"seed":>4s}  {"dv":>7s}  {"feas":>5s}  t(s)')
    print('-' * 50)

    t_start = time.time()
    best = None
    for m_revs in m_combos:
        for seed in seeds:
            t0 = time.time()
            try:
                res = differential_evolution(
                    lambda x: score(x, *a_ids, launch_range, m_revs,
                                     fb1_name, fb2_name),
                    bounds, maxiter=400, tol=1e-7, seed=seed, polish=True,
                    popsize=22, mutation=(0.5, 1.3), recombination=0.8,
                    updating='deferred')
            except Exception as e:
                continue
            ets = _unpack(res.x, launch_range, fb1_name, fb2_name)
            full = compute_path_with_two_flybys(*a_ids, *ets,
                                                  fb1_name, fb2_name, m_revs)
            dv = full['delta_v_total']
            feas = 'OK' if dv < 100 else 'FAIL'
            if dv < 100 and (best is None or dv < best['dv']):
                best = {'dv': dv, 'm_revs': m_revs, 'seed': seed,
                         'ets': ets, 'full': full}
                marker = ' ←★'
            else:
                marker = ''
            print(f'{str(m_revs):18s}  {seed:4d}  {dv:7.3f}  {feas:>5s}  '
                  f'{time.time()-t0:4.1f}{marker}')

    print()
    print(f'Total wall time: {time.time()-t_start:.0f}s')

    if best is None:
        print('No feasible Earth+Mars chain found.')
        return

    print(f'\n{"="*78}')
    print(f' BEST EARTH GA + MARS GA CHAIN')
    print(f'{"="*78}')
    full = best['full']; ets = best['ets']
    print(f'  Total Δv : {best["dv"]:.3f} km/s')
    print(f'  m_revs   : {best["m_revs"]}, seed: {best["seed"]}')
    print(f'  Mission  : {(ets[7]-ets[0])/YEAR:.2f} yr')
    print(f'\n  Timeline:')
    for label, et in [('Launch', ets[0]),
                       (f'{fb1_name.title()} flyby',  ets[1]),
                       (f'{fb2_name.title()} flyby',  ets[2]),
                       ('Arrive PARTHENOPE', ets[3]),
                       ('Depart PARTHENOPE', ets[4]),
                       ('Arrive PSYCHE',     ets[5]),
                       ('Depart PSYCHE',     ets[6]),
                       ('Arrive THEMIS',     ets[7])]:
        print(f'    {label:25s}: {spiceypy.et2utc(et, "C", 0)}')

    # Audit both flybys (use the existing audit function which expects only
    # one flyby; we'll do it manually for both)
    print(f'\n  Earth GA diagnostics:')
    earth_r,_   = get_state('399', ets[0])
    fb1_r,fb1_v = get_state('399', ets[1])
    fb2_r,fb2_v = get_state('4',    ets[2])

    # Recompute v_inf at Earth GA (use the same Lambert solver as inside)
    # Leg 1: Earth → Earth_GA — pick best m branch matching |v_inf|>=3
    best_l1 = None; m_revs = best['m_revs']
    for m_try in (0, 1, -1, 2, -2):
        V1, V2, ef = solve_lambert(earth_r, fb1_r, (ets[1]-ets[0])/DAY,
                                     m_try, MU_SUN)
        if ef != 1: continue
        ldv = np.linalg.norm(V1 - get_state('399', ets[0])[1])
        if ldv < 3.0: continue
        if best_l1 is None or ldv < best_l1[0]: best_l1 = (ldv, V1, V2)
    if best_l1:
        _, V1_em, V2_em = best_l1
        v_inf_in_e  = V2_em - fb1_v
        # Leg 2 V1 = departure from Earth GA
        V1_eM, V2_eM, _ = solve_lambert(fb1_r, fb2_r, (ets[2]-ets[1])/DAY,
                                           m_revs[1], MU_SUN)
        v_inf_out_e = V1_eM - fb1_v
        print(f'    v_inf_in/out      : {np.linalg.norm(v_inf_in_e):.3f} / '
              f'{np.linalg.norm(v_inf_out_e):.3f} km/s')
        residual = np.linalg.norm(v_inf_out_e) - np.linalg.norm(v_inf_in_e)
        print(f'    energy residual   : {residual:+.4f} km/s '
              f'({"BALLISTIC" if abs(residual)<0.05 else "powered"})')
        cosd = np.dot(v_inf_in_e, v_inf_out_e) / (np.linalg.norm(v_inf_in_e)*np.linalg.norm(v_inf_out_e))
        turn_e = np.degrees(np.arccos(np.clip(cosd,-1,1)))
        print(f'    turn angle        : {turn_e:.2f}°')

    print(f'\n  Mars GA diagnostics:')
    V1_eM, V2_eM, _ = solve_lambert(fb1_r, fb2_r, (ets[2]-ets[1])/DAY,
                                       m_revs[1], MU_SUN)
    a1_r, _ = get_state(a_ids[0], ets[3])
    V1_Ma, _, _ = solve_lambert(fb2_r, a1_r, (ets[3]-ets[2])/DAY,
                                   m_revs[2], MU_SUN)
    v_inf_in_m  = V2_eM - fb2_v
    v_inf_out_m = V1_Ma - fb2_v
    print(f'    v_inf_in/out      : {np.linalg.norm(v_inf_in_m):.3f} / '
          f'{np.linalg.norm(v_inf_out_m):.3f} km/s')
    residual = np.linalg.norm(v_inf_out_m) - np.linalg.norm(v_inf_in_m)
    print(f'    energy residual   : {residual:+.4f} km/s '
          f'({"BALLISTIC" if abs(residual)<0.05 else "powered"})')
    cosd = np.dot(v_inf_in_m, v_inf_out_m)/(np.linalg.norm(v_inf_in_m)*np.linalg.norm(v_inf_out_m))
    turn_m = np.degrees(np.arccos(np.clip(cosd,-1,1)))
    print(f'    turn angle        : {turn_m:.2f}°')

    print(f'\n  Δv breakdown:')
    print(f'    Launch           : {np.linalg.norm(full["delta_v_launch"]):7.3f} km/s')
    print(f'    Earth GA powered : {full["delta_v_fb1"]:7.3f} km/s')
    print(f'    Mars  GA powered : {full["delta_v_fb2"]:7.3f} km/s')
    print(f'    Arrive PARTHENOPE: {np.linalg.norm(full["delta_v_A1_arrive"]):7.3f}')
    print(f'    Depart PARTHENOPE: {np.linalg.norm(full["delta_v_A1_leave"]):7.3f}')
    print(f'    Arrive PSYCHE    : {np.linalg.norm(full["delta_v_A2_arrive"]):7.3f}')
    print(f'    Depart PSYCHE    : {np.linalg.norm(full["delta_v_A2_leave"]):7.3f}')
    print(f'    Arrive THEMIS    : {np.linalg.norm(full["delta_v_A3_arrive"]):7.3f}')
    print(f'    {"-"*30}  -------')
    print(f'    Total            : {best["dv"]:7.3f} km/s')

    # Save result
    import pickle
    out = {
        'triplet': ['PARTHENOPE','PSYCHE','THEMIS'],
        'flyby1': fb1_name, 'flyby2': fb2_name,
        'best': {**full,
                  'et_launch':  ets[0], 'et_fb1':    ets[1], 'et_fb2': ets[2],
                  'et_arrive_1':ets[3], 'et_stay_1': ets[4],
                  'et_arrive_2':ets[5], 'et_stay_2': ets[6],
                  'et_arrive_3':ets[7],
                  'm_revs': best['m_revs']},
    }
    out_path = 'optimal_asteroid_paths/pkl/parthenope_psyche_themis_earth_mars_chain.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
