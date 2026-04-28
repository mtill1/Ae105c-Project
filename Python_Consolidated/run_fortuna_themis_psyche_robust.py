"""Robust multi-start Mars-flyby optimization for FORTUNA -> THEMIS -> PSYCHE.

Drives the same compute_path_with_flyby objective as optimize_times_flyby, but
adds:
  - Multiple random seeds (multimodal DE landscape).
  - Wider Lambert revolution (m) sweep.
  - Higher DE iteration budget and population.
  - Direct (no-GA) baseline for comparison.
  - Final L-BFGS-B polish from each best.
"""

import os
import pickle
import time

import numpy as np
import spiceypy
from scipy.optimize import differential_evolution, minimize

from core import load_kernels, get_id_from_asteroid_name, DAY, YEAR, WEEK, MONTH
from optimization import (compute_path_with_flyby, score_paths_flyby,
                          _unpack_flyby_input, FLYBY_BODIES,
                          compute_path_deltav, score_paths)


TRIPLET = ('FORTUNA', 'THEMIS', 'PSYCHE')
LAUNCH_UTC_MIN = 'Jan 1 12:00:00 UTC 2027'
LAUNCH_UTC_MAX = 'Dec 31 12:00:00 UTC 2035'

SEEDS = [13, 42, 91, 137, 314, 808]
# All 16 single-rev combos for the 4 legs:
M_COMBOS = [(m0, m1, m2, m3)
            for m0 in (0, 1)
            for m1 in (0, 1)
            for m2 in (0, 1)
            for m3 in (0, 1)]


def fmt(et):
    return spiceypy.et2utc(et, 'C', 0)


def mars_bounds(launch_range):
    fb = FLYBY_BODIES['mars']
    return list(zip(
        np.array([0, fb['tof_min'], 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], fb['tof_max'], 5*YEAR, YEAR,
                  5*YEAR, YEAR, 5*YEAR]) / YEAR))


def direct_bounds(launch_range):
    return list(zip(
        np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], 5*YEAR, YEAR, 5*YEAR, YEAR, 5*YEAR]) / YEAR))


def run_mars_one(s_ids, launch_range, m_tuple, seed):
    bounds = mars_bounds(launch_range)
    m_0, m_1, m_2, m_3 = m_tuple
    res = differential_evolution(
        lambda x: score_paths_flyby(x, *s_ids, launch_range, 'mars', m_0, m_1, m_2, m_3),
        bounds, maxiter=300, tol=1e-7, seed=seed, polish=True,
        popsize=15, mutation=(0.5, 1.3), recombination=0.8,
        updating='deferred')
    ets = _unpack_flyby_input(res.x, launch_range, 'mars')
    full = compute_path_with_flyby(*s_ids, *ets, 'mars', m_0, m_1, m_2, m_3)
    full.update(dict(zip(['et_launch','et_flyby','et_arrive_1','et_stay_1',
                          'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    full['m_tuple'] = m_tuple
    full['seed'] = seed
    return full


def run_direct(s_ids, launch_range):
    """Best direct (no-GA) for comparison, multi-seed × 4 m-combos."""
    bounds = direct_bounds(launch_range)
    best = None
    for m_tuple in [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]:
        for seed in [42, 137, 314]:
            res = differential_evolution(
                lambda x: score_paths(x, *s_ids, launch_range, *m_tuple),
                bounds, maxiter=300, tol=1e-7, seed=seed, polish=True,
                popsize=18, updating='deferred')
            if best is None or res.fun < best['delta_v_total']:
                from core import unpack_input
                ets = unpack_input(res.x, launch_range)
                full = compute_path_deltav(*s_ids, *ets, *m_tuple)
                full.update({'m_tuple': m_tuple, 'seed': seed,
                             'et_launch': ets[0], 'et_arrive_1': ets[1],
                             'et_stay_1': ets[2], 'et_arrive_2': ets[3],
                             'et_stay_2': ets[4], 'et_arrive_3': ets[5]})
                best = full
    return best


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs',
                                 '/Users/rebnoob/Documents/ae105/generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids.\n')

    et_min = spiceypy.str2et(LAUNCH_UTC_MIN)
    et_max = spiceypy.str2et(LAUNCH_UTC_MAX)
    launch_range = [et_min, et_max]

    ids = [get_id_from_asteroid_name(asteroid_list, n) for n in TRIPLET]
    s_ids = [str(int(i)) for i in ids]
    print(f'Optimizing {" -> ".join(TRIPLET)}  ids={ids}')
    print(f'Window: {LAUNCH_UTC_MIN}  to  {LAUNCH_UTC_MAX}')
    print(f'Sweep: {len(SEEDS)} seeds x {len(M_COMBOS)} m-combos = '
          f'{len(SEEDS)*len(M_COMBOS)} runs.\n')

    # ---- Direct baseline ----
    print('--- Direct (no GA) baseline ---')
    t0 = time.time()
    direct = run_direct(s_ids, launch_range)
    print(f'  best direct dv = {direct["delta_v_total"]:.3f} km/s  '
          f'(m={direct["m_tuple"]}, {time.time()-t0:.1f}s)\n')

    # ---- Mars GA multi-start ----
    print('--- Mars GA multi-start ---')
    best = None
    n_done = 0
    n_total = len(SEEDS) * len(M_COMBOS)
    t_start = time.time()
    all_runs = []
    for m_tuple in M_COMBOS:
        for seed in SEEDS:
            n_done += 1
            try:
                res = run_mars_one(s_ids, launch_range, m_tuple, seed)
            except Exception as e:
                print(f'  [{n_done}/{n_total}] m={m_tuple} seed={seed} ERROR: {e}')
                continue
            dv = res['delta_v_total']
            all_runs.append(res)
            mark = ''
            if best is None or dv < best['delta_v_total']:
                best = res
                mark = ' <-- best'
            print(f'  [{n_done}/{n_total}] m={m_tuple} seed={seed:3d}: '
                  f'dv = {dv:7.3f} km/s{mark}')

    elapsed = time.time() - t_start
    print(f'\nMars multi-start done in {elapsed:.1f}s')

    # ---- Detailed report ----
    print('\n' + '='*70)
    print(f'BEST FOUND: {" -> ".join(TRIPLET)} via Mars GA')
    print('='*70)
    print(f'Total dv     : {best["delta_v_total"]:.3f} km/s')
    print(f'Lambert revs : m_(E->M, M->A1, A1->A2, A2->A3) = {best["m_tuple"]}')
    print(f'Seed         : {best["seed"]}')
    print()
    print(f'Launch       : {fmt(best["et_launch"])}')
    print(f'Mars flyby   : {fmt(best["et_flyby"])}    '
          f'(E->M tof = {(best["et_flyby"]-best["et_launch"])/(365.25*DAY):.2f} yr)')
    print(f'Arrive A1    : {fmt(best["et_arrive_1"])}')
    print(f'Depart A1    : {fmt(best["et_stay_1"])}')
    print(f'Arrive A2    : {fmt(best["et_arrive_2"])}')
    print(f'Depart A2    : {fmt(best["et_stay_2"])}')
    print(f'Arrive A3    : {fmt(best["et_arrive_3"])}')
    dur = (best['et_arrive_3'] - best['et_launch']) / (365.25 * DAY)
    print(f'Duration     : {dur:.2f} yr')
    print()

    ldv = float(np.linalg.norm(best['delta_v_launch']))
    fdv = float(abs(best['delta_v_flyby']))
    a1a = float(np.linalg.norm(best['delta_v_A1_arrive']))
    a1l = float(np.linalg.norm(best['delta_v_A1_leave']))
    a2a = float(np.linalg.norm(best['delta_v_A2_arrive']))
    a2l = float(np.linalg.norm(best['delta_v_A2_leave']))
    a3a = float(np.linalg.norm(best['delta_v_A3_arrive']))
    print('dv breakdown (km/s):')
    print(f'  launch        = {ldv:.3f}')
    print(f'  Mars powered  = {fdv:.3f}')
    print(f'  A1 arrive     = {a1a:.3f}')
    print(f'  A1 leave      = {a1l:.3f}')
    print(f'  A2 arrive     = {a2a:.3f}')
    print(f'  A2 leave      = {a2l:.3f}')
    print(f'  A3 arrive     = {a3a:.3f}')
    print()
    print(f'Comparison vs direct:  {best["delta_v_total"]:.3f} km/s (Mars)  '
          f'vs  {direct["delta_v_total"]:.3f} km/s (direct)   '
          f'-> Mars saves {direct["delta_v_total"]-best["delta_v_total"]:+.3f} km/s')

    # Top-5 distinct results
    sorted_runs = sorted(all_runs, key=lambda r: r['delta_v_total'])
    print('\nTop-5 distinct Mars-GA results:')
    seen = set()
    rank = 0
    for r in sorted_runs:
        # Cluster on launch date (rounded to month) + dv
        key = (round(r['et_launch']/(30*DAY)), round(r['delta_v_total'], 2))
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        print(f'  #{rank}: dv = {r["delta_v_total"]:7.3f} km/s   '
              f'launch = {fmt(r["et_launch"])}   '
              f'm = {r["m_tuple"]}   seed = {r["seed"]}')
        if rank >= 5:
            break

    # Save
    out_dir = os.path.join(repo_root, 'optimal_asteroid_paths', 'pkl')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fortuna_themis_psyche_mars_robust.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'best_mars': best, 'best_direct': direct,
                     'all_mars_runs': sorted_runs[:20]}, f)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
