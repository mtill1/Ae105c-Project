"""Optimize three specific triplets using Mars gravity assist only."""

import os
import pickle
import sys
import time

import numpy as np
import spiceypy

from core import load_kernels, get_id_from_asteroid_name, DAY
from optimization import optimize_times_flyby


TRIPLETS = [
    ('AEGINA',  'BEATRIX', 'VESTA'),
    ('FORTUNA', 'THEMIS',  'PSYCHE'),
    ('CERES',   'VESTA',   'PALLAS'),
]

LAUNCH_UTC_MIN = 'Jan 1 12:00:00 UTC 2027'
LAUNCH_UTC_MAX = 'Dec 31 12:00:00 UTC 2035'

# Lambert revolution counts to sweep on the (Earth->Mars, Mars->A1, A1->A2, A2->A3) legs.
M_SWEEP = [(0, 0, 0, 0),
           (0, 1, 0, 0),
           (0, 0, 1, 0),
           (0, 0, 0, 1),
           (1, 0, 0, 0)]


def fmt_et(et):
    return spiceypy.et2utc(et, 'C', 0)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs',
                                 '/Users/rebnoob/Documents/ae105/generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids.\n')

    et_min = spiceypy.str2et(LAUNCH_UTC_MIN)
    et_max = spiceypy.str2et(LAUNCH_UTC_MAX)
    launch_range = [et_min, et_max]

    all_results = []

    for triplet in TRIPLETS:
        n1, n2, n3 = triplet
        ids = [get_id_from_asteroid_name(asteroid_list, n) for n in triplet]
        if any(i == -1 for i in ids):
            missing = [n for n, i in zip(triplet, ids) if i == -1]
            print(f'SKIP {triplet}: not in BSP folder ({missing})')
            all_results.append((triplet, None))
            continue

        s_ids = [str(int(i)) for i in ids]
        print(f'=== {n1} -> {n2} -> {n3}  (ids={ids}) ===')
        t0 = time.time()

        best = None
        for m_tuple in M_SWEEP:
            m_0, m_1, m_2, m_3 = m_tuple
            try:
                res = optimize_times_flyby(*s_ids, launch_range, 'mars',
                                           m_0=m_0, m_1=m_1, m_2=m_2, m_3=m_3)
            except Exception as e:
                print(f'  m={m_tuple}: ERROR {e}')
                continue
            dv = res['delta_v_total']
            print(f'  m={m_tuple}: dv = {dv:.3f} km/s')
            if best is None or dv < best['delta_v_total']:
                best = res
                best['m_tuple'] = m_tuple

        elapsed = time.time() - t0
        print(f'  -> best dv = {best["delta_v_total"]:.3f} km/s   '
              f'(m={best["m_tuple"]}, {elapsed:.1f}s)')

        # Detailed breakdown
        print(f'  launch     : {fmt_et(best["et_launch"])}')
        print(f'  Mars flyby : {fmt_et(best["et_flyby"])}')
        print(f'  arrive  A1 : {fmt_et(best["et_arrive_1"])}')
        print(f'  depart  A1 : {fmt_et(best["et_stay_1"])}')
        print(f'  arrive  A2 : {fmt_et(best["et_arrive_2"])}')
        print(f'  depart  A2 : {fmt_et(best["et_stay_2"])}')
        print(f'  arrive  A3 : {fmt_et(best["et_arrive_3"])}')
        dur_yr = (best['et_arrive_3'] - best['et_launch']) / (365.25 * DAY)
        print(f'  duration   : {dur_yr:.2f} yr')

        # Per-maneuver breakdown
        ldv = float(np.linalg.norm(best['delta_v_launch']))
        fdv = float(abs(best['delta_v_flyby']))
        a1a = float(np.linalg.norm(best['delta_v_A1_arrive']))
        a1l = float(np.linalg.norm(best['delta_v_A1_leave']))
        a2a = float(np.linalg.norm(best['delta_v_A2_arrive']))
        a2l = float(np.linalg.norm(best['delta_v_A2_leave']))
        a3a = float(np.linalg.norm(best['delta_v_A3_arrive']))
        print(f'  dv breakdown (km/s):')
        print(f'    launch      = {ldv:.3f}')
        print(f'    Mars powered= {fdv:.3f}')
        print(f'    A1 arrive   = {a1a:.3f}')
        print(f'    A1 leave    = {a1l:.3f}')
        print(f'    A2 arrive   = {a2a:.3f}')
        print(f'    A2 leave    = {a2l:.3f}')
        print(f'    A3 arrive   = {a3a:.3f}')
        print()

        all_results.append((triplet, best))

    # Summary table
    print('=' * 70)
    print('SUMMARY (Mars gravity assist only)')
    print('=' * 70)
    print(f'{"Path":<40s}  {"dv [km/s]":>10s}  {"duration":>10s}')
    print('-' * 70)
    for triplet, res in all_results:
        path_str = ' -> '.join(triplet)
        if res is None:
            print(f'{path_str:<40s}  {"FAILED":>10s}')
        else:
            dur_yr = (res['et_arrive_3'] - res['et_launch']) / (365.25 * DAY)
            print(f'{path_str:<40s}  {res["delta_v_total"]:>10.3f}  {dur_yr:>9.2f}y')

    out_dir = os.path.join(repo_root, 'optimal_asteroid_paths', 'pkl')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'three_triplets_mars_only.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
