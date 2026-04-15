"""
optimization.py — Delta-v computation, scoring, time optimization, and data generation.

Uses scipy.optimize.differential_evolution for global optimization (replaces
the old grid search + L-BFGS-B approach). Includes two-level optimization
for scaling and beam search for structured path selection.

Combines: compute_path_deltav, compute_mars_path_deltav, score_paths, score_paths_mars,
          optimize_times, optimize_mars_times, generate_optimized_data,
          generate_mars_transfer_optimized, two_level_optimize, beam_search.
"""

import os
import pickle

import numpy as np
import spiceypy
from scipy.optimize import differential_evolution
from tqdm import tqdm

from core import (solve_lambert, get_state, get_mu, get_radius,
                  compute_flyby_dv, DAY, YEAR, WEEK, MONTH, MU_SUN,
                  MAX_MISSION_DURATION, unpack_input, unpack_mars_input)


# =============================================================================
# DELTA-V COMPUTATION
# =============================================================================

def compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                        et_arrive_1, et_stay_1, et_arrive_2,
                        et_stay_2, et_arrive_3, m_1, m_2, m_3):
    earth_launch_r, earth_launch_v = get_state('399', et_launch)
    a_1_arrive_r, a_1_arrive_v = get_state(str(a_id_1), et_arrive_1)
    a_1_leaving_r, a_1_leaving_v = get_state(str(a_id_1), et_stay_1)
    a_2_arrive_r, a_2_arrive_v = get_state(str(a_id_2), et_arrive_2)
    a_2_leaving_r, a_2_leaving_v = get_state(str(a_id_2), et_stay_2)
    a_3_arrive_r, a_3_arrive_v = get_state(str(a_id_3), et_arrive_3)

    earth_lv, a1_arr_lv, ef1 = solve_lambert(
        earth_launch_r, a_1_arrive_r, (et_arrive_1 - et_launch) / DAY, m_1, MU_SUN)
    a1_lv_lv, a2_arr_lv, ef2 = solve_lambert(
        a_1_leaving_r, a_2_arrive_r, (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    a2_lv_lv, a3_arr_lv, ef3 = solve_lambert(
        a_2_leaving_r, a_3_arrive_r, (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if ef1 != 1 or ef2 != 1 or ef3 != 1:
        return {'delta_v_launch': np.array([]), 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3}

    dv_launch = earth_lv - earth_launch_v
    dv_A1_arr = a1_arr_lv - a_1_arrive_v
    dv_A1_lv = a1_lv_lv - a_1_leaving_v
    dv_A2_arr = a2_arr_lv - a_2_arrive_v
    dv_A2_lv = a2_lv_lv - a_2_leaving_v
    dv_A3_arr = a3_arr_lv - a_3_arrive_v

    dv_total = (np.linalg.norm(dv_launch) + np.linalg.norm(dv_A1_arr)
                + np.linalg.norm(dv_A1_lv) + np.linalg.norm(dv_A2_arr)
                + np.linalg.norm(dv_A2_lv) + np.linalg.norm(dv_A3_arr))

    return {'delta_v_launch': dv_launch, 'delta_v_A1_arrive': dv_A1_arr,
            'delta_v_A1_leave': dv_A1_lv, 'delta_v_A2_arrive': dv_A2_arr,
            'delta_v_A2_leave': dv_A2_lv, 'delta_v_A3_arrive': dv_A3_arr,
            'delta_v_total': dv_total}


def compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                             et_arrive_1, et_stay_1, et_arrive_2,
                             et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars):
    earth_r, earth_v = get_state('399', et_launch)
    mars_r, mars_v = get_state('4', et_mars)
    a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
    a1_lv_r, a1_lv_v = get_state(str(a_id_1), et_stay_1)
    a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
    a2_lv_r, a2_lv_v = get_state(str(a_id_2), et_stay_2)
    a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)

    mu_mars = get_mu(4)
    safe_radius = get_radius(499) + 200  # 200 km altitude

    e_lv, mars_arr_v_lam, ef1 = solve_lambert(earth_r, mars_r, -(et_mars-et_launch)/DAY, m_mars, MU_SUN)
    v_mars_leave, a1_arr_lv, ef2 = solve_lambert(mars_r, a1_arr_r, -(et_arrive_1-et_mars)/DAY, m_1, MU_SUN)
    a1_lv_lv, a2_arr_lv, ef3 = solve_lambert(a1_lv_r, a2_arr_r, -(et_arrive_2-et_stay_1)/DAY, m_2, MU_SUN)
    a2_lv_lv, a3_arr_lv, ef4 = solve_lambert(a2_lv_r, a3_arr_r, -(et_arrive_3-et_stay_2)/DAY, m_3, MU_SUN)

    if ef1 != 1 or ef2 != 1 or ef3 != 1 or ef4 != 1:
        return {'delta_v_launch': np.array([]), 'v_mars_leave': np.array([]),
                'delta_v_mars': 1e3, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3}

    dv_launch = e_lv - earth_v
    dv_A1_arr = a1_arr_lv - a1_arr_v
    dv_A1_lv = a1_lv_lv - a1_lv_v
    dv_A2_arr = a2_arr_lv - a2_arr_v
    dv_A2_lv = a2_lv_lv - a2_lv_v
    dv_A3_arr = a3_arr_lv - a3_arr_v
    dv_mars = compute_flyby_dv(mars_arr_v_lam, v_mars_leave, mars_v, mu_mars, safe_radius)

    dv_total = (np.linalg.norm(dv_launch) + np.linalg.norm(dv_A1_arr)
                + np.linalg.norm(dv_A1_lv) + np.linalg.norm(dv_A2_arr)
                + np.linalg.norm(dv_A2_lv) + np.linalg.norm(dv_A3_arr)
                + abs(dv_mars))

    return {'delta_v_launch': dv_launch, 'v_mars_leave': v_mars_leave,
            'delta_v_mars': dv_mars, 'delta_v_A1_arrive': dv_A1_arr,
            'delta_v_A1_leave': dv_A1_lv, 'delta_v_A2_arrive': dv_A2_arr,
            'delta_v_A2_leave': dv_A2_lv, 'delta_v_A3_arrive': dv_A3_arr,
            'delta_v_total': dv_total}


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_paths(input_vec, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_input(input_vec, launch_range)
    if et_arrive_3 - et_launch > MAX_MISSION_DURATION:
        return 1e3
    return compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                               et_arrive_1, et_stay_1, et_arrive_2,
                               et_stay_2, et_arrive_3, m_1, m_2, m_3)['delta_v_total']

def score_paths_mars(input_vec, a_id_1, a_id_2, a_id_3, launch_range,
                     m_1, m_2, m_3, m_mars):
    et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_mars_input(input_vec, launch_range)
    if et_arrive_3 - et_launch > MAX_MISSION_DURATION:
        return 1e3
    return compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                                    et_arrive_1, et_stay_1, et_arrive_2,
                                    et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars)['delta_v_total']


# =============================================================================
# TIME OPTIMIZATION (differential_evolution replaces grid search)
# =============================================================================

def optimize_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    bounds = list(zip(
        np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], 5*YEAR, YEAR, 5*YEAR, YEAR, 5*YEAR]) / YEAR))

    res = differential_evolution(
        lambda x: score_paths(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3),
        bounds, maxiter=300, tol=1e-7, seed=42, polish=True, updating='deferred')

    ets = unpack_input(res.x, launch_range)
    result = compute_path_deltav(a_id_1, a_id_2, a_id_3, *ets, m_1, m_2, m_3)
    result.update(dict(zip(['et_launch','et_arrive_1','et_stay_1',
                            'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    return result


def optimize_times_quick(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    """Fast coarse optimization for two-level search first pass."""
    bounds = list(zip(
        np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], 5*YEAR, YEAR, 5*YEAR, YEAR, 5*YEAR]) / YEAR))
    res = differential_evolution(
        lambda x: score_paths(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3),
        bounds, maxiter=30, tol=0.1, seed=42, polish=False, popsize=5, updating='deferred')
    return res.fun


def optimize_mars_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3, m_mars):
    bounds = list(zip(
        np.array([0, 3*YEAR, 0.2*YEAR, 3*MONTH, 0.2*YEAR, 3*MONTH, 0.2*YEAR]) / YEAR,
        np.array([launch_range[1]-launch_range[0], 5*YEAR, 4*YEAR, YEAR, 4*YEAR, YEAR, 4*YEAR]) / YEAR))

    res = differential_evolution(
        lambda x: score_paths_mars(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3, m_mars),
        bounds, maxiter=300, tol=1e-6, seed=42, polish=True, updating='deferred')

    ets = unpack_mars_input(res.x, launch_range)
    result = compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, *ets, m_1, m_2, m_3, m_mars)
    result.update(dict(zip(['et_launch','et_mars','et_arrive_1','et_stay_1',
                            'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    return result


# =============================================================================
# DATA GENERATION — STANDARD (brute force N^3)
# =============================================================================

def generate_optimized_data(asteroid_list, m_1, m_2, m_3,
                            launch_utc_min, launch_utc_max):
    num = len(asteroid_list)
    default = {'delta_v_total': np.inf, 'et_launch': np.inf}
    data = [[[dict(default) for _ in range(num)] for _ in range(num)] for _ in range(num)]

    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    pbar = tqdm(total=num**3, desc=f'Ms: {m_1},{m_2},{m_3}')
    for i in range(num):
        for j in range(num):
            for k in range(num):
                ids = [asteroid_list[x]['ID'] for x in [i, j, k]]
                if ids[0]==ids[1] or ids[1]==ids[2] or ids[2]==ids[0]:
                    pbar.update(1); continue
                strs = [str(int(x)) for x in ids]
                data[i][j][k] = optimize_times(*strs, launch_dates, m_1, m_2, m_3)
                pbar.update(1)
    pbar.close()

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    path = f'./optimal_asteroid_paths/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(path, 'wb') as f: pickle.dump(data, f)
    return data


def generate_mars_transfer_optimized(asteroid_list, m_1, m_2, m_3, m_mars,
                                     launch_utc_min, launch_utc_max):
    num = len(asteroid_list)
    default = {'delta_v_total': np.inf, 'et_launch': np.inf}
    data = [[[dict(default) for _ in range(num)] for _ in range(num)] for _ in range(num)]

    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    pbar = tqdm(total=num**3, desc=f'Ms: {m_mars},{m_1},{m_2},{m_3}')
    for i in range(num):
        for j in range(num):
            for k in range(num):
                ids = [asteroid_list[x]['ID'] for x in [i, j, k]]
                if ids[0]==ids[1] or ids[1]==ids[2] or ids[2]==ids[0]:
                    pbar.update(1); continue
                strs = [str(int(x)) for x in ids]
                data[i][j][k] = optimize_mars_times(*strs, launch_dates, m_1, m_2, m_3, m_mars)
                pbar.update(1)
    pbar.close()

    os.makedirs('./optimal_asteroid_paths/mars_transfers', exist_ok=True)
    path = f'./optimal_asteroid_paths/mars_transfers/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(path, 'wb') as f: pickle.dump(data, f)
    return data


# =============================================================================
# TWO-LEVEL OPTIMIZATION (coarse filter + fine optimization)
# =============================================================================

def two_level_optimize(asteroid_list, m_1, m_2, m_3,
                       launch_utc_min, launch_utc_max,
                       top_n=50, science_scores=None, alpha=1.0):
    """Coarse-filter all N^3 triplets, then fine-optimize top candidates.

    Parameters
    ----------
    science_scores : dict or None — asteroid NAME -> score (0-10).
    alpha : float — 1.0 = pure delta-v, 0.7 = 70% dv + 30% science.
    """
    num = len(asteroid_list)
    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    # Pass 1: coarse
    print(f"Pass 1: Coarse evaluation ({num}^3 = {num**3} triplets)...")
    coarse = []
    tasks = [(i, j, k) for i in range(num) for j in range(num) for k in range(num)
             if len({asteroid_list[i]['ID'], asteroid_list[j]['ID'], asteroid_list[k]['ID']}) == 3]

    for i, j, k in tqdm(tasks, desc="Coarse"):
        strs = [str(int(asteroid_list[x]['ID'])) for x in [i, j, k]]
        dv = optimize_times_quick(*strs, launch_dates, m_1, m_2, m_3)
        coarse.append((i, j, k, dv))

    # Score with optional science weighting
    if science_scores and alpha < 1.0:
        for idx, (i, j, k, dv) in enumerate(coarse):
            sci = sum(science_scores.get(asteroid_list[x]['NAME'], 5.0) for x in [i, j, k])
            coarse[idx] = (i, j, k, alpha * dv + (1-alpha) * (30 - sci))

    coarse.sort(key=lambda x: x[3])
    top = coarse[:top_n]

    # Pass 2: fine
    print(f"\nPass 2: Fine optimization on top {len(top)} candidates...")
    results = []
    for i, j, k, _ in tqdm(top, desc="Fine"):
        strs = [str(int(asteroid_list[x]['ID'])) for x in [i, j, k]]
        result = optimize_times(*strs, launch_dates, m_1, m_2, m_3)
        results.append((i, j, k, result))

    results.sort(key=lambda x: x[3]['delta_v_total'])

    print(f"\n{'='*80}\nTOP 10 PATHS\n{'='*80}")
    for rank, (i, j, k, res) in enumerate(results[:10], 1):
        names = f"{asteroid_list[i]['NAME']} -> {asteroid_list[j]['NAME']} -> {asteroid_list[k]['NAME']}"
        dv = res['delta_v_total']
        ldv = np.linalg.norm(res['delta_v_launch']) if len(res['delta_v_launch']) > 0 else float('inf')
        print(f"  #{rank}: {names}  |  dv={dv:.2f}  launch_dv={ldv:.2f} km/s")

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    with open(f'./optimal_asteroid_paths/two_level_{m_1}_{m_2}_{m_3}.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results


# =============================================================================
# BEAM SEARCH (structured multi-stage path selection)
# =============================================================================

def beam_search(asteroid_list, launch_utc_min, launch_utc_max,
                beam_width=10, m_1=0, m_2=0, m_3=0,
                science_scores=None, alpha=1.0):
    """Beam search: keep top-K candidates at each leg instead of committing to one.

    Stage 1: Earth -> A_i (all asteroids), keep top K.
    Stage 2: For K survivors, A_i -> A_j, keep top K.
    Stage 3: For K survivors, A_j -> A_k, keep top K.
    """
    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    num = len(asteroid_list)

    def _sci_score(names):
        if not science_scores or alpha >= 1.0:
            return 0
        return -(1-alpha) * sum(science_scores.get(n, 5.0) for n in names)

    def _leg_dv(body1_id, body2_id, et_depart, tof_bounds, m):
        """Optimize a single leg and return (best_dv, et_depart, et_arrive)."""
        bounds = [(tof_bounds[0]/YEAR, tof_bounds[1]/YEAR)]
        def obj(x):
            et_arr = et_depart + x[0] * YEAR
            if et_arr - et_min > MAX_MISSION_DURATION:
                return 1e3
            r1, v1 = get_state(body1_id, et_depart)
            r2, v2 = get_state(body2_id, et_arr)
            vl1, vl2, flag = solve_lambert(r1, r2, (et_arr-et_depart)/DAY, m, MU_SUN)
            if flag != 1:
                return 1e3
            return np.linalg.norm(vl1-v1) + np.linalg.norm(vl2-v2)

        res = differential_evolution(obj, bounds, maxiter=50, tol=0.1,
                                     seed=42, polish=True, popsize=5, updating='deferred')
        et_arrive = et_depart + res.x[0] * YEAR
        return res.fun, et_depart, et_arrive

    # Stage 1
    print("Beam Stage 1: Earth -> each asteroid...")
    stage1 = []
    for i in tqdm(range(num), desc="Stage 1"):
        # Try multiple launch dates across the window
        best_dv = np.inf
        best_ets = None
        for launch_frac in np.linspace(0, 1, 5):
            et_launch = et_min + launch_frac * (et_max - et_min)
            dv, _, et_arr = _leg_dv('399', str(int(asteroid_list[i]['ID'])),
                                     et_launch, (2*WEEK, 5*YEAR), m_1)
            if dv < best_dv:
                best_dv = dv
                best_ets = (et_launch, et_arr)
        stage1.append((i, best_dv, best_ets))

    stage1.sort(key=lambda x: alpha*x[1] + _sci_score([asteroid_list[x[0]]['NAME']]))
    stage1 = stage1[:beam_width]

    # Stage 2
    print(f"Beam Stage 2: {len(stage1)} survivors -> each asteroid...")
    stage2 = []
    for i, dv1, (et_launch, et_arr1) in stage1:
        for j in range(num):
            if j == i: continue
            # Stay at asteroid 1, then depart
            for stay_frac in [0.3, 0.6]:
                et_stay_end = et_arr1 + stay_frac * YEAR
                dv2, _, et_arr2 = _leg_dv(str(int(asteroid_list[i]['ID'])),
                                           str(int(asteroid_list[j]['ID'])),
                                           et_stay_end, (2*WEEK, 5*YEAR), m_2)
                total = dv1 + dv2
                stage2.append((i, j, total, et_launch, et_arr1, et_stay_end, et_arr2))

    stage2.sort(key=lambda x: alpha*x[2] + _sci_score(
        [asteroid_list[x[0]]['NAME'], asteroid_list[x[1]]['NAME']]))
    stage2 = stage2[:beam_width]

    # Stage 3
    print(f"Beam Stage 3: {len(stage2)} survivors -> each asteroid...")
    stage3 = []
    for i, j, dv12, et_launch, et_arr1, et_stay1, et_arr2 in stage2:
        for k in range(num):
            if k == i or k == j: continue
            for stay_frac in [0.3, 0.6]:
                et_stay_end2 = et_arr2 + stay_frac * YEAR
                if et_stay_end2 - et_launch > MAX_MISSION_DURATION:
                    continue
                dv3, _, et_arr3 = _leg_dv(str(int(asteroid_list[j]['ID'])),
                                           str(int(asteroid_list[k]['ID'])),
                                           et_stay_end2, (2*WEEK, 5*YEAR), m_3)
                total = dv12 + dv3
                names = [asteroid_list[x]['NAME'] for x in [i, j, k]]
                stage3.append((i, j, k, total, names,
                               et_launch, et_arr1, et_stay1, et_arr2, et_stay_end2, et_arr3))

    stage3.sort(key=lambda x: alpha*x[3] + _sci_score(x[4]))

    print(f"\n{'='*80}\nBEAM SEARCH RESULTS (K={beam_width})\n{'='*80}")
    for rank, (i, j, k, dv, names, *_) in enumerate(stage3[:10], 1):
        print(f"  #{rank}: {' -> '.join(names)}  |  total dv = {dv:.2f} km/s")

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    with open(f'./optimal_asteroid_paths/beam_K{beam_width}.pkl', 'wb') as f:
        pickle.dump(stage3, f)
    return stage3
