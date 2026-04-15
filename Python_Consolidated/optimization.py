"""
optimization.py — Delta-v computation, scoring, time optimization, and data generation.

Combines: compute_path_deltav, compute_mars_path_deltav, score_paths, score_paths_mars,
          optimize_times, optimize_mars_times, generate_optimized_data, generate_mars_transfer_optimized.
"""

import os
import time
import pickle

import numpy as np
import spiceypy
from scipy.optimize import minimize
from tqdm import tqdm

from core import (solve_lambert, get_state, get_mu, get_radius,
                  compute_flyby_dv, DAY, YEAR, WEEK, MONTH, MU_SUN,
                  unpack_input, unpack_mars_input)


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

    earth_lambert_velocity, a_1_arrive_lambert_velocity, exit_flag_1 = \
        solve_lambert(earth_launch_r, a_1_arrive_r,
                      (et_arrive_1 - et_launch) / DAY, m_1, MU_SUN)
    a_1_leave_lambert_velocity, a_2_arrive_lambert_velocity, exit_flag_2 = \
        solve_lambert(a_1_leaving_r, a_2_arrive_r,
                      (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    a_2_leave_lambert_velocity, a_3_arrive_lambert_velocity, exit_flag_3 = \
        solve_lambert(a_2_leaving_r, a_3_arrive_r,
                      (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if exit_flag_1 != 1 or exit_flag_2 != 1 or exit_flag_3 != 1:
        return {
            'delta_v_launch': np.array([]),
            'delta_v_A1_arrive': np.array([]),
            'delta_v_A1_leave': np.array([]),
            'delta_v_A2_arrive': np.array([]),
            'delta_v_A2_leave': np.array([]),
            'delta_v_A3_arrive': np.array([]),
            'delta_v_total': 1e3,
        }

    delta_v_launch = earth_lambert_velocity - earth_launch_v
    delta_v_A1_arrive = a_1_arrive_lambert_velocity - a_1_arrive_v
    delta_v_A1_leave = a_1_leave_lambert_velocity - a_1_leaving_v
    delta_v_A2_arrive = a_2_arrive_lambert_velocity - a_2_arrive_v
    delta_v_A2_leave = a_2_leave_lambert_velocity - a_2_leaving_v
    delta_v_A3_arrive = a_3_arrive_lambert_velocity - a_3_arrive_v

    delta_v_total = (np.linalg.norm(delta_v_A1_arrive)
                     + np.linalg.norm(delta_v_A1_leave)
                     + np.linalg.norm(delta_v_A2_arrive)
                     + np.linalg.norm(delta_v_A2_leave)
                     + np.linalg.norm(delta_v_A3_arrive))

    return {
        'delta_v_launch': delta_v_launch,
        'delta_v_A1_arrive': delta_v_A1_arrive,
        'delta_v_A1_leave': delta_v_A1_leave,
        'delta_v_A2_arrive': delta_v_A2_arrive,
        'delta_v_A2_leave': delta_v_A2_leave,
        'delta_v_A3_arrive': delta_v_A3_arrive,
        'delta_v_total': delta_v_total,
    }

def compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                             et_arrive_1, et_stay_1, et_arrive_2,
                             et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars):
    earth_launch_r, earth_launch_v = get_state('399', et_launch)
    mars_flyby_r, mars_flyby_v = get_state('4', et_mars)
    a_1_arrive_r, a_1_arrive_v = get_state(str(a_id_1), et_arrive_1)
    a_1_leaving_r, a_1_leaving_v = get_state(str(a_id_1), et_stay_1)
    a_2_arrive_r, a_2_arrive_v = get_state(str(a_id_2), et_arrive_2)
    a_2_leaving_r, a_2_leaving_v = get_state(str(a_id_2), et_stay_2)
    a_3_arrive_r, a_3_arrive_v = get_state(str(a_id_3), et_arrive_3)

    mu_mars = get_mu(4)
    r_p = get_radius(499)
    HEIGHT_THRESHOLD = 200  # km
    safe_radius = r_p + HEIGHT_THRESHOLD

    earth_lambert_velocity, mars_arrive_velocity, exit_flag_1 = \
        solve_lambert(earth_launch_r, mars_flyby_r,
                      -(et_mars - et_launch) / DAY, m_mars, MU_SUN)

    v_mars_leave, a_1_arrive_lambert_velocity, exit_flag_2 = \
        solve_lambert(mars_flyby_r, a_1_arrive_r,
                      -(et_arrive_1 - et_mars) / DAY, m_1, MU_SUN)

    a_1_leave_lambert_velocity, a_2_arrive_lambert_velocity, exit_flag_3 = \
        solve_lambert(a_1_leaving_r, a_2_arrive_r,
                      -(et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    a_2_leave_lambert_velocity, a_3_arrive_lambert_velocity, exit_flag_4 = \
        solve_lambert(a_2_leaving_r, a_3_arrive_r,
                      -(et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if exit_flag_1 != 1 or exit_flag_2 != 1 or exit_flag_3 != 1 or exit_flag_4 != 1:
        return {
            'delta_v_launch': np.array([]),
            'v_mars_leave': np.array([]),
            'delta_v_mars': 1e3,
            'delta_v_A1_arrive': np.array([]),
            'delta_v_A1_leave': np.array([]),
            'delta_v_A2_arrive': np.array([]),
            'delta_v_A2_leave': np.array([]),
            'delta_v_A3_arrive': np.array([]),
            'delta_v_total': 1e3,
        }

    delta_v_launch = earth_lambert_velocity - earth_launch_v
    delta_v_A1_arrive = a_1_arrive_lambert_velocity - a_1_arrive_v
    delta_v_A1_leave = a_1_leave_lambert_velocity - a_1_leaving_v
    delta_v_A2_arrive = a_2_arrive_lambert_velocity - a_2_arrive_v
    delta_v_A2_leave = a_2_leave_lambert_velocity - a_2_leaving_v
    delta_v_A3_arrive = a_3_arrive_lambert_velocity - a_3_arrive_v

    delta_v_mars = compute_flyby_dv(mars_arrive_velocity, v_mars_leave,
                                    mars_flyby_v, mu_mars, safe_radius)

    delta_v_total = (np.linalg.norm(delta_v_launch)
                     + np.linalg.norm(delta_v_A1_arrive)
                     + np.linalg.norm(delta_v_A1_leave)
                     + np.linalg.norm(delta_v_A2_arrive)
                     + np.linalg.norm(delta_v_A2_leave)
                     + np.linalg.norm(delta_v_A3_arrive)
                     + abs(delta_v_mars))

    return {
        'delta_v_launch': delta_v_launch,
        'v_mars_leave': v_mars_leave,
        'delta_v_mars': delta_v_mars,
        'delta_v_A1_arrive': delta_v_A1_arrive,
        'delta_v_A1_leave': delta_v_A1_leave,
        'delta_v_A2_arrive': delta_v_A2_arrive,
        'delta_v_A2_leave': delta_v_A2_leave,
        'delta_v_A3_arrive': delta_v_A3_arrive,
        'delta_v_total': delta_v_total,
    }



# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_paths(input_vec, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_input(input_vec, launch_range)

    result = compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                                 et_arrive_1, et_stay_1, et_arrive_2,
                                 et_stay_2, et_arrive_3, m_1, m_2, m_3)

    return result['delta_v_total']

def score_paths_mars(input_vec, a_id_1, a_id_2, a_id_3, launch_range,
                     m_1, m_2, m_3, m_mars):
    et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_mars_input(input_vec, launch_range)

    result = compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                                      et_arrive_1, et_stay_1, et_arrive_2,
                                      et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars)

    return result['delta_v_total']



# =============================================================================
# TIME OPTIMIZATION
# =============================================================================

def optimize_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    MAX_STAYTIME = YEAR
    MIN_STAYTIME = 3 * MONTH

    N_RES = [5, 3, 2, 3, 2, 3]

    LOWER_BOUND = np.array([0, 2 * WEEK, MIN_STAYTIME, 2 * WEEK,
                            MIN_STAYTIME, 2 * WEEK]) / YEAR
    UPPER_BOUND = np.array([launch_range[1] - launch_range[0], 8 * YEAR, MAX_STAYTIME,
                            8 * YEAR, MAX_STAYTIME, 8 * YEAR]) / YEAR

    iter_vec = [0] * len(N_RES)

    DELTA_GUESS = (UPPER_BOUND - LOWER_BOUND) / np.array(N_RES)

    min_times = np.full(6, np.inf)
    min_score = np.inf

    while iter_vec[-1] <= N_RES[-1]:
        while any(iter_vec[j] > N_RES[j] for j in range(len(N_RES))):
            for j in range(1, len(N_RES)):
                if iter_vec[j] >= N_RES[j]:
                    iter_vec[j - 1] += 1
                    iter_vec[j] = 0

        input_guess = LOWER_BOUND + DELTA_GUESS * np.array(iter_vec)

        bounds = list(zip(LOWER_BOUND, UPPER_BOUND))

        result = minimize(
            lambda x: score_paths(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3),
            input_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-7}
        )

        optimized_vector = result.x
        delta_v_score = result.fun

        et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
            unpack_input(optimized_vector, launch_range)

        iter_vec[-1] += 1

        if delta_v_score < min_score:
            min_times = np.array([et_launch, et_arrive_1, et_stay_1,
                                  et_arrive_2, et_stay_2, et_arrive_3])
            min_score = delta_v_score

    et_launch = min_times[0]
    et_arrive_1 = min_times[1]
    et_stay_1 = min_times[2]
    et_arrive_2 = min_times[3]
    et_stay_2 = min_times[4]
    et_arrive_3 = min_times[5]

    result = compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                                 et_arrive_1, et_stay_1, et_arrive_2,
                                 et_stay_2, et_arrive_3, m_1, m_2, m_3)

    return {
        'delta_v_launch': result['delta_v_launch'],
        'delta_v_A1_arrive': result['delta_v_A1_arrive'],
        'delta_v_A1_leave': result['delta_v_A1_leave'],
        'delta_v_A2_arrive': result['delta_v_A2_arrive'],
        'delta_v_A2_leave': result['delta_v_A2_leave'],
        'delta_v_A3_arrive': result['delta_v_A3_arrive'],
        'delta_v_total': result['delta_v_total'],
        'et_launch': et_launch,
        'et_arrive_1': et_arrive_1,
        'et_stay_1': et_stay_1,
        'et_arrive_2': et_arrive_2,
        'et_stay_2': et_stay_2,
        'et_arrive_3': et_arrive_3,
    }

def optimize_mars_times(a_id_1, a_id_2, a_id_3, launch_range,
                        m_1, m_2, m_3, m_mars):
    MAX_STAYTIME = YEAR
    MIN_STAYTIME = 3 * MONTH

    N_RES = [5, 3, 3, 2, 3, 2, 3]

    LOWER_BOUND = np.array([0, 3 * YEAR, 0.2 * YEAR, MIN_STAYTIME, 0.2 * YEAR,
                            MIN_STAYTIME, 0.2 * YEAR]) / YEAR
    UPPER_BOUND = np.array([launch_range[1] - launch_range[0], 8 * YEAR, 4 * YEAR,
                            MAX_STAYTIME, 4 * YEAR, MAX_STAYTIME, 4 * YEAR]) / YEAR

    iter_vec = [0] * len(N_RES)

    DELTA_GUESS = (UPPER_BOUND - LOWER_BOUND) / np.array(N_RES)

    min_times = np.full(len(N_RES), np.inf)
    min_score = np.inf

    while iter_vec[-1] <= N_RES[-1]:
        while any(iter_vec[j] > N_RES[j] for j in range(len(N_RES))):
            for j in range(1, len(N_RES)):
                if iter_vec[j] >= N_RES[j]:
                    iter_vec[j - 1] += 1
                    iter_vec[j] = 0

        input_guess = LOWER_BOUND + DELTA_GUESS * np.array(iter_vec)

        bounds = list(zip(LOWER_BOUND, UPPER_BOUND))

        result = minimize(
            lambda x: score_paths_mars(x, a_id_1, a_id_2, a_id_3,
                                       launch_range, m_1, m_2, m_3, m_mars),
            input_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-3}
        )

        optimized_vector = result.x
        delta_v_score = result.fun

        et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
            unpack_mars_input(optimized_vector, launch_range)

        iter_vec[-1] += 1

        if delta_v_score < min_score:
            min_times = np.array([et_launch, et_mars, et_arrive_1, et_stay_1,
                                  et_arrive_2, et_stay_2, et_arrive_3])
            min_score = delta_v_score

    et_launch = min_times[0]
    et_mars = min_times[1]
    et_arrive_1 = min_times[2]
    et_stay_1 = min_times[3]
    et_arrive_2 = min_times[4]
    et_stay_2 = min_times[5]
    et_arrive_3 = min_times[6]

    result = compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                                      et_arrive_1, et_stay_1, et_arrive_2,
                                      et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars)

    return {
        'delta_v_launch': result['delta_v_launch'],
        'v_mars_leave': result['v_mars_leave'],
        'delta_v_mars': result['delta_v_mars'],
        'delta_v_A1_arrive': result['delta_v_A1_arrive'],
        'delta_v_A1_leave': result['delta_v_A1_leave'],
        'delta_v_A2_arrive': result['delta_v_A2_arrive'],
        'delta_v_A2_leave': result['delta_v_A2_leave'],
        'delta_v_A3_arrive': result['delta_v_A3_arrive'],
        'delta_v_total': result['delta_v_total'],
        'et_launch': et_launch,
        'et_mars': et_mars,
        'et_arrive_1': et_arrive_1,
        'et_stay_1': et_stay_1,
        'et_arrive_2': et_arrive_2,
        'et_stay_2': et_stay_2,
        'et_arrive_3': et_arrive_3,
    }



# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_optimized_data(asteroid_list, m_1, m_2, m_3,
                            launch_utc_min, launch_utc_max):
    num_asteroids = len(asteroid_list)
    total_operations = num_asteroids ** 3

    elapsed_times = np.zeros(total_operations)

    default_entry = {
        'delta_v_launch': np.inf,
        'delta_v_A1_arrive': np.inf,
        'delta_v_A1_leave': np.inf,
        'delta_v_A2_arrive': np.inf,
        'delta_v_A2_leave': np.inf,
        'delta_v_A3_arrive': np.inf,
        'delta_v_total': np.inf,
        'et_launch': np.inf,
        'et_arrive_1': np.inf,
        'et_stay_1': np.inf,
        'et_arrive_2': np.inf,
        'et_stay_2': np.inf,
        'et_arrive_3': np.inf,
    }

    asteroid_optimized_data = [
        [[dict(default_entry) for _ in range(num_asteroids)]
         for _ in range(num_asteroids)]
        for _ in range(num_asteroids)
    ]

    et_launch_min = spiceypy.str2et(launch_utc_min)
    et_launch_max = spiceypy.str2et(launch_utc_max)

    launch_dates = [et_launch_min, et_launch_max]

    pbar = tqdm(total=total_operations,
                desc=f'Ms: {m_1}, {m_2}, {m_3}')

    for i in range(num_asteroids):
        for j in range(num_asteroids):
            for k in range(num_asteroids):
                current_operation = k + j * num_asteroids + i * num_asteroids ** 2
                t_start = time.time()

                a_id_1 = asteroid_list[i]['ID']
                a_id_2 = asteroid_list[j]['ID']
                a_id_3 = asteroid_list[k]['ID']

                if a_id_1 == a_id_2 or a_id_2 == a_id_3 or a_id_3 == a_id_1:
                    elapsed_times[current_operation] = time.time() - t_start
                    pbar.update(1)
                    continue

                a_id_1_str = str(int(asteroid_list[i]['ID']))
                a_id_2_str = str(int(asteroid_list[j]['ID']))
                a_id_3_str = str(int(asteroid_list[k]['ID']))

                asteroid_optimized_data[i][j][k] = optimize_times(
                    a_id_1_str, a_id_2_str, a_id_3_str,
                    launch_dates, m_1, m_2, m_3)

                elapsed_times[current_operation] = time.time() - t_start
                pbar.update(1)

    pbar.close()

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    save_path = f'./optimal_asteroid_paths/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(asteroid_optimized_data, f)

    return asteroid_optimized_data

def generate_mars_transfer_optimized(asteroid_list, m_1, m_2, m_3, m_mars,
                                     launch_utc_min, launch_utc_max):
    num_asteroids = len(asteroid_list)
    total_operations = num_asteroids ** 3

    elapsed_times = np.zeros(total_operations)

    default_entry = {
        'delta_v_launch': np.inf,
        'v_mars_leave': np.inf,
        'delta_v_mars': np.inf,
        'delta_v_A1_arrive': np.inf,
        'delta_v_A1_leave': np.inf,
        'delta_v_A2_arrive': np.inf,
        'delta_v_A2_leave': np.inf,
        'delta_v_A3_arrive': np.inf,
        'delta_v_total': np.inf,
        'et_launch': np.inf,
        'et_mars': np.inf,
        'et_arrive_1': np.inf,
        'et_stay_1': np.inf,
        'et_arrive_2': np.inf,
        'et_stay_2': np.inf,
        'et_arrive_3': np.inf,
    }

    asteroid_optimized_data = [
        [[dict(default_entry) for _ in range(num_asteroids)]
         for _ in range(num_asteroids)]
        for _ in range(num_asteroids)
    ]

    et_launch_min = spiceypy.str2et(launch_utc_min)
    et_launch_max = spiceypy.str2et(launch_utc_max)

    launch_dates = [et_launch_min, et_launch_max]

    pbar = tqdm(total=total_operations,
                desc=f'Ms: {m_mars}, {m_1}, {m_2}, {m_3}')

    for i in range(num_asteroids):
        for j in range(num_asteroids):
            for k in range(num_asteroids):
                current_operation = k + j * num_asteroids + i * num_asteroids ** 2
                t_start = time.time()

                a_id_1 = asteroid_list[i]['ID']
                a_id_2 = asteroid_list[j]['ID']
                a_id_3 = asteroid_list[k]['ID']

                if a_id_1 == a_id_2 or a_id_2 == a_id_3 or a_id_3 == a_id_1:
                    elapsed_times[current_operation] = time.time() - t_start
                    pbar.update(1)
                    continue

                a_id_1_str = str(int(asteroid_list[i]['ID']))
                a_id_2_str = str(int(asteroid_list[j]['ID']))
                a_id_3_str = str(int(asteroid_list[k]['ID']))

                asteroid_optimized_data[i][j][k] = optimize_mars_times(
                    a_id_1_str, a_id_2_str, a_id_3_str,
                    launch_dates, m_1, m_2, m_3, m_mars)

                elapsed_times[current_operation] = time.time() - t_start
                pbar.update(1)

    pbar.close()

    os.makedirs('./optimal_asteroid_paths/mars_transfers', exist_ok=True)
    save_path = f'./optimal_asteroid_paths/mars_transfers/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(asteroid_optimized_data, f)

    return asteroid_optimized_data
