"""Consolidated greedy trajectory selection code.

This module consolidates all greedy algorithm files for asteroid trajectory
selection into a single file.  It includes delta-v computation, scoring,
time optimization, data generation, flight-path animation, path analysis,
plotting utilities, and the main runner script.

Source files consolidated:
    compute_path_deltav_greedy.py
    score_paths_greedy.py
    optimize_greedy_times.py
    generate_greedy_optimized_data.py
    greedy_flightpath_animation.py
    greedy_selector.py
    find_all_paths_greedy.py
    find_best_path_greedy.py
    plot_best_path.py
    plot_dv_range.py
"""

import os
import time
import pickle
import glob
import itertools

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd

from core import lambert, two_body_sim, load_kernels, DAY, YEAR, MONTH, WEEK, MINUTE, HOUR


###############################################################################
# DELTA-V COMPUTATION
###############################################################################

def compute_path_deltav_greedy(launch_body, landing_body, goal_body,
                               launch_date, arrival_date, goal_date,
                               m_1, m_2):
    """Compute delta-v for a greedy path with an optional gravity-assist flyby.

    Parameters
    ----------
    launch_body : str
        NAIF ID of the departure body.
    landing_body : str
        NAIF ID of the flyby body, or "-1" for a direct transfer.
    goal_body : str
        NAIF ID of the destination body.
    launch_date : float
        Ephemeris time of launch.
    arrival_date : float
        Ephemeris time of flyby (ignored if landing_body == "-1").
    goal_date : float
        Ephemeris time of arrival at the goal body.
    m_1 : float
        Lambert revolution parameter for the first leg.
        Sign controls transfer direction; abs value is rounded to number of revolutions.
    m_2 : float
        Lambert revolution parameter for the second leg.

    Returns
    -------
    dv_launch : float
    dv_arrive : float
    dv_goal : float
    dv_total : float
    lambert_launch : ndarray or None
    lambert_arrive_in : ndarray or None
    lambert_arrive_out : ndarray or None
    lambert_goal : ndarray or None
    """
    sgn_1 = np.sign(m_1)
    if sgn_1 == 0:
        sgn_1 = 1

    sgn_2 = np.sign(m_2)
    if sgn_2 == 0:
        sgn_2 = 1

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]

    # --- Direct transfer (no flyby) ---
    if landing_body == "-1":
        launch_state, _ = spiceypy.spkezr(launch_body, launch_date,
                                          'ECLIPJ2000', 'NONE', '10')
        goal_state, _ = spiceypy.spkezr(goal_body, goal_date,
                                        'ECLIPJ2000', 'NONE', '10')

        lambert_launch, lambert_goal, _, exit_flag = lambert(
            launch_state[0:3], goal_state[0:3],
            sgn_1 * (goal_date - launch_date) / DAY,
            round(abs(m_1)), myu_sun)

        lambert_arrive_out = None
        lambert_arrive_in = None

        if exit_flag != 1:
            return (1e3, 1e3, 1e3, 1e3, None, None, None, None)

        dv_launch = np.linalg.norm(lambert_launch - launch_state[3:6])
        dv_goal = np.linalg.norm(lambert_goal - goal_state[3:6])
        dv_arrive = 0.0
        dv_total = dv_launch + dv_goal

        return (dv_launch, dv_arrive, dv_goal, dv_total,
                lambert_launch, lambert_arrive_in, lambert_arrive_out, lambert_goal)

    # --- Two-leg transfer with flyby ---
    launch_state, _ = spiceypy.spkezr(launch_body, launch_date,
                                      'ECLIPJ2000', 'NONE', '10')
    arrive_state, _ = spiceypy.spkezr(landing_body, arrival_date,
                                      'ECLIPJ2000', 'NONE', '10')
    goal_state, _ = spiceypy.spkezr(goal_body, goal_date,
                                    'ECLIPJ2000', 'NONE', '10')

    HEIGHT_THRESHOLD = 100  # km

    lambert_launch, lambert_arrive_in, _, exit_flag_1 = lambert(
        launch_state[0:3], arrive_state[0:3],
        sgn_1 * (arrival_date - launch_date) / DAY,
        round(abs(m_1)), myu_sun)

    lambert_arrive_out, lambert_goal, _, exit_flag_2 = lambert(
        arrive_state[0:3], goal_state[0:3],
        sgn_2 * (goal_date - arrival_date) / DAY,
        round(abs(m_2)), myu_sun)

    if exit_flag_1 != 1 or exit_flag_2 != 1:
        return (1e3, 1e3, 1e3, 1e3, None, None, None, None)

    dv_launch = np.linalg.norm(lambert_launch - launch_state[3:6])

    v_inf_in = lambert_arrive_in - arrive_state[3:6]
    v_inf_out = lambert_arrive_out - arrive_state[3:6]

    flyby_id = int(float(landing_body))
    _, myu_flyby_vals = spiceypy.bodvcd(flyby_id, 'GM', 10)
    myu_flyby = myu_flyby_vals[0]

    # For Mars (body 4), use body 499 for radii
    if flyby_id == 4:
        _, r_p_vals = spiceypy.bodvcd(499, 'RADII', 10)
    else:
        _, r_p_vals = spiceypy.bodvcd(flyby_id, 'RADII', 10)
    r_p = r_p_vals[0]

    delta_des = np.arccos(
        np.clip(np.dot(v_inf_in, v_inf_out)
                / (np.linalg.norm(v_inf_in) * np.linalg.norm(v_inf_out)),
                -1.0, 1.0))
    delta_max = 2 * np.arcsin(
        1 / (1 + (r_p + HEIGHT_THRESHOLD)
             * (np.linalg.norm(v_inf_in) ** 2) / myu_flyby))

    dv_arrive = 0.0
    if delta_des > delta_max:
        v_p_in = np.sqrt(np.linalg.norm(v_inf_in) ** 2
                         + 2 * myu_flyby / (HEIGHT_THRESHOLD + r_p))
        v_p_out = np.sqrt(np.linalg.norm(v_inf_out) ** 2
                          + 2 * myu_flyby / (HEIGHT_THRESHOLD + r_p))
        d_delta = delta_des - delta_max
        dv_arrive = np.sqrt(v_p_in ** 2 + v_p_out ** 2
                            - 2 * v_p_in * v_p_out * np.cos(d_delta))

    dv_goal = np.linalg.norm(lambert_goal - goal_state[3:6])
    dv_total = dv_launch + dv_arrive + dv_goal

    return (dv_launch, dv_arrive, dv_goal, dv_total,
            lambert_launch, lambert_arrive_in, lambert_arrive_out, lambert_goal)


###############################################################################
# SCORING
###############################################################################

def score_paths_greedy(time_vector, launch_range, launch_body,
                       landing_body, goal_body, m_1, m_2):
    """Evaluate delta-v for a greedy path given a time parameter vector.

    Parameters
    ----------
    time_vector : array-like of length 3
        [launch_offset, transfer_1_time, transfer_2_time] in units of YEAR.
    launch_range : array-like of length 2
        [et_min, et_max] launch window in ephemeris time.
    launch_body : str
        NAIF ID of departure body.
    landing_body : str
        NAIF ID of flyby body, or "-1" for direct transfer.
    goal_body : str
        NAIF ID of destination body.
    m_1 : float
        Lambert revolution parameter for the first leg.
    m_2 : float
        Lambert revolution parameter for the second leg.

    Returns
    -------
    dv_total : float
        Total delta-v in km/s.
    """
    launch_date = time_vector[0] * YEAR + launch_range[0]
    arrival_date = time_vector[1] * YEAR + launch_date
    goal_date = time_vector[2] * YEAR + arrival_date

    _, _, _, dv_total, _, _, _, _ = compute_path_deltav_greedy(
        launch_body, landing_body, goal_body,
        launch_date, arrival_date, goal_date, m_1, m_2)

    return dv_total


###############################################################################
# OPTIMIZATION
###############################################################################

def optimize_greedy_times(a_id_1, a_id_2, a_id_3, launch_range, M):
    """Optimize departure/flyby/arrival times for a 3-leg greedy trajectory.

    Parameters
    ----------
    a_id_1 : str
        NAIF ID of the first asteroid.
    a_id_2 : str
        NAIF ID of the second asteroid.
    a_id_3 : str
        NAIF ID of the third asteroid.
    launch_range : array-like of length 2
        [et_min, et_max] launch window in ephemeris time.
    M : array-like, shape (3, 2)
        Lambert revolution parameters for each leg.
        M[i, 0] = m_1 for leg i, M[i, 1] = m_2 for leg i.

    Returns
    -------
    optimized_output : list of 3 dicts
        Each dict contains: dv_launch, dv_arrive, dv_goal, dv_total,
        LAMBERT_LAUNCH, LAMBERT_ARRIVE_IN, LAMBERT_ARRIVE_OUT, LAMBERT_GOAL,
        FLYBY_BODY, et_launch, et_flyby, et_goal.
    """
    N_RES = 2

    LOWER_BOUND = np.array([0, 0.5 * YEAR, 0.5 * YEAR]) / YEAR
    UPPER_BOUND = np.array([launch_range[1] - launch_range[0],
                            2.5 * YEAR, 2.5 * YEAR]) / YEAR

    DELTA_GUESS = (UPPER_BOUND - LOWER_BOUND) / N_RES

    FLYBY_IDS = ["399", "4", "-1"]

    # bodies_strs: index 0 = Earth, 1 = a_id_1, 2 = a_id_2, 3 = a_id_3
    bodies_strs = ["399", str(a_id_1), str(a_id_2), str(a_id_3)]

    # Initialize output as list of 3 dicts with Inf values
    optimized_output = []
    for _ in range(3):
        optimized_output.append({
            "dv_launch": np.inf, "dv_arrive": np.inf,
            "dv_goal": np.inf, "dv_total": np.inf,
            "LAMBERT_LAUNCH": np.inf, "LAMBERT_ARRIVE_IN": np.inf,
            "LAMBERT_ARRIVE_OUT": np.inf, "LAMBERT_GOAL": np.inf,
            "FLYBY_BODY": np.inf, "et_launch": np.inf,
            "et_flyby": np.inf, "et_goal": np.inf,
        })

    current_launch_range = list(launch_range)

    for i in range(3):
        min_dv_total = np.inf
        lower_bound = np.array([0, 0.5 * YEAR, 0.5 * YEAR]) / YEAR
        upper_bound = np.array([current_launch_range[1] - current_launch_range[0],
                                3 * YEAR, 3 * YEAR]) / YEAR

        min_current_body_id = None
        min_launch_date = -1
        min_arrival_date = -1
        min_arrival_body_id = ''
        min_goal_date = -1

        for n_1 in range(1, N_RES + 1):
            for n_2 in range(1, 2):  # 1:1 in MATLAB = just n_2=1
                for n_3 in range(1, 2):  # 1:1 in MATLAB = just n_3=1
                    for flyby_index in range(len(FLYBY_IDS)):
                        # Skip direct transfer for the first leg
                        if i == 0 and flyby_index == 2:
                            continue

                        current_body_id = bodies_strs[i]

                        # Skip if flyby body is the same as current body
                        if FLYBY_IDS[flyby_index] == current_body_id:
                            continue

                        input_guess = upper_bound - DELTA_GUESS * np.array([n_1, n_2, n_3])

                        # Clip guess to bounds
                        input_guess = np.clip(input_guess, lower_bound, upper_bound)

                        bounds = list(zip(lower_bound, upper_bound))

                        # Define objective for this combination
                        def objective(x, _clr=list(current_launch_range),
                                      _cbi=bodies_strs[i],
                                      _fbi=FLYBY_IDS[flyby_index],
                                      _gbi=bodies_strs[i + 1],
                                      _m1=M[i][0], _m2=M[i][1]):
                            return score_paths_greedy(x, _clr, _cbi, _fbi, _gbi, _m1, _m2)

                        result = minimize(
                            objective,
                            input_guess,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'ftol': 1e-10, 'gtol': 1e-10}
                        )

                        optimized_time_vector = result.x
                        delta_v_score = result.fun

                        launch_date = optimized_time_vector[0] * YEAR + current_launch_range[0]
                        arrival_date = optimized_time_vector[1] * YEAR + launch_date
                        goal_date = optimized_time_vector[2] * YEAR + arrival_date

                        if delta_v_score < min_dv_total:
                            min_dv_total = delta_v_score
                            min_current_body_id = current_body_id
                            min_launch_date = launch_date
                            min_arrival_date = arrival_date
                            min_arrival_body_id = FLYBY_IDS[flyby_index]
                            min_goal_date = goal_date

        # Update launch range for next leg
        current_launch_range[0] = goal_date + 3 * MONTH
        current_launch_range[1] = goal_date + 6 * MONTH

        # Compute final delta-v values for the best combination
        (dv_launch, dv_arrive, dv_goal, dv_total,
         lambert_launch, lambert_arrive_in, lambert_arrive_out, lambert_goal) = \
            compute_path_deltav_greedy(
                min_current_body_id, min_arrival_body_id,
                bodies_strs[i + 1], min_launch_date,
                min_arrival_date, min_goal_date,
                M[i][0], M[i][1])

        optimized_output[i] = {
            "dv_launch": dv_launch,
            "dv_arrive": dv_arrive,
            "dv_goal": dv_goal,
            "dv_total": dv_total,
            "LAMBERT_LAUNCH": lambert_launch,
            "LAMBERT_ARRIVE_IN": lambert_arrive_in,
            "LAMBERT_ARRIVE_OUT": lambert_arrive_out,
            "LAMBERT_GOAL": lambert_goal,
            "FLYBY_BODY": min_arrival_body_id,
            "et_launch": min_launch_date,
            "et_flyby": min_arrival_date,
            "et_goal": min_goal_date,
        }

    return optimized_output


###############################################################################
# DATA GENERATION
###############################################################################

def generate_greedy_optimized_data(asteroid_list, M, launch_utc_min,
                                   launch_utc_max, save_filename):
    """Compute greedy-optimized trajectories for all asteroid triple permutations.

    Parameters
    ----------
    asteroid_list : list of dict
        Each dict has 'ID' (int) and 'NAME' (str) keys.
    M : array-like, shape (3, 2)
        Lambert revolution parameters for each leg.
    launch_utc_min : str
        Earliest launch date as a UTC string (e.g. 'Jan 1 12:00:00 UTC 2027').
    launch_utc_max : str
        Latest launch date as a UTC string.
    save_filename : str
        Base filename for the output pickle file (saved in ./greedy_asteroid_paths/).

    Returns
    -------
    asteroid_optimized_data : dict
        Keyed by (i, j, k) tuples; values are lists of 3 dicts from
        optimize_greedy_times.
    """
    NUM_ASTEROIDS = len(asteroid_list)
    TOTAL_OPERATIONS = NUM_ASTEROIDS ** 3

    elapsed_times = np.zeros(TOTAL_OPERATIONS)

    # Use a dict keyed by (i, j, k) for the 4D structure
    asteroid_optimized_data = {}

    et_launch_min = spiceypy.str2et(launch_utc_min)
    et_launch_max = spiceypy.str2et(launch_utc_max)

    launch_dates = [et_launch_min, et_launch_max]

    pbar = tqdm(total=TOTAL_OPERATIONS, desc=save_filename)

    for i in range(NUM_ASTEROIDS):
        for j in range(NUM_ASTEROIDS):
            for k in range(NUM_ASTEROIDS):
                current_operation = (k + j * NUM_ASTEROIDS
                                     + i * NUM_ASTEROIDS ** 2)
                t_start = time.time()

                a_id_1 = asteroid_list[i]['ID']
                a_id_2 = asteroid_list[j]['ID']
                a_id_3 = asteroid_list[k]['ID']

                # Skip if any two asteroids are the same
                if a_id_1 == a_id_2 or a_id_2 == a_id_3 or a_id_3 == a_id_1:
                    elapsed_times[current_operation] = time.time() - t_start
                    pbar.update(1)
                    continue

                a_id_1_str = str(int(a_id_1))
                a_id_2_str = str(int(a_id_2))
                a_id_3_str = str(int(a_id_3))

                result = optimize_greedy_times(
                    a_id_1_str, a_id_2_str, a_id_3_str,
                    launch_dates, M)

                asteroid_optimized_data[(i, j, k)] = result

                elapsed_times[current_operation] = time.time() - t_start
                elapsed_sum = np.sum(elapsed_times[:current_operation + 1])
                eta = ((elapsed_sum * TOTAL_OPERATIONS
                        / (current_operation + 1)) - elapsed_sum)

                pbar.set_postfix_str(
                    f'Op {current_operation + 1}/{TOTAL_OPERATIONS} '
                    f'(ETA: {eta / 60:.2f} min)')
                pbar.update(1)

    pbar.close()

    # Save to pickle file
    os.makedirs('./greedy_asteroid_paths', exist_ok=True)
    save_path = f'./greedy_asteroid_paths/{save_filename}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(asteroid_optimized_data, f)

    return asteroid_optimized_data


###############################################################################
# ANIMATION
###############################################################################

def greedy_flightpath_animation(path_defined_vector, asteroid_list,
                                a_index_1, a_index_2, a_index_3,
                                t_duration, output_video_name):
    """Create a 3D animated video of the greedy three-leg trajectory.

    Parameters
    ----------
    path_defined_vector : list of 3 dicts
        Each dict contains et_launch, et_flyby, et_goal, LAMBERT_LAUNCH,
        LAMBERT_ARRIVE_IN, LAMBERT_ARRIVE_OUT, LAMBERT_GOAL.
    asteroid_list : list of dict
        Each dict has 'ID' (int) and 'NAME' (str) keys.
    a_index_1, a_index_2, a_index_3 : int
        Indices into asteroid_list for the three asteroids.
    t_duration : float
        Desired video duration in seconds.
    output_video_name : str
        Output filename (e.g. "GREEDY_BEST_PATH.mp4").
    """
    a_id_1 = str(int(asteroid_list[a_index_1]['ID']))
    a_id_2 = str(int(asteroid_list[a_index_2]['ID']))
    a_id_3 = str(int(asteroid_list[a_index_3]['ID']))

    a_name_1 = asteroid_list[a_index_1]['NAME']
    a_name_2 = asteroid_list[a_index_2]['NAME']
    a_name_3 = asteroid_list[a_index_3]['NAME']

    # Mission time span for orbit backdrop
    mission_time = np.arange(path_defined_vector[0]['et_launch'],
                             path_defined_vector[2]['et_goal'],
                             1 * WEEK)

    # Time differences for each sub-leg
    t_diff_1_1 = (path_defined_vector[0]['et_flyby']
                  - path_defined_vector[0]['et_launch'])
    t_diff_1_2 = (path_defined_vector[0]['et_goal']
                  - path_defined_vector[0]['et_flyby'])

    t_diff_2_1 = (path_defined_vector[1]['et_flyby']
                  - path_defined_vector[1]['et_launch'])
    t_diff_2_2 = (path_defined_vector[1]['et_goal']
                  - path_defined_vector[1]['et_flyby'])

    t_diff_3_1 = (path_defined_vector[2]['et_flyby']
                  - path_defined_vector[2]['et_launch'])
    t_diff_3_2 = (path_defined_vector[2]['et_goal']
                  - path_defined_vector[2]['et_flyby'])

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]

    # Get departure states
    earth_launch_state, _ = spiceypy.spkezr(
        '399', path_defined_vector[0]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')
    a1_state, _ = spiceypy.spkezr(
        a_id_1, path_defined_vector[1]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')
    a2_state, _ = spiceypy.spkezr(
        a_id_2, path_defined_vector[2]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')

    # --- Leg 1 sub-leg 1 ---
    x_0 = np.concatenate([earth_launch_state[0:3],
                          path_defined_vector[0]['LAMBERT_LAUNCH']])
    X_1_1, T_1_1 = two_body_sim(t_diff_1_1, x_0, myu_sun)

    # --- Leg 1 sub-leg 2 ---
    if path_defined_vector[0]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_1_1[-1, 0:3],
                              path_defined_vector[0]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_1_1[-1, 0:6]
    X_1_2, T_1_2 = two_body_sim(t_diff_1_2, x_0, myu_sun)

    # --- Leg 2 sub-leg 1 ---
    x_0 = np.concatenate([a1_state[0:3],
                          path_defined_vector[1]['LAMBERT_LAUNCH']])
    X_2_1, T_2_1 = two_body_sim(t_diff_2_1, x_0, myu_sun)

    # --- Leg 2 sub-leg 2 ---
    if path_defined_vector[1]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_2_1[-1, 0:3],
                              path_defined_vector[1]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_2_1[-1, 0:6]
    X_2_2, T_2_2 = two_body_sim(t_diff_2_2, x_0, myu_sun)

    # --- Leg 3 sub-leg 1 ---
    x_0 = np.concatenate([a2_state[0:3],
                          path_defined_vector[2]['LAMBERT_LAUNCH']])
    X_3_1, T_3_1 = two_body_sim(t_diff_3_1, x_0, myu_sun)

    # --- Leg 3 sub-leg 2 ---
    if path_defined_vector[2]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_3_1[-1, 0:3],
                              path_defined_vector[2]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_3_1[-1, 0:6]
    X_3_2, T_3_2 = two_body_sim(t_diff_3_2, x_0, myu_sun)

    # --- Compute total frame count and FPS ---
    N = (2 * (len(T_1_1) + len(T_1_2))
         + (len(T_2_1) + len(T_2_2))
         + (len(T_3_1) + len(T_3_2)))
    FPS = max(1, N / t_duration)

    # Pre-compute full-mission orbit positions for backdrop
    earth_orbit = np.array([spiceypy.spkezr('399', t, 'ECLIPJ2000', 'NONE', '10')[0]
                            for t in mission_time])
    mars_orbit = np.array([spiceypy.spkezr('4', t, 'ECLIPJ2000', 'NONE', '10')[0]
                           for t in mission_time])
    a1_orbit = np.array([spiceypy.spkezr(a_id_1, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])
    a2_orbit = np.array([spiceypy.spkezr(a_id_2, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])
    a3_orbit = np.array([spiceypy.spkezr(a_id_3, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])

    fig = plt.figure(figsize=(16, 10))
    writer = FFMpegWriter(fps=FPS)

    with writer.saving(fig, output_video_name, dpi=100):
        # Leg 1 sub-leg 1
        _animate_section(fig, writer, mission_time, X_1_1, T_1_1,
                         path_defined_vector[0]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 1 sub-leg 2
        _animate_section(fig, writer, mission_time, X_1_2, T_1_2,
                         path_defined_vector[0]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Stay at asteroid 1
        _animate_stay(fig, writer, mission_time, 1,
                      len(T_1_1) + len(T_1_2),
                      path_defined_vector[0]['et_goal'],
                      path_defined_vector[1]['et_launch'],
                      a_id_1, a_id_2, a_id_3,
                      a_name_1, a_name_2, a_name_3,
                      earth_orbit, a1_orbit, a2_orbit, a3_orbit)
        # Leg 2 sub-leg 1
        _animate_section(fig, writer, mission_time, X_2_1, T_2_1,
                         path_defined_vector[1]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 2 sub-leg 2
        _animate_section(fig, writer, mission_time, X_2_2, T_2_2,
                         path_defined_vector[1]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Stay at asteroid 2
        _animate_stay(fig, writer, mission_time, 2,
                      len(T_2_1) + len(T_2_2),
                      path_defined_vector[1]['et_goal'],
                      path_defined_vector[2]['et_launch'],
                      a_id_1, a_id_2, a_id_3,
                      a_name_1, a_name_2, a_name_3,
                      earth_orbit, a1_orbit, a2_orbit, a3_orbit)
        # Leg 3 sub-leg 1
        _animate_section(fig, writer, mission_time, X_3_1, T_3_1,
                         path_defined_vector[2]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 3 sub-leg 2
        _animate_section(fig, writer, mission_time, X_3_2, T_3_2,
                         path_defined_vector[2]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)

    plt.close(fig)


def _animate_section(fig, writer, mission_time, X_i, T_i, et_launch,
                     a_id_1, a_id_2, a_id_3,
                     a_name_1, a_name_2, a_name_3,
                     earth_orbit, mars_orbit,
                     a1_orbit, a2_orbit, a3_orbit):
    """Animate one transfer sub-leg."""
    for i in range(len(T_i)):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True, which='both')

        # Orbit backdrop
        ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
                label='Earth', color='cyan')
        ax.plot(mars_orbit[:, 0], mars_orbit[:, 1], mars_orbit[:, 2],
                label='Mars', color='magenta')
        ax.plot(a1_orbit[:, 0], a1_orbit[:, 1], a1_orbit[:, 2],
                label=a_name_1, color='red')
        ax.plot(a2_orbit[:, 0], a2_orbit[:, 1], a2_orbit[:, 2],
                label=a_name_2, color='green')
        ax.plot(a3_orbit[:, 0], a3_orbit[:, 1], a3_orbit[:, 2],
                label=a_name_3, color='blue')

        # Current positions
        t_now = T_i[i] + et_launch
        for body_id, color in [('399', 'cyan'), ('4', 'magenta')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)
        for body_id, color in [(a_id_1, 'red'), (a_id_2, 'green'), (a_id_3, 'blue')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)

        # Spacecraft position
        ax.scatter([X_i[i, 0]], [X_i[i, 1]], [X_i[i, 2]], c='white', s=30)
        # Full trajectory
        ax.plot(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                label='Trajectory', color='white')

        ax.legend()
        ax.set_title(spiceypy.et2utc(t_now, 'C', 6))
        fig.set_facecolor('black')
        ax.set_facecolor('black')

        writer.grab_frame()


def _animate_stay(fig, writer, mission_time, asteroid_stay_number, n_time,
                  et_start, et_stop,
                  a_id_1, a_id_2, a_id_3,
                  a_name_1, a_name_2, a_name_3,
                  earth_orbit, a1_orbit, a2_orbit, a3_orbit):
    """Animate the stay period at an asteroid."""
    t_range = np.linspace(et_start, et_stop, n_time)

    # Which asteroid are we staying at?
    stay_id = a_id_1 if asteroid_stay_number == 1 else a_id_2
    stay_positions = np.array([
        spiceypy.spkezr(stay_id, t, 'ECLIPJ2000', 'NONE', '10')[0]
        for t in t_range])

    for i in range(n_time):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True, which='both')

        # Orbit backdrop
        ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
                label='Earth', color='cyan')
        ax.plot(a1_orbit[:, 0], a1_orbit[:, 1], a1_orbit[:, 2],
                label=a_name_1, color='red')
        ax.plot(a2_orbit[:, 0], a2_orbit[:, 1], a2_orbit[:, 2],
                label=a_name_2, color='green')
        ax.plot(a3_orbit[:, 0], a3_orbit[:, 1], a3_orbit[:, 2],
                label=a_name_3, color='blue')

        # Current positions
        t_now = t_range[i]
        st, _ = spiceypy.spkezr('399', t_now, 'ECLIPJ2000', 'NONE', '10')
        ax.scatter([st[0]], [st[1]], [st[2]], c='cyan', s=20)
        for body_id, color in [(a_id_1, 'red'), (a_id_2, 'green'), (a_id_3, 'blue')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)

        # Trajectory (stay path)
        ax.plot(stay_positions[:, 0], stay_positions[:, 1], stay_positions[:, 2],
                label='Trajectory', color='white')

        ax.legend()
        ax.set_title(spiceypy.et2utc(t_now, 'C', 6))
        fig.set_facecolor('black')
        ax.set_facecolor('black')

        writer.grab_frame()


###############################################################################
# ANALYSIS
###############################################################################

def find_best_path_greedy(asteroid_optimized_data, asteroid_list):
    """Find the top 6 paths with lowest total delta-v.

    Parameters
    ----------
    asteroid_optimized_data : dict
        Keyed by (i, j, k) tuples; values are lists of 3 dicts.
    asteroid_list : list of dict
        Each dict has 'ID' and 'NAME' keys.

    Returns
    -------
    i_mins, j_mins, k_mins : lists of int
        Indices of the top 6 paths.
    """
    NUM_ASTEROIDS = len(asteroid_list)

    i_mins = [-1] * 6
    j_mins = [-1] * 6
    k_mins = [-1] * 6

    TOTAL_DV_MAP = np.ones((NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS))

    for n in range(6):
        i_min_current = -1
        j_min_current = -1
        k_min_current = -1
        min_delta_v = np.inf

        for i in range(NUM_ASTEROIDS):
            for j in range(NUM_ASTEROIDS):
                for k in range(NUM_ASTEROIDS):
                    if i == j or j == k or k == i:
                        continue

                    if (i, j, k) not in asteroid_optimized_data:
                        continue

                    data = asteroid_optimized_data[(i, j, k)]

                    # total_dv = leg1.dv_goal + leg2.dv_total + leg3.dv_total
                    total_dv = data[0]['dv_goal']
                    for l in range(1, 3):
                        total_dv += data[l]['dv_total']

                    TOTAL_DV_MAP[i, j, k] = total_dv

                    if min_delta_v > total_dv:
                        avoid = False
                        for p in range(n):
                            if (i_mins[p] == i and j_mins[p] == j
                                    and k_mins[p] == k):
                                avoid = True
                        if avoid:
                            continue

                        min_delta_v = total_dv
                        i_min_current = i
                        j_min_current = j
                        k_min_current = k

        i_mins[n] = i_min_current
        j_mins[n] = j_min_current
        k_mins[n] = k_min_current

        if i_min_current < 0 or j_min_current < 0 or k_min_current < 0:
            print(f'Number {n + 1} | No valid path found')
            continue

        data = asteroid_optimized_data[(i_mins[n], j_mins[n], k_mins[n])]
        dv_launch = data[0]['dv_launch']

        print(f'Number {n + 1} | ASTEROID TRANSFERS dv: {min_delta_v:.2f} km/s '
              f'(EARTH LAUNCH DV {dv_launch:.2f}) | '
              f'{asteroid_list[i_mins[n]]["NAME"]} (INDEX {i_mins[n]}) | '
              f'{asteroid_list[j_mins[n]]["NAME"]} (INDEX {j_mins[n]}) | '
              f'{asteroid_list[k_mins[n]]["NAME"]} (INDEX {k_mins[n]})')

        for leg_idx in range(3):
            leg = data[leg_idx]
            launch_utc = spiceypy.et2utc(abs(leg['et_launch']), 'C', 1)
            flyby_utc = spiceypy.et2utc(abs(leg['et_flyby']), 'C', 1)
            goal_utc = spiceypy.et2utc(leg['et_goal'], 'C', 1)

            print(f'  LEG {leg_idx + 1}: '
                  f'LAUNCH AT {launch_utc} (dv {leg["dv_launch"]:.2f}) | '
                  f'FLYBY {leg["FLYBY_BODY"]} AT {flyby_utc} '
                  f'(dv {leg["dv_arrive"]:.2f}) | '
                  f'ARRIVE AT {goal_utc} (dv {leg["dv_goal"]:.2f})')
        print()

    return i_mins, j_mins, k_mins


def find_all_paths_greedy_main():
    """Load all greedy optimized data files and find the best path in each."""
    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    NUM_ASTEROIDS = len(asteroid_list)

    pkl_files = sorted(glob.glob(os.path.join("greedy_asteroid_paths", "*.pkl")))

    TOTAL_DV_MAP = np.ones((NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS))

    DATA = []

    for p, pkl_path in enumerate(pkl_files):
        filename = os.path.basename(pkl_path)

        with open(pkl_path, 'rb') as f:
            asteroid_optimized_data = pickle.load(f)

        i_mins = [-1]
        j_mins = [-1]
        k_mins = [-1]

        for n in range(1):  # MATLAB 1:1 = single iteration
            i_min_current = -1
            j_min_current = -1
            k_min_current = -1
            min_delta_v = np.inf

            for i in range(NUM_ASTEROIDS):
                for j in range(NUM_ASTEROIDS):
                    for k in range(NUM_ASTEROIDS):
                        if i == j or j == k or k == i:
                            continue

                        if (i, j, k) not in asteroid_optimized_data:
                            continue

                        data = asteroid_optimized_data[(i, j, k)]

                        # total_dv = leg1.dv_goal + leg2.dv_total + leg3.dv_total
                        total_dv = data[0]['dv_goal']
                        for l in range(1, 3):
                            total_dv += data[l]['dv_total']

                        TOTAL_DV_MAP[i, j, k] = total_dv

                        if min_delta_v > total_dv:
                            avoid = False
                            for w in range(n):
                                if (i_mins[w] == i and j_mins[w] == j
                                        and k_mins[w] == k):
                                    avoid = True
                                    break
                            if avoid:
                                continue

                            min_delta_v = total_dv
                            i_min_current = i
                            j_min_current = j
                            k_min_current = k

        i_mins[0] = i_min_current
        j_mins[0] = j_min_current
        k_mins[0] = k_min_current

        if i_min_current < 0 or j_min_current < 0 or k_min_current < 0:
            continue

        data = asteroid_optimized_data[(i_mins[0], j_mins[0], k_mins[0])]

        dv_launch = data[0]['dv_launch']

        DATA.append({
            'dv_asteroids': min_delta_v,
            'dv_launch': dv_launch,
            'asteroid 1': asteroid_list[i_mins[0]]['NAME'],
            'asteroid 2': asteroid_list[j_mins[0]]['NAME'],
            'asteroid 3': asteroid_list[k_mins[0]]['NAME'],
            'filename': filename,
        })

        # Print detailed info
        print(f'FILE: {filename} | ASTEROID TRANSFERS dv: {min_delta_v:.2f} km/s '
              f'(EARTH LAUNCH DV {dv_launch:.2f}) | '
              f'{asteroid_list[i_mins[0]]["NAME"]} (INDEX {i_mins[0]}) | '
              f'{asteroid_list[j_mins[0]]["NAME"]} (INDEX {j_mins[0]}) | '
              f'{asteroid_list[k_mins[0]]["NAME"]} (INDEX {k_mins[0]})')

        for leg_idx in range(3):
            leg = data[leg_idx]
            launch_utc = spiceypy.et2utc(
                abs(leg['et_launch']), 'C', 1)
            flyby_utc = spiceypy.et2utc(
                abs(leg['et_flyby']), 'C', 1)
            goal_utc = spiceypy.et2utc(leg['et_goal'], 'C', 1)

            print(f'  LEG {leg_idx + 1}: '
                  f'LAUNCH AT {launch_utc} (dv {leg["dv_launch"]:.2f}) | '
                  f'FLYBY {leg["FLYBY_BODY"]} AT {flyby_utc} (dv {leg["dv_arrive"]:.2f}) | '
                  f'ARRIVE AT {goal_utc} (dv {leg["dv_goal"]:.2f})')
        print()

    # Write CSV
    df = pd.DataFrame(DATA)
    df.to_csv('asteroid_optimized_data_table.csv', index=False)
    print(f'Wrote asteroid_optimized_data_table.csv ({len(DATA)} rows)')


def plot_best_path(asteroid_list, asteroid_optimized_data):
    """Find top 6 greedy paths (excluding launch dv) and animate the best one."""
    NUM_ASTEROIDS = len(asteroid_list)

    i_mins = [-1] * 6
    j_mins = [-1] * 6
    k_mins = [-1] * 6

    TOTAL_DV_MAP = np.ones((NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS))

    for n in range(6):
        i_min_current = -1
        j_min_current = -1
        k_min_current = -1
        min_delta_v = np.inf

        for i in range(NUM_ASTEROIDS):
            for j in range(NUM_ASTEROIDS):
                for k in range(NUM_ASTEROIDS):
                    if i == j or j == k or k == i:
                        continue

                    total_dv = 0.0
                    for l in range(1, 3):
                        total_dv += asteroid_optimized_data[i][j][k][l]['dv_total']
                    TOTAL_DV_MAP[i, j, k] = total_dv

                    if min_delta_v > total_dv:
                        avoid = False
                        for p in range(n):
                            if i_mins[p] == i and j_mins[p] == j and k_mins[p] == k:
                                avoid = True
                                break
                        if avoid:
                            continue

                        min_delta_v = total_dv
                        i_min_current = i
                        j_min_current = j
                        k_min_current = k

        i_mins[n] = i_min_current
        j_mins[n] = j_min_current
        k_mins[n] = k_min_current

    greedy_flightpath_animation(
        asteroid_optimized_data[i_mins[0]][j_mins[0]][k_mins[0]],
        asteroid_list, i_mins[0], j_mins[0], k_mins[0],
        12, "GREEDY_BEST_PATH.mp4")


def plot_best_path_main():
    """Main entry point for plot_best_path."""
    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "path/to/kernels")
    with open("greedy_asteroid_paths/greedy_data.pkl", "rb") as f:
        asteroid_optimized_data = pickle.load(f)
    plot_best_path(asteroid_list, asteroid_optimized_data)


def plot_dv_range(asteroid_list):
    """Surface plot of delta-v as a function of two time parameters."""
    LAUNCH_RANGE = [
        spiceypy.str2et('Jan 1 12:00:00 UTC 2028'),
        spiceypy.str2et('Dec 31 12:00:00 UTC 2028'),
    ]

    LAUNCH_BODY = str(int(asteroid_list[1]['ID']))
    LANDING_BODY = "-1"
    GOAL_BODY = str(int(asteroid_list[2]['ID']))
    M_1 = -1

    t_1 = np.linspace(0, 0.5, 120)
    t_2 = np.linspace(0, 3, 120)

    score_ij = np.zeros((len(t_1), len(t_2)))

    for i in range(len(t_1)):
        for j in range(len(t_2)):
            score_ij[i, j] = score_paths_greedy(
                np.array([t_1[i], 0, t_2[j]]),
                LAUNCH_RANGE, LAUNCH_BODY, LANDING_BODY, GOAL_BODY, M_1, -1)

    T_1, T_2 = np.meshgrid(t_1, t_2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_1, T_2, score_ij.T, cmap='viridis')
    ax.set_xlabel('t_1')
    ax.set_ylabel('t_2')
    ax.set_zlabel('Delta-V (km/s)')
    plt.show()


def plot_dv_range_main():
    """Main entry point for plot_dv_range."""
    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "path/to/kernels")
    plot_dv_range(asteroid_list)


###############################################################################
# RUNNER SCRIPTS
###############################################################################

def greedy_selector_main():
    """Main entry point for the greedy selector script."""
    try:
        asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    except Exception as e:
        print(f'An error occurred: {e}')
        raise RuntimeError(
            "Most likely you need to add NOTABLE_ASTEROID_BSPs to your path"
            " and you must be in the main github folder.")

    M_VALUES = [0.1, -0.1, 1, -1]
    M_length = len(M_VALUES)

    MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027'
    MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035'

    # Iterate over all combinations of 6 indices into M_VALUES
    # This replicates the MATLAB odometer-style iteration
    iter_vec = [0, 0, 0, 0, 0, 0]  # 0-indexed

    while iter_vec[5] < M_length:
        # Build M matrix: reshape 6 values into (3, 2)
        m_vals = [M_VALUES[idx] for idx in iter_vec]
        M = np.array(m_vals).reshape(3, 2)

        save_filename = "greedy_test"
        for idx in iter_vec:
            save_filename += f"_{M_VALUES[idx]:.1f}"

        generate_greedy_optimized_data(
            asteroid_list, M,
            MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC,
            save_filename)

        # Increment odometer (carry from left to right)
        iter_vec[0] += 1
        for j in range(len(iter_vec) - 1):
            if iter_vec[j] >= M_length:
                iter_vec[j + 1] += 1
                iter_vec[j] = 0
        if iter_vec[-1] >= M_length:
            break


if __name__ == '__main__':
    greedy_selector_main()
