"""Optimize greedy trajectory times using sequential three-leg optimization."""

import numpy as np
from scipy.optimize import minimize

from ..constants import YEAR, MONTH
from .score_paths_greedy import score_paths_greedy
from .compute_path_deltav_greedy import compute_path_deltav_greedy


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
