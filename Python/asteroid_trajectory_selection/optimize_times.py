import numpy as np
from scipy.optimize import minimize

from .constants import MINUTE, HOUR, DAY, WEEK, MONTH, YEAR
from .score_paths import score_paths
from .unpack_input import unpack_input
from .compute_path_deltav import compute_path_deltav


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
