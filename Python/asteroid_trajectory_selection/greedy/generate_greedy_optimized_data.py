"""Generate optimized greedy trajectory data for all asteroid permutations."""

import os
import time
import pickle

import numpy as np
import spiceypy
from tqdm import tqdm

from .optimize_greedy_times import optimize_greedy_times


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
