"""Find the top 10 paths by total delta-v from pre-computed optimized data.

Translates find_best_path.m to Python.

Expects `asteroid_list` and `asteroid_optimized_data` to be loaded
before calling `find_best_path`, or run this file as a script
which loads them from pickle files.
"""

import pickle
import numpy as np
import spiceypy

from .load_kernels import load_kernels


def find_best_path(asteroid_list, asteroid_optimized_data):
    """Search through optimized data to find the top 10 paths by delta-v.

    Parameters
    ----------
    asteroid_list : list of dict
        Each dict has 'ID' and 'NAME' keys.
    asteroid_optimized_data : 3-D structure (nested dicts or array)
        asteroid_optimized_data[i][j][k] is a dict with at minimum
        'delta_v_total', 'delta_v_launch', 'et_launch', 'et_arrive_1',
        'et_stay_1', 'et_arrive_2', 'et_stay_2', 'et_arrive_3'.

    Returns
    -------
    i_mins, j_mins, k_mins : list of int
        Zero-based indices of the top 10 paths.
    """
    num_asteroids = len(asteroid_list)

    i_mins = [-1] * 10
    j_mins = [-1] * 10
    k_mins = [-1] * 10

    for n in range(10):
        i_min_current = -1
        j_min_current = -1
        k_min_current = -1
        min_delta_v = np.inf

        for i in range(num_asteroids):
            for j in range(num_asteroids):
                for k in range(num_asteroids):
                    if i == j or j == k or k == i:
                        continue

                    dv_total = asteroid_optimized_data[i][j][k]['delta_v_total']

                    if min_delta_v > dv_total:
                        # Check if this combination was already selected
                        avoid = False
                        for p in range(n):
                            if (i_mins[p] == i and j_mins[p] == j
                                    and k_mins[p] == k):
                                avoid = True
                                break

                        if avoid:
                            continue

                        min_delta_v = dv_total
                        i_min_current = i
                        j_min_current = j
                        k_min_current = k

        i_mins[n] = i_min_current
        j_mins[n] = j_min_current
        k_mins[n] = k_min_current

        # Display the result for this rank
        ii, jj, kk = i_mins[n], j_mins[n], k_mins[n]
        data = asteroid_optimized_data[ii][jj][kk]
        launch_dv = np.linalg.norm(data['delta_v_launch'])

        print(f"Number {n + 1} | Minimum delta_v: {min_delta_v:.2f} km/s "
              f"(LAUNCH DV {launch_dv:.2f}) | "
              f"{asteroid_list[ii]['NAME']} (INDEX {ii}) | "
              f"{asteroid_list[jj]['NAME']} (INDEX {jj}) | "
              f"{asteroid_list[kk]['NAME']} (INDEX {kk})")

        print(f"LAUNCH: {spiceypy.et2utc(data['et_launch'], 'C', 1)}")
        print(f" ARRIVE 1: {spiceypy.et2utc(data['et_arrive_1'], 'C', 1)} | "
              f"STAY 1: {spiceypy.et2utc(data['et_stay_1'], 'C', 1)}")
        print(f" ARRIVE 2: {spiceypy.et2utc(data['et_arrive_2'], 'C', 1)} | "
              f"STAY 2: {spiceypy.et2utc(data['et_stay_2'], 'C', 1)}")
        print(f" ARRIVE 3: {spiceypy.et2utc(data['et_arrive_3'], 'C', 1)}")
        print()

    # Print overall best
    ii, jj, kk = i_mins[0], j_mins[0], k_mins[0]
    best_data = asteroid_optimized_data[ii][jj][kk]
    best_dv = best_data['delta_v_total']
    best_launch_dv = np.linalg.norm(best_data['delta_v_launch'])

    print(f"\nMinimum delta_v: {best_dv:.2f} km/s (LAUNCH DV {best_launch_dv:.2f}) "
          f"found for asteroids "
          f"{asteroid_list[ii]['NAME']} (INDEX {ii}), "
          f"{asteroid_list[jj]['NAME']} (INDEX {jj}), "
          f"{asteroid_list[kk]['NAME']} (INDEX {kk})")

    return i_mins, j_mins, k_mins


if __name__ == '__main__':
    # Load pre-computed data from pickle files
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    with open('asteroid_optimized_data.pkl', 'rb') as f:
        asteroid_optimized_data = pickle.load(f)

    i_mins, j_mins, k_mins = find_best_path(asteroid_list,
                                              asteroid_optimized_data)
