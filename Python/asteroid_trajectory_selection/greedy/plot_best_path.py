"""Find top 6 greedy paths (excluding launch dv) and animate the best one."""

import numpy as np
import spiceypy

from .greedy_flightpath_animation import greedy_flightpath_animation


def plot_best_path(asteroid_list, asteroid_optimized_data):
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


if __name__ == '__main__':
    import pickle
    from ..load_kernels import load_kernels

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "path/to/kernels")
    with open("greedy_asteroid_paths/greedy_data.pkl", "rb") as f:
        asteroid_optimized_data = pickle.load(f)
    plot_best_path(asteroid_list, asteroid_optimized_data)
