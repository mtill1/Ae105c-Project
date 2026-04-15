"""Load all greedy optimized data files and find the best path in each."""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import spiceypy

from ..load_kernels import load_kernels


def main():
    """Main entry point."""
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


if __name__ == '__main__':
    main()
