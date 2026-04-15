"""Animate each of the best paths found by find_best_path.

Translates graph_all_best_paths.m to Python.
"""

import pickle
import matplotlib.pyplot as plt

from .load_kernels import load_kernels
from .find_best_path import find_best_path
from .flightpath_animation import flightpath_animation


def main():
    """Load data and create animation videos for each of the top paths."""
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    with open('asteroid_optimized_data.pkl', 'rb') as f:
        asteroid_optimized_data = pickle.load(f)

    i_mins, j_mins, k_mins = find_best_path(asteroid_list,
                                              asteroid_optimized_data)

    for n in range(len(i_mins)):
        ii, jj, kk = i_mins[n], j_mins[n], k_mins[n]
        pdv = asteroid_optimized_data[ii][jj][kk]

        video_name = (f"{asteroid_list[ii]['NAME']}_"
                      f"{asteroid_list[jj]['NAME']}_"
                      f"{asteroid_list[kk]['NAME']}_2D.mp4")

        flightpath_animation(pdv, asteroid_list, ii, jj, kk, 10, video_name)

        plt.close('all')


if __name__ == '__main__':
    main()
