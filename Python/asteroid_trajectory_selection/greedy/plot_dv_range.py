"""Surface plot of delta-v as a function of two time parameters."""

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .score_paths_greedy import score_paths_greedy


def plot_dv_range(asteroid_list):
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


if __name__ == '__main__':
    from ..load_kernels import load_kernels
    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "path/to/kernels")
    plot_dv_range(asteroid_list)
