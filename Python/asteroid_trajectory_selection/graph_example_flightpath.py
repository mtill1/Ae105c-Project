"""One-shot script: animate a specific example flightpath.

Translates graph_example_flightpath.m to Python.
"""

import pickle

from .load_kernels import load_kernels
from .flightpath_animation import flightpath_animation


def main():
    """Animate the trajectory for asteroid indices (7, 4, 9) (zero-based)."""
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    with open('asteroid_optimized_data.pkl', 'rb') as f:
        asteroid_optimized_data = pickle.load(f)

    # MATLAB used 1-based indices (8, 5, 10) -> Python 0-based (7, 4, 9)
    pdv = asteroid_optimized_data[7][4][9]
    flightpath_animation(pdv, asteroid_list, 7, 4, 9, 8, "TEST.mp4")


if __name__ == '__main__':
    main()
