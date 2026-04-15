"""Asteroid selector script.

Loads SPICE kernels, sets launch-date parameters, and calls
generate_optimized_data to compute optimal trajectories.

Translates asteroid_selector.m to Python.
"""

import sys

from .load_kernels import load_kernels
from .generate_optimized_data import generate_optimized_data


def main():
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    try:
        asteroid_list = load_kernels(bsp_folder, generic_kernels)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Most likely you need to add NOTABLE_ASTEROID_BSPs to your path "
              "and you must be in the main github folder.")
        sys.exit(1)

    MAX_M = 2
    MIN_M = 0

    MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027'
    MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035'

    asteroid_optimized_data = generate_optimized_data(
        asteroid_list, 0, 0, 0,
        MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC
    )

    return asteroid_optimized_data


if __name__ == '__main__':
    main()
