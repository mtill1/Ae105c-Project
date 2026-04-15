"""Iterate over all M_VALUES combinations and run greedy optimization."""

import itertools
import numpy as np

from ..load_kernels import load_kernels
from .generate_greedy_optimized_data import generate_greedy_optimized_data


def main():
    """Main entry point for the greedy selector script."""
    try:
        asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    except Exception as e:
        print(f'An error occurred: {e}')
        raise RuntimeError(
            "Most likely you need to add NOTABLE_ASTEROID_BSPs to your path"
            " and you must be in the main github folder.")

    M_VALUES = [0.1, -0.1, 1, -1]
    M_length = len(M_VALUES)

    MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027'
    MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035'

    # Iterate over all combinations of 6 indices into M_VALUES
    # This replicates the MATLAB odometer-style iteration
    iter_vec = [0, 0, 0, 0, 0, 0]  # 0-indexed

    while iter_vec[5] < M_length:
        # Build M matrix: reshape 6 values into (3, 2)
        m_vals = [M_VALUES[idx] for idx in iter_vec]
        M = np.array(m_vals).reshape(3, 2)

        save_filename = "greedy_test"
        for idx in iter_vec:
            save_filename += f"_{M_VALUES[idx]:.1f}"

        generate_greedy_optimized_data(
            asteroid_list, M,
            MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC,
            save_filename)

        # Increment odometer (carry from left to right)
        iter_vec[0] += 1
        for j in range(len(iter_vec) - 1):
            if iter_vec[j] >= M_length:
                iter_vec[j + 1] += 1
                iter_vec[j] = 0
        if iter_vec[-1] >= M_length:
            break


if __name__ == '__main__':
    main()
