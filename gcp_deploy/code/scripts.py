"""Runner scripts -- entry points for all major workflows."""

import sys
import pickle
import numpy as np
import spiceypy
import matplotlib.pyplot as plt

from core import load_kernels, get_state, MU_SUN
from optimization import (generate_optimized_data, generate_mars_transfer_optimized,
                         two_level_optimize, beam_search)
from visualization import flightpath_animation, graph_asteroids


# ============================================================================
# ASTEROID SELECTOR
# ============================================================================

def run_asteroid_selector():
    """Load SPICE kernels and compute optimal asteroid trajectories."""
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


# ============================================================================
# MARS TRANSFER SELECTOR
# ============================================================================

def run_mars_transfer_selector():
    """Load SPICE kernels and compute Mars-flyby transfer trajectories."""
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

    MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2029'
    MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035'

    asteroid_optimized_data = generate_mars_transfer_optimized(
        asteroid_list, 0, 0, 0, 1,
        MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC
    )

    return asteroid_optimized_data


# ============================================================================
# FIND BEST PATH
# ============================================================================

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


# ============================================================================
# GRAPH ALL BEST PATHS
# ============================================================================

def run_graph_all_best_paths():
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


# ============================================================================
# GRAPH EXAMPLE FLIGHTPATH
# ============================================================================

def run_graph_example_flightpath():
    """Animate the trajectory for asteroid indices (7, 4, 9) (zero-based)."""
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    with open('asteroid_optimized_data.pkl', 'rb') as f:
        asteroid_optimized_data = pickle.load(f)

    # MATLAB used 1-based indices (8, 5, 10) -> Python 0-based (7, 4, 9)
    pdv = asteroid_optimized_data[7][4][9]
    flightpath_animation(pdv, asteroid_list, 7, 4, 9, 8, "TEST.mp4")


# ============================================================================
# GRAPHING NOTABLE ASTEROIDS
# ============================================================================

def run_graphing_notable_asteroids():
    """Load SPICE kernels and create an animated video of notable asteroids."""
    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    graph_asteroids(
        asteroid_list,
        t_duration=8,
        fps=80,
        start_date='Jan 1 12:00:00 UTC 2027',
        end_date='Dec 31 12:00:00 UTC 2031',
        output_video_name='Jan_1_2027-Dec_31_2031-Notable-Asteroids.mp4'
    )


# ============================================================================
# TWO-LEVEL OPTIMIZATION (recommended for large asteroid pools)
# ============================================================================

def run_two_level_optimize(science_csv=None, alpha=1.0):
    """Run two-level optimization: coarse filter all N^3, then fine-tune top 50.

    Parameters
    ----------
    science_csv : str or None — path to asteroid_tradeoff.csv for science scoring
    alpha : float — 1.0 = pure delta-v, 0.7 = 70% dv + 30% science
    """
    import pandas as pd

    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "/Users/rebnoob/Documents/ae105/generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    science_scores = None
    if science_csv:
        df = pd.read_csv(science_csv)
        science_scores = {}
        for _, row in df.iterrows():
            name = str(row['Name_DecRadius']).split('(')[0].strip().upper()
            science_scores[name] = row['Total_WeightedScore']

    return two_level_optimize(
        asteroid_list, 0, 0, 0,
        'Jan 1 12:00:00 UTC 2027', 'Dec 31 12:00:00 UTC 2035',
        top_n=50, science_scores=science_scores, alpha=alpha)


# ============================================================================
# BEAM SEARCH (structured multi-stage path selection)
# ============================================================================

def run_beam_search(beam_width=10, science_csv=None, alpha=1.0):
    """Run beam search with configurable beam width and optional science weighting."""
    import pandas as pd

    bsp_folder = "NOTABLE_ASTEROID_BSPs"
    generic_kernels = "/Users/rebnoob/Documents/ae105/generic_kernels"

    asteroid_list = load_kernels(bsp_folder, generic_kernels)

    science_scores = None
    if science_csv:
        df = pd.read_csv(science_csv)
        science_scores = {}
        for _, row in df.iterrows():
            name = str(row['Name_DecRadius']).split('(')[0].strip().upper()
            science_scores[name] = row['Total_WeightedScore']

    return beam_search(
        asteroid_list,
        'Jan 1 12:00:00 UTC 2027', 'Dec 31 12:00:00 UTC 2035',
        beam_width=beam_width, science_scores=science_scores, alpha=alpha)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    commands = {
        'asteroid_selector': 'Brute-force N^3 optimization (original)',
        'mars_transfer_selector': 'Mars flyby optimization',
        'two_level_optimize': 'Two-level: coarse filter + fine optimize (recommended)',
        'beam_search': 'Beam search: structured multi-stage selection',
        'find_best_path': 'Find top paths from saved results',
        'graph_all_best_paths': 'Animate all best paths',
        'graphing_notable_asteroids': 'Animate notable asteroid orbits',
    }

    print("Available workflows:")
    for name, desc in commands.items():
        print(f"  {name:30s} — {desc}")
    print("\nUsage: from scripts import run_two_level_optimize; run_two_level_optimize()")
