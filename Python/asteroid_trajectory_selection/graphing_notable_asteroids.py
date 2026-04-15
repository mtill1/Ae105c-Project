"""Load kernels and graph all notable asteroids.

Translates graphing_notable_asteroids.m to Python.
"""

from .load_kernels import load_kernels
from .graph_asteroids import graph_asteroids


def main():
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


if __name__ == '__main__':
    main()
