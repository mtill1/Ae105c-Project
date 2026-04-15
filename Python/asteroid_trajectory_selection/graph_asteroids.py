"""Animated video of all asteroid orbits over time.

Translates GRAPH_ASTEROIDS.m to Python using matplotlib + FFMpegWriter.
"""

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from .constants import WEEK


def graph_asteroids(asteroid_list, t_duration, fps, start_date, end_date,
                    output_video_name):
    """Create an animated MP4 showing all asteroid orbits over time.

    Parameters
    ----------
    asteroid_list : list of dict
        Each dict has 'ID' and 'NAME' keys.
    t_duration : float
        Desired video duration in seconds.
    fps : int
        Frames per second.
    start_date : str
        UTC start date string (e.g. 'Jan 1 12:00:00 UTC 2027').
    end_date : str
        UTC end date string.
    output_video_name : str
        Output filename (e.g. "asteroids.mp4").
    """
    et0 = spiceypy.str2et(start_date)
    etf = spiceypy.str2et(end_date)

    t_range = np.arange(et0, etf, 4 * WEEK)

    N = int(fps * t_duration)
    K = len(t_range) / N

    num_asteroids = len(asteroid_list)

    # Pre-compute all asteroid positions: shape (num_asteroids, len(t_range), 3)
    asteroid_positions = np.zeros((num_asteroids, len(t_range), 3))
    for j, ast in enumerate(asteroid_list):
        a_id = str(int(ast['ID']))
        for ti, et in enumerate(t_range):
            state, _ = spiceypy.spkezr(a_id, float(et), 'ECLIPJ2000', 'NONE', '0')
            asteroid_positions[j, ti, :] = state[0:3]

    colors = plt.cm.hsv(np.linspace(0, 1, num_asteroids))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_video_name, dpi=150):
        for i in tqdm(range(N), desc='Encoding frames'):
            ax.clear()
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

            idx = min(int(np.ceil((i + 1) * K)) - 1, len(t_range) - 1)

            position_list = np.zeros((num_asteroids, 3))

            for j, ast in enumerate(asteroid_list):
                ax.plot(asteroid_positions[j, :, 0],
                        asteroid_positions[j, :, 1],
                        asteroid_positions[j, :, 2],
                        color=colors[j], label=ast['NAME'])

                position_list[j, :] = asteroid_positions[j, idx, :]

            ax.scatter(position_list[:, 0], position_list[:, 1],
                       position_list[:, 2], s=60, c=colors)

            title_date = spiceypy.et2utc(float(t_range[idx]), 'C', 1)
            ax.set_title(f'Asteroid Trajectories at {title_date}')
            ax.set_xlabel('X Position (km)')
            ax.set_ylabel('Y Position (km)')
            ax.set_zlabel('Z Position (km)')
            ax.legend(fontsize='small')
            ax.view_init(elev=30, azim=-60)  # 3D view (view(3) equivalent)

            writer.grab_frame()

    plt.close(fig)
    print(f"Video saved: {output_video_name}")
