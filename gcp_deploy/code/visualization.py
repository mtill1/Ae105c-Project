"""
visualization.py — Flightpath animation, asteroid orbit graphing, and static trajectory plots.
"""

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from core import two_body_sim, get_state, WEEK, MU_SUN


# =============================================================================
# FLIGHTPATH ANIMATION
# =============================================================================

def _get_positions(body_id, et_array):
    """Get positions at many epochs; return (N, 3) array in km."""
    return np.array([get_state(str(body_id), float(et))[0] for et in et_array])


def _draw_frame(ax, mission_pos, a1_pos, a2_pos, a3_pos, earth_pos,
                earth_curr, a1_curr, a2_curr, a3_curr, sc_pos,
                trajectory, a_name_1, a_name_2, a_name_3, title_str):
    """Draw a single animation frame on *ax*."""
    ax.clear()

    # Orbit paths
    ax.plot(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2],
            color='cyan', label='Earth')
    ax.plot(a1_pos[:, 0], a1_pos[:, 1], a1_pos[:, 2],
            color='red', label=a_name_1)
    ax.plot(a2_pos[:, 0], a2_pos[:, 1], a2_pos[:, 2],
            color='green', label=a_name_2)
    ax.plot(a3_pos[:, 0], a3_pos[:, 1], a3_pos[:, 2],
            color='blue', label=a_name_3)

    # Current positions (scatter)
    ax.scatter(*earth_curr[0:3], color='cyan', s=30)
    ax.scatter(*a1_curr[0:3], color='red', s=30)
    ax.scatter(*a2_curr[0:3], color='green', s=30)
    ax.scatter(*a3_curr[0:3], color='blue', s=30)

    # Spacecraft
    if sc_pos is not None:
        ax.scatter(*sc_pos, color='white', s=30)

    # Trajectory line
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color='white', label='Trajectory')

    ax.legend(fontsize='small')
    ax.view_init(elev=90, azim=-90)  # top-down (view(2) equivalent)
    ax.set_title(title_str)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)


def _animate_section(writer, fig, ax, mission_time, X_i, T_i, et_launch,
                     a_id_1, a_id_2, a_id_3, a_name_1, a_name_2, a_name_3,
                     desc='SECTION'):
    """Render frames for a transfer arc section."""
    # Pre-compute orbit paths over full mission time
    earth_pos = _get_positions('399', mission_time)
    a1_pos = _get_positions(a_id_1, mission_time)
    a2_pos = _get_positions(a_id_2, mission_time)
    a3_pos = _get_positions(a_id_3, mission_time)

    for i in tqdm(range(len(T_i)), desc=desc):
        et_now = T_i[i] + et_launch

        earth_curr = np.concatenate(get_state('399', float(et_now)))
        a1_curr = np.concatenate(get_state(a_id_1, float(et_now)))
        a2_curr = np.concatenate(get_state(a_id_2, float(et_now)))
        a3_curr = np.concatenate(get_state(a_id_3, float(et_now)))

        title_str = spiceypy.et2utc(float(et_now), 'C', 6)

        _draw_frame(ax, None, a1_pos, a2_pos, a3_pos, earth_pos,
                    earth_curr, a1_curr, a2_curr, a3_curr,
                    X_i[i, 0:3], X_i[:, 0:3],
                    a_name_1, a_name_2, a_name_3, title_str)
        writer.grab_frame()


def _animate_stay_paths(writer, fig, ax, mission_time, asteroid_stay_number,
                        n_time, et_start, et_stop,
                        a_id_1, a_id_2, a_id_3,
                        a_name_1, a_name_2, a_name_3, desc='STAY'):
    """Render frames while the spacecraft stays at an asteroid."""
    # Pre-compute orbit paths over full mission time
    earth_pos = _get_positions('399', mission_time)
    a1_pos = _get_positions(a_id_1, mission_time)
    a2_pos = _get_positions(a_id_2, mission_time)
    a3_pos = _get_positions(a_id_3, mission_time)

    t_range = np.linspace(et_start, et_stop, n_time)

    stay_id = a_id_1 if asteroid_stay_number == 1 else a_id_2
    a_stay_pos = _get_positions(stay_id, t_range)

    for i in tqdm(range(n_time), desc=desc):
        et_now = t_range[i]

        earth_curr = np.concatenate(get_state('399', float(et_now)))
        a1_curr = np.concatenate(get_state(a_id_1, float(et_now)))
        a2_curr = np.concatenate(get_state(a_id_2, float(et_now)))
        a3_curr = np.concatenate(get_state(a_id_3, float(et_now)))

        title_str = spiceypy.et2utc(float(et_now), 'C', 6)

        _draw_frame(ax, None, a1_pos, a2_pos, a3_pos, earth_pos,
                    earth_curr, a1_curr, a2_curr, a3_curr,
                    None, a_stay_pos,
                    a_name_1, a_name_2, a_name_3, title_str)
        writer.grab_frame()


def flightpath_animation(path_defined_vector, asteroid_list,
                         a_index_1, a_index_2, a_index_3,
                         t_duration, output_video_name):
    """Create an animated MP4 of the three-asteroid trajectory.

    Parameters
    ----------
    path_defined_vector : dict
        Keys: et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2,
              et_arrive_3, delta_v_launch, delta_v_A1_leave, delta_v_A2_leave.
    asteroid_list : list of dict
        Each dict has 'ID' and 'NAME' keys.
    a_index_1, a_index_2, a_index_3 : int
        Zero-based indices into asteroid_list.
    t_duration : float
        Desired video duration in seconds.
    output_video_name : str
        Output filename (e.g. "trajectory.mp4").
    """
    a_id_1 = str(int(asteroid_list[a_index_1]['ID']))
    a_id_2 = str(int(asteroid_list[a_index_2]['ID']))
    a_id_3 = str(int(asteroid_list[a_index_3]['ID']))

    a_name_1 = asteroid_list[a_index_1]['NAME']
    a_name_2 = asteroid_list[a_index_2]['NAME']
    a_name_3 = asteroid_list[a_index_3]['NAME']

    pdv = path_defined_vector  # shorthand

    # Mission time array (for orbit background lines)
    mission_time = np.arange(pdv['et_launch'], pdv['et_arrive_3'], 2 * WEEK)

    # Time differences for each leg
    t_diff_1 = pdv['et_arrive_1'] - pdv['et_launch']
    t_diff_2 = pdv['et_arrive_2'] - pdv['et_stay_1']
    t_diff_3 = pdv['et_arrive_3'] - pdv['et_stay_2']

    # Sun gravitational parameter
    mu_sun = MU_SUN

    # --- Leg 1: Earth to Asteroid 1 ---
    r_earth, v_earth = get_state('399', pdv['et_launch'])
    x_0 = np.concatenate([r_earth, v_earth + pdv['delta_v_launch']])
    X_1, T_1 = two_body_sim(t_diff_1, x_0, mu_sun)

    # --- Leg 2: Asteroid 1 to Asteroid 2 ---
    r_a1, v_a1 = get_state(a_id_1, pdv['et_stay_1'])
    x_0 = np.concatenate([r_a1, v_a1 + pdv['delta_v_A1_leave']])
    X_2, T_2 = two_body_sim(t_diff_2, x_0, mu_sun)

    # --- Leg 3: Asteroid 2 to Asteroid 3 ---
    r_a2, v_a2 = get_state(a_id_2, pdv['et_stay_2'])
    x_0 = np.concatenate([r_a2, v_a2 + pdv['delta_v_A2_leave']])
    X_3, T_3 = two_body_sim(t_diff_3, x_0, mu_sun)

    # Total frame count and FPS
    N = 2 * len(T_1) + 2 * len(T_2) + len(T_3)
    fps = N / t_duration

    # Set up figure and video writer
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_video_name, dpi=150):
        # Leg 1 transfer
        _animate_section(writer, fig, ax, mission_time, X_1, T_1,
                         pdv['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         desc='Leg 1 transfer')

        # Stay at asteroid 1
        _animate_stay_paths(writer, fig, ax, mission_time, 1, len(T_1),
                            pdv['et_arrive_1'], pdv['et_stay_1'],
                            a_id_1, a_id_2, a_id_3,
                            a_name_1, a_name_2, a_name_3,
                            desc='Stay at A1')

        # Leg 2 transfer
        _animate_section(writer, fig, ax, mission_time, X_2, T_2,
                         pdv['et_stay_1'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         desc='Leg 2 transfer')

        # Stay at asteroid 2
        _animate_stay_paths(writer, fig, ax, mission_time, 2, len(T_2),
                            pdv['et_arrive_2'], pdv['et_stay_2'],
                            a_id_1, a_id_2, a_id_3,
                            a_name_1, a_name_2, a_name_3,
                            desc='Stay at A2')

        # Leg 3 transfer
        _animate_section(writer, fig, ax, mission_time, X_3, T_3,
                         pdv['et_stay_2'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         desc='Leg 3 transfer')

    plt.close(fig)
    print(f"Video saved: {output_video_name}")


# =============================================================================
# ASTEROID ORBIT ANIMATION
# =============================================================================

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


# =============================================================================
# STATIC TRAJECTORY PLOT
# =============================================================================

def graph_asteroid_flightpath(path_defined_vector, a_id_1, a_id_2):
    """Plot the three trajectory legs in 3D.

    Parameters
    ----------
    path_defined_vector : dict
        Keys: et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2,
              et_arrive_3, delta_v_launch, delta_v_A1_leave, delta_v_A2_leave.
    a_id_1 : str
        NAIF ID string for asteroid 1.
    a_id_2 : str
        NAIF ID string for asteroid 2.
    """
    pdv = path_defined_vector

    t_1 = pdv['et_arrive_1'] - pdv['et_launch']
    t_2 = pdv['et_arrive_2'] - pdv['et_stay_1']
    t_3 = pdv['et_arrive_3'] - pdv['et_stay_2']

    mu_sun = MU_SUN

    # --- Leg 1: Earth to Asteroid 1 ---
    r_earth, v_earth = get_state('399', pdv['et_launch'])
    x_0 = np.concatenate([r_earth, v_earth + pdv['delta_v_launch']])
    X_1, T_1 = two_body_sim(t_1, x_0, mu_sun)

    # --- Leg 2: Asteroid 1 to Asteroid 2 ---
    r_a1, v_a1 = get_state(a_id_1, pdv['et_stay_1'])
    x_0 = np.concatenate([r_a1, v_a1 + pdv['delta_v_A1_leave']])
    X_2, T_2 = two_body_sim(t_2, x_0, mu_sun)

    # --- Leg 3: Asteroid 2 to Asteroid 3 ---
    r_a2, v_a2 = get_state(a_id_2, pdv['et_stay_2'])
    x_0 = np.concatenate([r_a2, v_a2 + pdv['delta_v_A2_leave']])
    X_3, T_3 = two_body_sim(t_3, x_0, mu_sun)

    # --- Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    ax.plot(X_1[:, 0], X_1[:, 1], X_1[:, 2], 'r', label='Trajectory to A1')
    ax.plot(X_2[:, 0], X_2[:, 1], X_2[:, 2], 'g', label='Trajectory from A1 to A2')
    ax.plot(X_3[:, 0], X_3[:, 1], X_3[:, 2], 'b', label='Trajectory from A2')

    ax.legend()
    ax.view_init(elev=30, azim=-60)  # 3D view
    plt.show()

