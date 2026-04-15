"""Animate a greedy three-leg flightpath with optional flyby sub-legs."""

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from ..constants import MINUTE, HOUR, DAY, WEEK
from ..two_body_sim import two_body_sim


def greedy_flightpath_animation(path_defined_vector, asteroid_list,
                                a_index_1, a_index_2, a_index_3,
                                t_duration, output_video_name):
    """Create a 3D animated video of the greedy three-leg trajectory.

    Parameters
    ----------
    path_defined_vector : list of 3 dicts
        Each dict contains et_launch, et_flyby, et_goal, LAMBERT_LAUNCH,
        LAMBERT_ARRIVE_IN, LAMBERT_ARRIVE_OUT, LAMBERT_GOAL.
    asteroid_list : list of dict
        Each dict has 'ID' (int) and 'NAME' (str) keys.
    a_index_1, a_index_2, a_index_3 : int
        Indices into asteroid_list for the three asteroids.
    t_duration : float
        Desired video duration in seconds.
    output_video_name : str
        Output filename (e.g. "GREEDY_BEST_PATH.mp4").
    """
    a_id_1 = str(int(asteroid_list[a_index_1]['ID']))
    a_id_2 = str(int(asteroid_list[a_index_2]['ID']))
    a_id_3 = str(int(asteroid_list[a_index_3]['ID']))

    a_name_1 = asteroid_list[a_index_1]['NAME']
    a_name_2 = asteroid_list[a_index_2]['NAME']
    a_name_3 = asteroid_list[a_index_3]['NAME']

    # Mission time span for orbit backdrop
    mission_time = np.arange(path_defined_vector[0]['et_launch'],
                             path_defined_vector[2]['et_goal'],
                             1 * WEEK)

    # Time differences for each sub-leg
    t_diff_1_1 = (path_defined_vector[0]['et_flyby']
                  - path_defined_vector[0]['et_launch'])
    t_diff_1_2 = (path_defined_vector[0]['et_goal']
                  - path_defined_vector[0]['et_flyby'])

    t_diff_2_1 = (path_defined_vector[1]['et_flyby']
                  - path_defined_vector[1]['et_launch'])
    t_diff_2_2 = (path_defined_vector[1]['et_goal']
                  - path_defined_vector[1]['et_flyby'])

    t_diff_3_1 = (path_defined_vector[2]['et_flyby']
                  - path_defined_vector[2]['et_launch'])
    t_diff_3_2 = (path_defined_vector[2]['et_goal']
                  - path_defined_vector[2]['et_flyby'])

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]

    # Get departure states
    earth_launch_state, _ = spiceypy.spkezr(
        '399', path_defined_vector[0]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')
    a1_state, _ = spiceypy.spkezr(
        a_id_1, path_defined_vector[1]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')
    a2_state, _ = spiceypy.spkezr(
        a_id_2, path_defined_vector[2]['et_launch'],
        'ECLIPJ2000', 'NONE', '10')

    # --- Leg 1 sub-leg 1 ---
    x_0 = np.concatenate([earth_launch_state[0:3],
                          path_defined_vector[0]['LAMBERT_LAUNCH']])
    X_1_1, T_1_1 = two_body_sim(t_diff_1_1, x_0, myu_sun)

    # --- Leg 1 sub-leg 2 ---
    if path_defined_vector[0]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_1_1[-1, 0:3],
                              path_defined_vector[0]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_1_1[-1, 0:6]
    X_1_2, T_1_2 = two_body_sim(t_diff_1_2, x_0, myu_sun)

    # --- Leg 2 sub-leg 1 ---
    x_0 = np.concatenate([a1_state[0:3],
                          path_defined_vector[1]['LAMBERT_LAUNCH']])
    X_2_1, T_2_1 = two_body_sim(t_diff_2_1, x_0, myu_sun)

    # --- Leg 2 sub-leg 2 ---
    if path_defined_vector[1]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_2_1[-1, 0:3],
                              path_defined_vector[1]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_2_1[-1, 0:6]
    X_2_2, T_2_2 = two_body_sim(t_diff_2_2, x_0, myu_sun)

    # --- Leg 3 sub-leg 1 ---
    x_0 = np.concatenate([a2_state[0:3],
                          path_defined_vector[2]['LAMBERT_LAUNCH']])
    X_3_1, T_3_1 = two_body_sim(t_diff_3_1, x_0, myu_sun)

    # --- Leg 3 sub-leg 2 ---
    if path_defined_vector[2]['LAMBERT_ARRIVE_OUT'] is not None:
        x_0 = np.concatenate([X_3_1[-1, 0:3],
                              path_defined_vector[2]['LAMBERT_ARRIVE_OUT']])
    else:
        x_0 = X_3_1[-1, 0:6]
    X_3_2, T_3_2 = two_body_sim(t_diff_3_2, x_0, myu_sun)

    # --- Compute total frame count and FPS ---
    N = (2 * (len(T_1_1) + len(T_1_2))
         + (len(T_2_1) + len(T_2_2))
         + (len(T_3_1) + len(T_3_2)))
    FPS = max(1, N / t_duration)

    # Pre-compute full-mission orbit positions for backdrop
    earth_orbit = np.array([spiceypy.spkezr('399', t, 'ECLIPJ2000', 'NONE', '10')[0]
                            for t in mission_time])
    mars_orbit = np.array([spiceypy.spkezr('4', t, 'ECLIPJ2000', 'NONE', '10')[0]
                           for t in mission_time])
    a1_orbit = np.array([spiceypy.spkezr(a_id_1, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])
    a2_orbit = np.array([spiceypy.spkezr(a_id_2, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])
    a3_orbit = np.array([spiceypy.spkezr(a_id_3, t, 'ECLIPJ2000', 'NONE', '10')[0]
                         for t in mission_time])

    fig = plt.figure(figsize=(16, 10))
    writer = FFMpegWriter(fps=FPS)

    with writer.saving(fig, output_video_name, dpi=100):
        # Leg 1 sub-leg 1
        _animate_section(fig, writer, mission_time, X_1_1, T_1_1,
                         path_defined_vector[0]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 1 sub-leg 2
        _animate_section(fig, writer, mission_time, X_1_2, T_1_2,
                         path_defined_vector[0]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Stay at asteroid 1
        _animate_stay(fig, writer, mission_time, 1,
                      len(T_1_1) + len(T_1_2),
                      path_defined_vector[0]['et_goal'],
                      path_defined_vector[1]['et_launch'],
                      a_id_1, a_id_2, a_id_3,
                      a_name_1, a_name_2, a_name_3,
                      earth_orbit, a1_orbit, a2_orbit, a3_orbit)
        # Leg 2 sub-leg 1
        _animate_section(fig, writer, mission_time, X_2_1, T_2_1,
                         path_defined_vector[1]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 2 sub-leg 2
        _animate_section(fig, writer, mission_time, X_2_2, T_2_2,
                         path_defined_vector[1]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Stay at asteroid 2
        _animate_stay(fig, writer, mission_time, 2,
                      len(T_2_1) + len(T_2_2),
                      path_defined_vector[1]['et_goal'],
                      path_defined_vector[2]['et_launch'],
                      a_id_1, a_id_2, a_id_3,
                      a_name_1, a_name_2, a_name_3,
                      earth_orbit, a1_orbit, a2_orbit, a3_orbit)
        # Leg 3 sub-leg 1
        _animate_section(fig, writer, mission_time, X_3_1, T_3_1,
                         path_defined_vector[2]['et_launch'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)
        # Leg 3 sub-leg 2
        _animate_section(fig, writer, mission_time, X_3_2, T_3_2,
                         path_defined_vector[2]['et_flyby'],
                         a_id_1, a_id_2, a_id_3,
                         a_name_1, a_name_2, a_name_3,
                         earth_orbit, mars_orbit,
                         a1_orbit, a2_orbit, a3_orbit)

    plt.close(fig)


def _animate_section(fig, writer, mission_time, X_i, T_i, et_launch,
                     a_id_1, a_id_2, a_id_3,
                     a_name_1, a_name_2, a_name_3,
                     earth_orbit, mars_orbit,
                     a1_orbit, a2_orbit, a3_orbit):
    """Animate one transfer sub-leg."""
    for i in range(len(T_i)):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True, which='both')

        # Orbit backdrop
        ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
                label='Earth', color='cyan')
        ax.plot(mars_orbit[:, 0], mars_orbit[:, 1], mars_orbit[:, 2],
                label='Mars', color='magenta')
        ax.plot(a1_orbit[:, 0], a1_orbit[:, 1], a1_orbit[:, 2],
                label=a_name_1, color='red')
        ax.plot(a2_orbit[:, 0], a2_orbit[:, 1], a2_orbit[:, 2],
                label=a_name_2, color='green')
        ax.plot(a3_orbit[:, 0], a3_orbit[:, 1], a3_orbit[:, 2],
                label=a_name_3, color='blue')

        # Current positions
        t_now = T_i[i] + et_launch
        for body_id, color in [('399', 'cyan'), ('4', 'magenta')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)
        for body_id, color in [(a_id_1, 'red'), (a_id_2, 'green'), (a_id_3, 'blue')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)

        # Spacecraft position
        ax.scatter([X_i[i, 0]], [X_i[i, 1]], [X_i[i, 2]], c='white', s=30)
        # Full trajectory
        ax.plot(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                label='Trajectory', color='white')

        ax.legend()
        ax.set_title(spiceypy.et2utc(t_now, 'C', 6))
        fig.set_facecolor('black')
        ax.set_facecolor('black')

        writer.grab_frame()


def _animate_stay(fig, writer, mission_time, asteroid_stay_number, n_time,
                  et_start, et_stop,
                  a_id_1, a_id_2, a_id_3,
                  a_name_1, a_name_2, a_name_3,
                  earth_orbit, a1_orbit, a2_orbit, a3_orbit):
    """Animate the stay period at an asteroid."""
    t_range = np.linspace(et_start, et_stop, n_time)

    # Which asteroid are we staying at?
    stay_id = a_id_1 if asteroid_stay_number == 1 else a_id_2
    stay_positions = np.array([
        spiceypy.spkezr(stay_id, t, 'ECLIPJ2000', 'NONE', '10')[0]
        for t in t_range])

    for i in range(n_time):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True, which='both')

        # Orbit backdrop
        ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
                label='Earth', color='cyan')
        ax.plot(a1_orbit[:, 0], a1_orbit[:, 1], a1_orbit[:, 2],
                label=a_name_1, color='red')
        ax.plot(a2_orbit[:, 0], a2_orbit[:, 1], a2_orbit[:, 2],
                label=a_name_2, color='green')
        ax.plot(a3_orbit[:, 0], a3_orbit[:, 1], a3_orbit[:, 2],
                label=a_name_3, color='blue')

        # Current positions
        t_now = t_range[i]
        st, _ = spiceypy.spkezr('399', t_now, 'ECLIPJ2000', 'NONE', '10')
        ax.scatter([st[0]], [st[1]], [st[2]], c='cyan', s=20)
        for body_id, color in [(a_id_1, 'red'), (a_id_2, 'green'), (a_id_3, 'blue')]:
            st, _ = spiceypy.spkezr(body_id, t_now, 'ECLIPJ2000', 'NONE', '10')
            ax.scatter([st[0]], [st[1]], [st[2]], c=color, s=20)

        # Trajectory (stay path)
        ax.plot(stay_positions[:, 0], stay_positions[:, 1], stay_positions[:, 2],
                label='Trajectory', color='white')

        ax.legend()
        ax.set_title(spiceypy.et2utc(t_now, 'C', 6))
        fig.set_facecolor('black')
        ax.set_facecolor('black')

        writer.grab_frame()
