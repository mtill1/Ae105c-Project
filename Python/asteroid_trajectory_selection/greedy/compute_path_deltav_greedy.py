"""Compute delta-v for a greedy three-leg trajectory with optional flyby."""

import numpy as np
import spiceypy

from ..constants import DAY
from ..lambert import lambert


def compute_path_deltav_greedy(launch_body, landing_body, goal_body,
                               launch_date, arrival_date, goal_date,
                               m_1, m_2):
    """Compute delta-v for a greedy path with an optional gravity-assist flyby.

    Parameters
    ----------
    launch_body : str
        NAIF ID of the departure body.
    landing_body : str
        NAIF ID of the flyby body, or "-1" for a direct transfer.
    goal_body : str
        NAIF ID of the destination body.
    launch_date : float
        Ephemeris time of launch.
    arrival_date : float
        Ephemeris time of flyby (ignored if landing_body == "-1").
    goal_date : float
        Ephemeris time of arrival at the goal body.
    m_1 : float
        Lambert revolution parameter for the first leg.
        Sign controls transfer direction; abs value is rounded to number of revolutions.
    m_2 : float
        Lambert revolution parameter for the second leg.

    Returns
    -------
    dv_launch : float
    dv_arrive : float
    dv_goal : float
    dv_total : float
    lambert_launch : ndarray or None
    lambert_arrive_in : ndarray or None
    lambert_arrive_out : ndarray or None
    lambert_goal : ndarray or None
    """
    sgn_1 = np.sign(m_1)
    if sgn_1 == 0:
        sgn_1 = 1

    sgn_2 = np.sign(m_2)
    if sgn_2 == 0:
        sgn_2 = 1

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]

    # --- Direct transfer (no flyby) ---
    if landing_body == "-1":
        launch_state, _ = spiceypy.spkezr(launch_body, launch_date,
                                          'ECLIPJ2000', 'NONE', '10')
        goal_state, _ = spiceypy.spkezr(goal_body, goal_date,
                                        'ECLIPJ2000', 'NONE', '10')

        lambert_launch, lambert_goal, _, exit_flag = lambert(
            launch_state[0:3], goal_state[0:3],
            sgn_1 * (goal_date - launch_date) / DAY,
            round(abs(m_1)), myu_sun)

        lambert_arrive_out = None
        lambert_arrive_in = None

        if exit_flag != 1:
            return (1e3, 1e3, 1e3, 1e3, None, None, None, None)

        dv_launch = np.linalg.norm(lambert_launch - launch_state[3:6])
        dv_goal = np.linalg.norm(lambert_goal - goal_state[3:6])
        dv_arrive = 0.0
        dv_total = dv_launch + dv_goal

        return (dv_launch, dv_arrive, dv_goal, dv_total,
                lambert_launch, lambert_arrive_in, lambert_arrive_out, lambert_goal)

    # --- Two-leg transfer with flyby ---
    launch_state, _ = spiceypy.spkezr(launch_body, launch_date,
                                      'ECLIPJ2000', 'NONE', '10')
    arrive_state, _ = spiceypy.spkezr(landing_body, arrival_date,
                                      'ECLIPJ2000', 'NONE', '10')
    goal_state, _ = spiceypy.spkezr(goal_body, goal_date,
                                    'ECLIPJ2000', 'NONE', '10')

    HEIGHT_THRESHOLD = 100  # km

    lambert_launch, lambert_arrive_in, _, exit_flag_1 = lambert(
        launch_state[0:3], arrive_state[0:3],
        sgn_1 * (arrival_date - launch_date) / DAY,
        round(abs(m_1)), myu_sun)

    lambert_arrive_out, lambert_goal, _, exit_flag_2 = lambert(
        arrive_state[0:3], goal_state[0:3],
        sgn_2 * (goal_date - arrival_date) / DAY,
        round(abs(m_2)), myu_sun)

    if exit_flag_1 != 1 or exit_flag_2 != 1:
        return (1e3, 1e3, 1e3, 1e3, None, None, None, None)

    dv_launch = np.linalg.norm(lambert_launch - launch_state[3:6])

    v_inf_in = lambert_arrive_in - arrive_state[3:6]
    v_inf_out = lambert_arrive_out - arrive_state[3:6]

    flyby_id = int(float(landing_body))
    _, myu_flyby_vals = spiceypy.bodvcd(flyby_id, 'GM', 10)
    myu_flyby = myu_flyby_vals[0]

    # For Mars (body 4), use body 499 for radii
    if flyby_id == 4:
        _, r_p_vals = spiceypy.bodvcd(499, 'RADII', 10)
    else:
        _, r_p_vals = spiceypy.bodvcd(flyby_id, 'RADII', 10)
    r_p = r_p_vals[0]

    delta_des = np.arccos(
        np.clip(np.dot(v_inf_in, v_inf_out)
                / (np.linalg.norm(v_inf_in) * np.linalg.norm(v_inf_out)),
                -1.0, 1.0))
    delta_max = 2 * np.arcsin(
        1 / (1 + (r_p + HEIGHT_THRESHOLD)
             * (np.linalg.norm(v_inf_in) ** 2) / myu_flyby))

    dv_arrive = 0.0
    if delta_des > delta_max:
        v_p_in = np.sqrt(np.linalg.norm(v_inf_in) ** 2
                         + 2 * myu_flyby / (HEIGHT_THRESHOLD + r_p))
        v_p_out = np.sqrt(np.linalg.norm(v_inf_out) ** 2
                          + 2 * myu_flyby / (HEIGHT_THRESHOLD + r_p))
        d_delta = delta_des - delta_max
        dv_arrive = np.sqrt(v_p_in ** 2 + v_p_out ** 2
                            - 2 * v_p_in * v_p_out * np.cos(d_delta))

    dv_goal = np.linalg.norm(lambert_goal - goal_state[3:6])
    dv_total = dv_launch + dv_arrive + dv_goal

    return (dv_launch, dv_arrive, dv_goal, dv_total,
            lambert_launch, lambert_arrive_in, lambert_arrive_out, lambert_goal)
