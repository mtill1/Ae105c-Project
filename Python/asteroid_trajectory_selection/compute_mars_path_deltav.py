import numpy as np
import spiceypy

from .constants import DAY
from .lambert import lambert


def compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                             et_arrive_1, et_stay_1, et_arrive_2,
                             et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars):
    earth_launch_state, _ = spiceypy.spkezr('399', et_launch, 'ECLIPJ2000', 'NONE', '10')
    mars_flyby_state, _ = spiceypy.spkezr('4', et_mars, 'ECLIPJ2000', 'NONE', '10')
    a_1_arrive_state, _ = spiceypy.spkezr(str(a_id_1), et_arrive_1, 'ECLIPJ2000', 'NONE', '10')
    a_1_leaving_state, _ = spiceypy.spkezr(str(a_id_1), et_stay_1, 'ECLIPJ2000', 'NONE', '10')
    a_2_arrive_state, _ = spiceypy.spkezr(str(a_id_2), et_arrive_2, 'ECLIPJ2000', 'NONE', '10')
    a_2_leaving_state, _ = spiceypy.spkezr(str(a_id_2), et_stay_2, 'ECLIPJ2000', 'NONE', '10')
    a_3_arrive_state, _ = spiceypy.spkezr(str(a_id_3), et_arrive_3, 'ECLIPJ2000', 'NONE', '10')

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]
    _, myu_mars_vals = spiceypy.bodvcd(4, 'GM', 10)
    myu_mars = myu_mars_vals[0]
    _, r_p_vals = spiceypy.bodvcd(499, 'RADII', 10)
    r_p = r_p_vals[0]

    earth_lambert_velocity, mars_arrive_velocity, _, exit_flag_1 = \
        lambert(earth_launch_state[0:3], mars_flyby_state[0:3],
                -(et_mars - et_launch) / DAY, m_mars, myu_sun)

    v_mars_leave, a_1_arrive_lambert_velocity, _, exit_flag_2 = \
        lambert(mars_flyby_state[0:3], a_1_arrive_state[0:3],
                -(et_arrive_1 - et_mars) / DAY, m_1, myu_sun)

    a_1_leave_lambert_velocity, a_2_arrive_lambert_velocity, _, exit_flag_3 = \
        lambert(a_1_leaving_state[0:3], a_2_arrive_state[0:3],
                -(et_arrive_2 - et_stay_1) / DAY, m_2, myu_sun)
    a_2_leave_lambert_velocity, a_3_arrive_lambert_velocity, _, exit_flag_4 = \
        lambert(a_2_leaving_state[0:3], a_3_arrive_state[0:3],
                -(et_arrive_3 - et_stay_2) / DAY, m_3, myu_sun)

    if exit_flag_1 != 1 or exit_flag_2 != 1 or exit_flag_3 != 1 or exit_flag_4 != 1:
        return {
            'delta_v_launch': np.array([]),
            'v_mars_leave': np.array([]),
            'delta_v_mars': 1e3,
            'delta_v_A1_arrive': np.array([]),
            'delta_v_A1_leave': np.array([]),
            'delta_v_A2_arrive': np.array([]),
            'delta_v_A2_leave': np.array([]),
            'delta_v_A3_arrive': np.array([]),
            'delta_v_total': 1e3,
        }

    delta_v_launch = earth_lambert_velocity - earth_launch_state[3:6]
    delta_v_A1_arrive = a_1_arrive_lambert_velocity - a_1_arrive_state[3:6]
    delta_v_A1_leave = a_1_leave_lambert_velocity - a_1_leaving_state[3:6]
    delta_v_A2_arrive = a_2_arrive_lambert_velocity - a_2_arrive_state[3:6]
    delta_v_A2_leave = a_2_leave_lambert_velocity - a_2_leaving_state[3:6]
    delta_v_A3_arrive = a_3_arrive_lambert_velocity - a_3_arrive_state[3:6]

    delta_v_mars = 0.0

    v_inf_mars_in = mars_arrive_velocity - mars_flyby_state[3:6]
    v_inf_mars_out = v_mars_leave - mars_flyby_state[3:6]

    HEIGHT_THRESHOLD = 200  # km

    delta_max = 2 * np.arcsin(
        1 / (1 + (r_p + HEIGHT_THRESHOLD) * (np.linalg.norm(v_inf_mars_in) ** 2) / myu_mars))
    delta_des = np.arccos(
        np.dot(v_inf_mars_in, v_inf_mars_out)
        / (np.linalg.norm(v_inf_mars_in) * np.linalg.norm(v_inf_mars_out)))

    if delta_des > delta_max or np.linalg.norm(v_inf_mars_in) != np.linalg.norm(v_inf_mars_out):
        v_p_in = np.sqrt(np.dot(v_inf_mars_in, v_inf_mars_in)
                         + 2 * myu_mars / (r_p + HEIGHT_THRESHOLD))
        v_p_out = np.sqrt(np.dot(v_inf_mars_out, v_inf_mars_out)
                          + 2 * myu_mars / (r_p + HEIGHT_THRESHOLD))
        delta_v_mars = v_p_out - v_p_in

    delta_v_total = (np.linalg.norm(delta_v_launch)
                     + np.linalg.norm(delta_v_A1_arrive)
                     + np.linalg.norm(delta_v_A1_leave)
                     + np.linalg.norm(delta_v_A2_arrive)
                     + np.linalg.norm(delta_v_A2_leave)
                     + np.linalg.norm(delta_v_A3_arrive)
                     + abs(delta_v_mars))

    return {
        'delta_v_launch': delta_v_launch,
        'v_mars_leave': v_mars_leave,
        'delta_v_mars': delta_v_mars,
        'delta_v_A1_arrive': delta_v_A1_arrive,
        'delta_v_A1_leave': delta_v_A1_leave,
        'delta_v_A2_arrive': delta_v_A2_arrive,
        'delta_v_A2_leave': delta_v_A2_leave,
        'delta_v_A3_arrive': delta_v_A3_arrive,
        'delta_v_total': delta_v_total,
    }
