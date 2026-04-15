import numpy as np
import spiceypy

from .constants import DAY
from .lambert import lambert


def compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                        et_arrive_1, et_stay_1, et_arrive_2,
                        et_stay_2, et_arrive_3, m_1, m_2, m_3):
    earth_launch_state, _ = spiceypy.spkezr('399', et_launch, 'ECLIPJ2000', 'NONE', '10')
    a_1_arrive_state, _ = spiceypy.spkezr(str(a_id_1), et_arrive_1, 'ECLIPJ2000', 'NONE', '10')
    a_1_leaving_state, _ = spiceypy.spkezr(str(a_id_1), et_stay_1, 'ECLIPJ2000', 'NONE', '10')
    a_2_arrive_state, _ = spiceypy.spkezr(str(a_id_2), et_arrive_2, 'ECLIPJ2000', 'NONE', '10')
    a_2_leaving_state, _ = spiceypy.spkezr(str(a_id_2), et_stay_2, 'ECLIPJ2000', 'NONE', '10')
    a_3_arrive_state, _ = spiceypy.spkezr(str(a_id_3), et_arrive_3, 'ECLIPJ2000', 'NONE', '10')

    _, myu_sun_vals = spiceypy.bodvcd(10, 'GM', 10)
    myu_sun = myu_sun_vals[0]

    earth_lambert_velocity, a_1_arrive_lambert_velocity, _, exit_flag_1 = \
        lambert(earth_launch_state[0:3], a_1_arrive_state[0:3],
                (et_arrive_1 - et_launch) / DAY, m_1, myu_sun)
    a_1_leave_lambert_velocity, a_2_arrive_lambert_velocity, _, exit_flag_2 = \
        lambert(a_1_leaving_state[0:3], a_2_arrive_state[0:3],
                (et_arrive_2 - et_stay_1) / DAY, m_2, myu_sun)
    a_2_leave_lambert_velocity, a_3_arrive_lambert_velocity, _, exit_flag_3 = \
        lambert(a_2_leaving_state[0:3], a_3_arrive_state[0:3],
                (et_arrive_3 - et_stay_2) / DAY, m_3, myu_sun)

    if exit_flag_1 != 1 or exit_flag_2 != 1 or exit_flag_3 != 1:
        return {
            'delta_v_launch': np.array([]),
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

    delta_v_total = (np.linalg.norm(delta_v_A1_arrive)
                     + np.linalg.norm(delta_v_A1_leave)
                     + np.linalg.norm(delta_v_A2_arrive)
                     + np.linalg.norm(delta_v_A2_leave)
                     + np.linalg.norm(delta_v_A3_arrive))

    return {
        'delta_v_launch': delta_v_launch,
        'delta_v_A1_arrive': delta_v_A1_arrive,
        'delta_v_A1_leave': delta_v_A1_leave,
        'delta_v_A2_arrive': delta_v_A2_arrive,
        'delta_v_A2_leave': delta_v_A2_leave,
        'delta_v_A3_arrive': delta_v_A3_arrive,
        'delta_v_total': delta_v_total,
    }
