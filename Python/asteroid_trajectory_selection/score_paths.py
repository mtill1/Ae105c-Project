from .unpack_input import unpack_input
from .compute_path_deltav import compute_path_deltav


def score_paths(input_vec, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_input(input_vec, launch_range)

    result = compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                                 et_arrive_1, et_stay_1, et_arrive_2,
                                 et_stay_2, et_arrive_3, m_1, m_2, m_3)

    return result['delta_v_total']
