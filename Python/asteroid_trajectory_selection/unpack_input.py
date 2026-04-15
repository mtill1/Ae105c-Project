"""Unpack optimization input vectors into epoch times."""

from .constants import YEAR


def unpack_input(input_vec, launch_range):
    """Unpack a 6-element input vector into six epoch times.

    Returns (et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3).
    """
    input_vec_scaled = YEAR * input_vec

    et_launch = input_vec_scaled[0] + launch_range[0]
    t_arrive_1 = input_vec_scaled[1]
    t_stay_1 = input_vec_scaled[2]
    t_arrive_2 = input_vec_scaled[3]
    t_stay_2 = input_vec_scaled[4]
    t_arrive_3 = input_vec_scaled[5]

    et_arrive_1 = et_launch + t_arrive_1
    et_stay_1 = et_arrive_1 + t_stay_1
    et_arrive_2 = et_stay_1 + t_arrive_2
    et_stay_2 = et_arrive_2 + t_stay_2
    et_arrive_3 = et_stay_2 + t_arrive_3

    return et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3


def unpack_mars_input(input_vec, launch_range):
    """Unpack a 7-element input vector (with Mars flyby) into seven epoch times.

    Returns (et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3).
    """
    input_vec_scaled = YEAR * input_vec

    et_launch = input_vec_scaled[0] + launch_range[0]
    t_mars = input_vec_scaled[1]
    t_arrive_1 = input_vec_scaled[2]
    t_stay_1 = input_vec_scaled[3]
    t_arrive_2 = input_vec_scaled[4]
    t_stay_2 = input_vec_scaled[5]
    t_arrive_3 = input_vec_scaled[6]

    et_mars = et_launch + t_mars
    et_arrive_1 = et_mars + t_arrive_1
    et_stay_1 = et_arrive_1 + t_stay_1
    et_arrive_2 = et_stay_1 + t_arrive_2
    et_stay_2 = et_arrive_2 + t_stay_2
    et_arrive_3 = et_stay_2 + t_arrive_3

    return et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3
