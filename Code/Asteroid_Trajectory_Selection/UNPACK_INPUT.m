function [et_launch, et_arrive_1, et_stay_1, et_arrive_2, ...
        et_stay_2, et_arrive_3] = UNPACK_INPUT(input_vec, LAUNCH_RANGE)
    
    MINUTE = 60;
    HOUR = MINUTE * 60;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;
    MONTH = 4 * WEEK;
    YEAR = 12*MONTH;
    
    input_vec_scaled = YEAR * input_vec;
    et_launch = input_vec_scaled(1) + LAUNCH_RANGE(1);
    t_arrive_1 = input_vec_scaled(2);
    t_stay_1 = input_vec_scaled(3);
    t_arrive_2 = input_vec_scaled(4);
    t_stay_2 = input_vec_scaled(5);
    t_arrive_3 = input_vec_scaled(6);

    et_arrive_1 = et_launch + t_arrive_1;
    et_stay_1 = et_arrive_1 + t_stay_1;
    et_arrive_2 = et_stay_1 + t_arrive_2;
    et_stay_2 = et_arrive_2 + t_stay_2;
    et_arrive_3 = et_stay_2 + t_arrive_3;
end