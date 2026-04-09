function return_score = SCORE_PATHS(input_vec, A_ID_1, A_ID_2, A_ID_3, ...
    LAUNCH_RANGE, M_1, M_2, M_3)

    [et_launch, et_arrive_1, et_stay_1, et_arrive_2, ...
        et_stay_2, et_arrive_3] = UNPACK_INPUT(input_vec, LAUNCH_RANGE);



    [~, ~, ~, ~, ~, ~, delta_v_total] = COMPUTE_PATH_DELTAV(A_ID_1, ...
        A_ID_2, A_ID_3, et_launch, et_arrive_1, et_stay_1, et_arrive_2, ...
        et_stay_2, et_arrive_3, M_1, M_2, M_3);
    
    return_score = delta_v_total;
end


