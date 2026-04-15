function return_score = SCORE_PATHS_MARS(input_vec, A_ID_1, A_ID_2, A_ID_3, ...
    LAUNCH_RANGE, M_1, M_2, M_3, M_MARS)

    [et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, ...
        et_stay_2, et_arrive_3] = UNPACK_MARS_INPUT(input_vec, LAUNCH_RANGE);



    [~, ~, ~, ~, ~, ~, ~, ~, delta_v_total] = COMPUTE_MARS_PATH_DELTAV(A_ID_1, ...
        A_ID_2, A_ID_3, et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, ...
        et_stay_2, et_arrive_3, M_1, M_2, M_3, M_MARS);
    
    return_score = delta_v_total;
end


