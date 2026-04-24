function optimized_output = OPTIMIZE_TIMES(A_ID_1, A_ID_2, A_ID_3, ...
    LAUNCH_RANGE, M_1, M_2, M_3)
    
    MINUTE = 60;
    HOUR = MINUTE * 60;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;
    MONTH = 4 * WEEK;
    YEAR = 12 * MONTH;
    
    MAX_STAYTIME = YEAR;
    MIN_STAYTIME = 3 * MONTH;
    
    AVERAGE_STAYTIME = mean([MIN_STAYTIME, MAX_STAYTIME]);

    N_RES = [5, 3, 2, 3, 2, 3];

    LOWER_BOUND = [0, 2*WEEK, MIN_STAYTIME, 2*WEEK, ...
            MIN_STAYTIME, 2*WEEK]/YEAR;
    UPPER_BOUND = [LAUNCH_RANGE(2) - LAUNCH_RANGE(1), 8*YEAR, MAX_STAYTIME, 8*YEAR, ...
            MAX_STAYTIME, 8*YEAR]/YEAR;

    iter_vec = 0 .* N_RES;

    DELTA_GUESS = (UPPER_BOUND - LOWER_BOUND)./N_RES;

    min_times = [inf, inf, inf, inf, inf, inf];
    min_score = inf;

    while iter_vec(end) <= N_RES(end)
        while any(iter_vec > N_RES)
            for j = 2:length(N_RES)

                if iter_vec(j) >= N_RES(j)
    
                    iter_vec(j-1) = iter_vec(j-1) + 1;
                    iter_vec(j) = 0;
                end
            end
        end

    
        INPUT_GUESS = LOWER_BOUND + DELTA_GUESS .* iter_vec; % Travel for 5 months, stay for 4.5
        
    
        options = optimoptions('fmincon','Display','none', 'StepTolerance', 1e-7);
    
        [optimized_vector, delta_v_score] = fmincon(@(x) SCORE_PATHS(x, ...
            A_ID_1, A_ID_2, A_ID_3, LAUNCH_RANGE, M_1, M_2, M_3), ...
            INPUT_GUESS, [], [], [], [], LOWER_BOUND, UPPER_BOUND, [], options);
        
        [et_launch, et_arrive_1, et_stay_1, et_arrive_2, ...
            et_stay_2, et_arrive_3] = UNPACK_INPUT(optimized_vector, LAUNCH_RANGE);

        iter_vec(end) = iter_vec(end) + 1;

        if delta_v_score < min_score
            min_times = [et_launch, et_arrive_1, et_stay_1, et_arrive_2, ...
            et_stay_2, et_arrive_3];
            min_score = delta_v_score;
        end
    end
    
    et_launch = min_times(1);
    et_arrive_1 = min_times(2);
    et_stay_1 = min_times(3);
    et_arrive_2 = min_times(4);
    et_stay_2 = min_times(5);
    et_arrive_3 = min_times(6);
    
    
    [delta_v_launch, delta_v_A1_arrive, delta_v_A1_leave, ...
    delta_v_A2_arrive, delta_v_A2_leave, delta_v_A3_arrive, delta_v_total] ...
    = COMPUTE_PATH_DELTAV(A_ID_1, A_ID_2, A_ID_3, et_launch, ...
    et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3, M_1, M_2, M_3);

    optimized_output = struct("delta_v_launch", delta_v_launch, ...
        "delta_v_A1_arrive", delta_v_A1_arrive, "delta_v_A1_leave", delta_v_A1_leave, ...
        "delta_v_A2_arrive", delta_v_A2_arrive, "delta_v_A2_leave", delta_v_A2_leave, ...
        "delta_v_A3_arrive", delta_v_A3_arrive, "delta_v_total", delta_v_total, ...
        "et_launch", et_launch, "et_arrive_1", et_arrive_1, "et_stay_1", et_stay_1, ...
        "et_arrive_2", et_arrive_2, "et_stay_2", et_stay_2, ...
        "et_arrive_3", et_arrive_3);
end

