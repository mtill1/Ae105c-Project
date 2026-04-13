function optimized_output = OPTIMIZE_GREEDY_TIMES(A_ID_1, A_ID_2, A_ID_3, ...
    LAUNCH_RANGE, M)
    
    MINUTE = 60;
    HOUR = MINUTE * 60;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;
    MONTH = 4 * WEEK;
    YEAR = 12 * MONTH;
    
    MAX_STAYTIME = YEAR;
    MIN_STAYTIME = 3 * MONTH;
    
    AVERAGE_STAYTIME = mean([MIN_STAYTIME, MAX_STAYTIME]);

    N_RES = 2;

    LOWER_BOUND = [0, 2*WEEK, 2*WEEK]/YEAR;
    UPPER_BOUND = [LAUNCH_RANGE(2) - LAUNCH_RANGE(1), 1.5*YEAR, 1.5*YEAR]/YEAR;

    DELTA_GUESS = (UPPER_BOUND - LOWER_BOUND)./N_RES;


    FLYBY_IDs = ["399", "4", "-1"];

    bodies = ['399', A_ID_1, A_ID_2, A_ID_3];
    bodies_strs = ["399", string(A_ID_1), string(A_ID_2), string(A_ID_3)];

    optimized_output(3) = struct("dv_launch", Inf, ...
        "dv_arrive", Inf, "dv_goal", Inf, "dv_total", Inf, ...
        "LAMBERT_LAUNCH", Inf, "LAMBERT_ARRIVE_IN", Inf, ...
        "LAMBERT_ARRIVE_OUT", Inf, "LAMBERT_GOAL", Inf, ...
            "FLYBY_BODY", Inf, "et_launch", Inf, ...
            "et_flyby", Inf, "et_goal", Inf);

    min_launch_date = -1;
    min_arrival_date = -1;
    min_goal_date = -1;
    min_arrival_body_id = '';

    current_launch_range = LAUNCH_RANGE;

    for i = 1:3
        %fprintf("BODIES %d, %s", i, bodies_strs(i));
        min_dv_total = Inf;
        LOWER_BOUND = [0, 2*WEEK, 2*WEEK]/YEAR;
        UPPER_BOUND = [current_launch_range(2) - current_launch_range(1), 3*YEAR, 3*YEAR]/YEAR;
        for n_1 = 1:N_RES
            for n_2 = 1:1
                for n_3 = 1:1
                    for flyby_index = 1:length(FLYBY_IDs)
                        if i == 1 && flyby_index == 3
                            continue
                        end
                        
                        current_body_id = bodies_strs(i);

                        if FLYBY_IDs(flyby_index) == current_body_id
                            continue
                        end

                        INPUT_GUESS = UPPER_BOUND - DELTA_GUESS .* ...
                            [n_1, n_2, n_3];
    
                        options = optimoptions('fmincon','Display','none', ...
                            'StepTolerance', 1e-10);

                        [optimized_time_vector, delta_v_score] = fmincon( ...
                            @(x) SCORE_PATHS_GREEDY(x, current_launch_range, ...
                            bodies_strs(i), FLYBY_IDs(flyby_index), bodies_strs(i + 1), M(i,1), M(i, 2)), ...
                            INPUT_GUESS, [], [], [], [], LOWER_BOUND, UPPER_BOUND, [], options);
                        
                        launch_date = optimized_time_vector(1) * YEAR + current_launch_range(1);
                        arrival_date = optimized_time_vector(2) * YEAR + launch_date;
                        goal_date = optimized_time_vector(3) * YEAR + arrival_date;
                
                        if delta_v_score < min_dv_total
                            min_dv_total = delta_v_score;
                            min_current_body_id = current_body_id;
                            min_launch_date = launch_date;
                            min_arrival_date = arrival_date;
                            min_arrival_body_id = FLYBY_IDs(flyby_index);
                            min_goal_date = goal_date;
                        end
                    end   
                end
            end
        end
        
        current_launch_range(1) = goal_date + 3 * MONTH;
        current_launch_range(2) = goal_date + 6 * MONTH;
        

        [dv_launch, dv_arrive, dv_goal, dv_total, LAMBERT_LAUNCH, ...
            LAMBERT_ARRIVE_IN, LAMBERT_ARRIVE_OUT, LAMBERT_GOAL] = ...
            COMPUTE_PATH_DELTAV_GREEDY(min_current_body_id, ...
            min_arrival_body_id, bodies_strs(i + 1), min_launch_date, ...
            min_arrival_date, min_goal_date, M(i, 1), M(i, 2));


        optimized_output(i) = struct("dv_launch", dv_launch, ...
            "dv_arrive", dv_arrive, "dv_goal", dv_goal, "dv_total", dv_total, ...
            "LAMBERT_LAUNCH", LAMBERT_LAUNCH, "LAMBERT_ARRIVE_IN", LAMBERT_ARRIVE_IN, ...
            "LAMBERT_ARRIVE_OUT", LAMBERT_ARRIVE_OUT, "LAMBERT_GOAL", LAMBERT_GOAL, ...
            "FLYBY_BODY", min_arrival_body_id, "et_launch", min_launch_date, ...
            "et_flyby", min_arrival_date, "et_goal", min_goal_date);

    end
    
end

