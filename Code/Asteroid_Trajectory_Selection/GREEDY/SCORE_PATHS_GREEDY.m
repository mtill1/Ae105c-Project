function return_score = SCORE_PATHS_GREEDY(time_vector, LAUNCH_RANGE, ...
    launch_body, landing_body, goal_body, M_1, M_2)

    MINUTE = 60;
    HOUR = MINUTE * 60;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;
    MONTH = 4 * WEEK;
    YEAR = 12*MONTH;

    launch_date = time_vector(1) * YEAR + LAUNCH_RANGE(1);
    arrival_date = time_vector(2) * YEAR + launch_date;
    goal_date = time_vector(3) * YEAR + arrival_date;

    [~, ~, ~, dv_total, ~, ...
        ~, ~, ~] = ...
    COMPUTE_PATH_DELTAV_GREEDY(launch_body, ...
        landing_body, goal_body, launch_date, arrival_date, ...
        goal_date, M_1, M_2);
    
    
    return_score = dv_total;
end


