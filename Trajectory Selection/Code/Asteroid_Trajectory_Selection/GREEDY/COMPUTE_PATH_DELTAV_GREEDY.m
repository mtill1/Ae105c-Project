function [dv_launch, dv_arrive, dv_goal, dv_total, LAMBERT_LAUNCH, ...
        LAMBERT_ARRIVE_IN, LAMBERT_ARRIVE_OUT, LAMBERT_GOAL] = ...
        COMPUTE_PATH_DELTAV_GREEDY(launch_body, ...
        landing_body, goal_body, launch_date, arrival_date, goal_date, M_1, M_2)

    MINUTE = 60;
    HOUR = 60 * MINUTE;
    DAY = 24 * HOUR;
    
    sgn_1 = sign(M_1);
    if sgn_1 == 0
        sgn_1 = 1;
    end
    
    sgn_2 = sign(M_2);
    if sgn_2 == 0
        sgn_2 = 1;
    end


    MYU_SUN = cspice_bodvcd(10, 'GM', 10);

    if landing_body == "-1"
        LAUNCH_STATE = cspice_spkezr(char(launch_body), launch_date, 'ECLIPJ2000', 'NONE', '10');
        GOAL_STATE = cspice_spkezr(char(goal_body), goal_date, 'ECLIPJ2000', 'NONE', '10');
        [LAMBERT_LAUNCH, LAMBERT_GOAL, ~, EXIT_FLAG] = ...
            lambert(LAUNCH_STATE(1:3)', GOAL_STATE(1:3)', ...
            sgn_1 * (goal_date - launch_date) / DAY, round(abs(M_1)), MYU_SUN);
        LAMBERT_ARRIVE_OUT = [];
        LAMBERT_ARRIVE_IN = [];

        if EXIT_FLAG ~= 1
            dv_launch = 1e3;
            dv_arrive = 1e3;
            dv_total = 1e3;
            dv_goal = 1e3;
    
            LAMBERT_LAUNCH = [];
            LAMBERT_ARRIVE_IN = [];
            LAMBERT_ARRIVE_OUT = [];
            LAMBERT_GOAL = [];
            %disp("TEST")
            return
            
        end

        dv_launch = norm(LAMBERT_LAUNCH' - LAUNCH_STATE(4:6));
        dv_goal = norm(LAMBERT_GOAL' - GOAL_STATE(4:6));
        
        dv_arrive = 0;
        dv_total = dv_launch + dv_goal;
        
        return
    end

    LAUNCH_STATE = cspice_spkezr(char(launch_body), launch_date, 'ECLIPJ2000', 'NONE', '10');
    ARRIVE_STATE = cspice_spkezr(char(landing_body), arrival_date, 'ECLIPJ2000', 'NONE', '10');
    GOAL_STATE = cspice_spkezr(char(goal_body), goal_date, 'ECLIPJ2000', 'NONE', '10');
    
    HEIGHT_THRESHOLD = 50; % km




    

    [LAMBERT_LAUNCH, LAMBERT_ARRIVE_IN, ~, EXIT_FLAG_1] = ...
        lambert(LAUNCH_STATE(1:3)', ARRIVE_STATE(1:3)', ...
        sgn_1 * (arrival_date - launch_date) / DAY, round(abs(M_1)), MYU_SUN);

    [LAMBERT_ARRIVE_OUT, LAMBERT_GOAL, ~, EXIT_FLAG_2] = ...
        lambert(ARRIVE_STATE(1:3)', GOAL_STATE(1:3)', ...
        sgn_2 * (goal_date - arrival_date) / DAY, round(abs(M_2)), MYU_SUN);

    

    if EXIT_FLAG_1 ~= 1 || EXIT_FLAG_2 ~= 1
        dv_launch = 1e3;
        dv_arrive = 1e3;
        dv_total = 1e3;
        dv_goal = 1e3;

        LAMBERT_LAUNCH = [];
        LAMBERT_ARRIVE_IN = [];
        LAMBERT_ARRIVE_OUT = [];
        LAMBERT_GOAL = [];
        %disp("TEST")
        return
        
    end

    dv_launch = norm(LAMBERT_LAUNCH' - LAUNCH_STATE(4:6));

    v_inf_in = LAMBERT_ARRIVE_IN' - ARRIVE_STATE(4:6);
    v_inf_out = LAMBERT_ARRIVE_OUT' - ARRIVE_STATE(4:6);

    myu_flyby = cspice_bodvcd(str2double(landing_body), 'GM', 10);

    
    if str2double(landing_body) == 4
        R_p = cspice_bodvcd(499, 'RADII', 10);
    else
        R_p = cspice_bodvcd(str2double(landing_body), 'RADII', 10);
    end

    R_p = R_p(1);

    delta_des = acos(dot(v_inf_in, v_inf_out) / (norm(v_inf_in) * norm(v_inf_out)));
    delta_max = 2 * asin(1 / (1 + (R_p + HEIGHT_THRESHOLD) * (norm(v_inf_in).^2) / myu_flyby));

    dv_arrive = 0;
    if delta_des > delta_max
        v_p_in = sqrt(norm(v_inf_in).^2 + 2 * myu_flyby / (HEIGHT_THRESHOLD + R_p));
        v_p_out = sqrt(norm(v_inf_out).^2 + 2 * myu_flyby / (HEIGHT_THRESHOLD + R_p));
        d_delta = delta_des - delta_max;
        dv_arrive = sqrt(v_p_in.^2 + v_p_out.^2 - 2 * v_p_in * v_p_out * cos(d_delta));
    end


    dv_goal = norm(LAMBERT_GOAL - GOAL_STATE(4:6));
    dv_total = dv_launch + dv_arrive + dv_goal;
end