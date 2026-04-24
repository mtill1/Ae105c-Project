function [delta_v_launch, V_MARS_LEAVE, delta_v_mars, delta_v_A1_arrive, delta_v_A1_leave, ...
    delta_v_A2_arrive, delta_v_A2_leave, delta_v_A3_arrive, delta_v_total] ...
    = COMPUTE_MARS_PATH_DELTAV(A_ID_1, A_ID_2, A_ID_3, et_launch, et_mars, ...
    et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3, M_1, M_2, M_3, M_MARS)

    MINUTE = 60;
    HOUR = 60 * MINUTE;
    DAY = 24 * HOUR;

    EARTH_LAUNCH_STATE = cspice_spkezr('399', et_launch, 'ECLIPJ2000', 'NONE', '10');
    MARS_FLYBY_STATE = cspice_spkezr('4', et_mars, 'ECLIPJ2000', 'NONE', '10');
    A_1_ARRIVE_STATE = cspice_spkezr(A_ID_1, et_arrive_1, 'ECLIPJ2000', 'NONE', '10');
    A_1_LEAVING_STATE = cspice_spkezr(A_ID_1, et_stay_1, 'ECLIPJ2000', 'NONE', '10');
    A_2_ARRIVE_STATE = cspice_spkezr(A_ID_2, et_arrive_2, 'ECLIPJ2000', 'NONE', '10');
    A_2_LEAVING_STATE = cspice_spkezr(A_ID_2, et_stay_2, 'ECLIPJ2000', 'NONE', '10');
    A_3_ARRIVE_STATE = cspice_spkezr(A_ID_3, et_arrive_3, 'ECLIPJ2000', 'NONE', '10');

    MYU_SUN = cspice_bodvcd(10, 'GM', 10);
    MYU_MARS = cspice_bodvcd(4, 'GM', 10);
    R_p = cspice_bodvcd(499, 'RADII', 10);
    R_p = R_p(1);
    

    [EARTH_LAMBERT_VELOCITY, MARS_ARRIVE_VELOCITY, ~, EXIT_FLAG_1] = ...
        lambert(EARTH_LAUNCH_STATE(1:3)', MARS_FLYBY_STATE(1:3)', ...
        -(et_mars - et_launch) / DAY, M_MARS, MYU_SUN);



    [V_MARS_LEAVE, A_1_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_2] = ...
        lambert(MARS_FLYBY_STATE(1:3)', A_1_ARRIVE_STATE(1:3)', ...
        -(et_arrive_1 - et_mars) / DAY, M_1, MYU_SUN);
    
    [A_1_LEAVE_LAMBERT_VELOCITY, A_2_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_3] = ...
        lambert(A_1_LEAVING_STATE(1:3)', A_2_ARRIVE_STATE(1:3)', ...
        -(et_arrive_2 - et_stay_1) / DAY, M_2, MYU_SUN);
    [A_2_LEAVE_LAMBERT_VELOCITY, A_3_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_4] = ...
        lambert(A_2_LEAVING_STATE(1:3)', A_3_ARRIVE_STATE(1:3)', ...
        -(et_arrive_3 - et_stay_2) / DAY, M_3, MYU_SUN);

    if EXIT_FLAG_1 ~= 1 || EXIT_FLAG_2 ~= 1 || EXIT_FLAG_3 ~= 1 || EXIT_FLAG_4 ~= 1 
        %fprintf("LAMBERT SOLVER FAILED TO CONVERGE FOR ASTEROIDS WITH IDS %s, %s, %s.\n", ...
        %    A_ID_1, A_ID_2, A_ID_3);

        delta_v_launch = [];
        delta_v_A1_arrive = [];
        delta_v_A1_leave = [];
        delta_v_A2_arrive = [];
        delta_v_A2_leave = [];
        delta_v_A3_arrive = [];
        delta_v_total = 1e3;
        delta_v_mars = 1e3;
        return
        
    end

    delta_v_launch = EARTH_LAMBERT_VELOCITY' - EARTH_LAUNCH_STATE(4:6);
    delta_v_A1_arrive = A_1_ARRIVE_LAMBERT_VELOCITY' - A_1_ARRIVE_STATE(4:6);
    delta_v_A1_leave = A_1_LEAVE_LAMBERT_VELOCITY' - A_1_LEAVING_STATE(4:6);
    delta_v_A2_arrive = A_2_ARRIVE_LAMBERT_VELOCITY' - A_2_ARRIVE_STATE(4:6);
    delta_v_A2_leave = A_2_LEAVE_LAMBERT_VELOCITY' - A_2_LEAVING_STATE(4:6);
    delta_v_A3_arrive = A_3_ARRIVE_LAMBERT_VELOCITY' - A_3_ARRIVE_STATE(4:6);

    delta_v_mars = 0;

    v_inf_mars_in = MARS_ARRIVE_VELOCITY' - MARS_FLYBY_STATE(4:6);
    v_inf_mars_out = V_MARS_LEAVE' - MARS_FLYBY_STATE(4:6);
    
    
    HEIGHT_THRESHOLD = 200; %km
    
    delta_max = 2 * asin(1 / (1 + (R_p + HEIGHT_THRESHOLD) * (norm(v_inf_mars_in).^2) / MYU_MARS));
    delta_des = acos(dot(v_inf_mars_in, v_inf_mars_out) / (norm(v_inf_mars_in) * norm(v_inf_mars_out)));

    if delta_des > delta_max || norm(v_inf_mars_in) ~= norm(v_inf_mars_out)
        v_p_in = sqrt(dot(v_inf_mars_in, v_inf_mars_in) + 2 * MYU_MARS / (R_p + HEIGHT_THRESHOLD));
        v_p_out = sqrt(dot(v_inf_mars_out, v_inf_mars_out) + 2 * MYU_MARS / (R_p + HEIGHT_THRESHOLD));
        
        delta_v_mars = v_p_out - v_p_in;
    end
    

    delta_v_total = norm(delta_v_launch) + norm(delta_v_A1_arrive) + ...
        norm(delta_v_A1_leave) + norm(delta_v_A2_arrive) + ...
        norm(delta_v_A2_leave) + norm(delta_v_A3_arrive) + abs(delta_v_mars);


end