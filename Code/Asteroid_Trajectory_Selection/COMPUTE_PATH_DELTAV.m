function [delta_v_launch, delta_v_A1_arrive, delta_v_A1_leave, ...
    delta_v_A2_arrive, delta_v_A2_leave, delta_v_A3_arrive, delta_v_total] ...
    = COMPUTE_PATH_DELTAV(A_ID_1, A_ID_2, A_ID_3, et_launch, ...
    et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3, M_1, M_2, M_3)

    MINUTE = 60;
    HOUR = 60 * MINUTE;
    DAY = 24 * HOUR;

    EARTH_LAUNCH_STATE = cspice_spkezr('399', et_launch, 'ECLIPJ2000', 'NONE', '10');
    A_1_ARRIVE_STATE = cspice_spkezr(A_ID_1, et_arrive_1, 'ECLIPJ2000', 'NONE', '10');
    A_1_LEAVING_STATE = cspice_spkezr(A_ID_1, et_stay_1, 'ECLIPJ2000', 'NONE', '10');
    A_2_ARRIVE_STATE = cspice_spkezr(A_ID_2, et_arrive_2, 'ECLIPJ2000', 'NONE', '10');
    A_2_LEAVING_STATE = cspice_spkezr(A_ID_2, et_stay_2, 'ECLIPJ2000', 'NONE', '10');
    A_3_ARRIVE_STATE = cspice_spkezr(A_ID_3, et_arrive_3, 'ECLIPJ2000', 'NONE', '10');

    MYU_SUN = cspice_bodvcd(10, 'GM', 10);

    [EARTH_LAMBERT_VELOCITY, A_1_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_1] = ...
        lambert(EARTH_LAUNCH_STATE(1:3)', A_1_ARRIVE_STATE(1:3)', ...
        (et_arrive_1 - et_launch) / DAY, M_1, MYU_SUN);
    [A_1_LEAVE_LAMBERT_VELOCITY, A_2_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_2] = ...
        lambert(A_1_LEAVING_STATE(1:3)', A_2_ARRIVE_STATE(1:3)', ...
        (et_arrive_2 - et_stay_1) / DAY, M_2, MYU_SUN);
    [A_2_LEAVE_LAMBERT_VELOCITY, A_3_ARRIVE_LAMBERT_VELOCITY, ~, EXIT_FLAG_3] = ...
        lambert(A_2_LEAVING_STATE(1:3)', A_3_ARRIVE_STATE(1:3)', ...
        (et_arrive_3 - et_stay_2) / DAY, M_3, MYU_SUN);

    if EXIT_FLAG_1 ~= 1 || EXIT_FLAG_2 ~= 1 || EXIT_FLAG_3 ~= 1
        fprintf("LAMBERT SOLVER FAILED TO CONVERGE FOR ASTEROIDS WITH IDS %s, %s, %s.\n", ...
            A_ID_1, A_ID_2, A_ID_3);
        delta_v_launch = -1;
        delta_v_A1_arrive = -1;
        delta_v_A1_leave = -1;
        delta_v_A2_arrive = -1;
        delta_v_A2_leave = -1;
        delta_v_A3_arrive = -1;
        delta_v_total = -1;
        return
        
    end

    delta_v_launch = EARTH_LAMBERT_VELOCITY' - EARTH_LAUNCH_STATE(4:6);
    delta_v_A1_arrive = A_1_ARRIVE_LAMBERT_VELOCITY' - A_1_ARRIVE_STATE(4:6);
    delta_v_A1_leave = A_1_LEAVE_LAMBERT_VELOCITY' - A_1_LEAVING_STATE(4:6);
    delta_v_A2_arrive = A_2_ARRIVE_LAMBERT_VELOCITY' - A_2_ARRIVE_STATE(4:6);
    delta_v_A2_leave = A_2_LEAVE_LAMBERT_VELOCITY' - A_2_LEAVING_STATE(4:6);
    delta_v_A3_arrive = A_3_ARRIVE_LAMBERT_VELOCITY' - A_3_ARRIVE_STATE(4:6);
    

    delta_v_total = norm(delta_v_launch) + norm(delta_v_A1_arrive) + ...
        norm(delta_v_A1_leave) + norm(delta_v_A2_arrive) + ...
        norm(delta_v_A2_leave) + norm(delta_v_A3_arrive);



end