%struct("delta_v_launch", delta_v_launch, ...
%        "delta_v_A1_arrive", delta_v_A1_arrive, "delta_v_A1_leave", delta_v_A1_leave, ...
%        "delta_v_A2_arrive", delta_v_A2_arrive, "delta_v_A2_leave", delta_v_A2_leave, ...
%        "delta_v_A3_arrive", delta_v_A3_arrive, "delta_v_total", delta_v_total, ...
%        "et_launch", et_launch, "et_arrive_1", et_arrive_1, "et_stay_1", et_stay_1, ...
%        "et_arrive_2", et_arrive_2, "et_stay_2", et_stay_2, ...
%        "et_arrive_3", et_arrive_3)


function PlotDirectFlightpathSegments(PATH_DEFINED_VECTOR, A_ID_1, A_ID_2)
    t_1 = PATH_DEFINED_VECTOR.et_arrive_1 - PATH_DEFINED_VECTOR.et_launch;
    t_2 = PATH_DEFINED_VECTOR.et_arrive_2 - PATH_DEFINED_VECTOR.et_stay_1;
    t_3 = PATH_DEFINED_VECTOR.et_arrive_3 - PATH_DEFINED_VECTOR.et_stay_2;
    
    MYU_SUN = cspice_bodvcd(10, 'GM', 10);
    EARTH_LAUNCH_STATE = cspice_spkezr('399', PATH_DEFINED_VECTOR.et_launch, ...
        'ECLIPJ2000', 'NONE', '10');

    x_0 = [EARTH_LAUNCH_STATE(1:3); ...
        EARTH_LAUNCH_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_launch];
    [X_1, T_1] = TWO_BODY_SIM(t_1, x_0, MYU_SUN)

    A_1_LEAVING_STATE = cspice_spkezr(A_ID_1, PATH_DEFINED_VECTOR.et_stay_1, ...
        'ECLIPJ2000', 'NONE', '10');

    x_0 = [A_1_LEAVING_STATE(1:3); ...
        A_1_LEAVING_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_A1_leave];
    [X_2, T_2] = TWO_BODY_SIM(t_2, x_0, MYU_SUN)

    A_2_LEAVING_STATE = cspice_spkezr(A_ID_2, PATH_DEFINED_VECTOR.et_stay_2, ...
        'ECLIPJ2000', 'NONE', '10');

    x_0 = [A_2_LEAVING_STATE(1:3);...
        A_2_LEAVING_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_A2_leave];
    [X_3, T_3] = TWO_BODY_SIM(t_3, x_0, MYU_SUN)

    clf;

    hold on;
    grid minor;
    plot3(X_1(:,1), X_1(:,2), X_1(:,3), 'r', 'DisplayName', 'Trajectory to A1');
    plot3(X_2(:,1), X_2(:,2), X_2(:,3), 'g', 'DisplayName', 'Trajectory from A1 to A2');
    plot3(X_3(:,1), X_3(:,2), X_3(:,3), 'b', 'DisplayName', 'Trajectory from A2');
    legend show;
    view(3);
    lockSpatialPlotScaling(gca, 'ecliptic3d');
    title('ECLIPJ2000 heliocentric trajectory segments');
    xlabel('X (km)');
    ylabel('Y (km)');
    zlabel('Z (km)');
    grid on;

    hold off;
end

