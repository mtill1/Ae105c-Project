function AnimateMarsFlightpath(PATH_DEFINED_VECTOR, asteroid_list, A_INDEX_1, A_INDEX_2, A_INDEX_3, t_duration, OUTPUT_VIDEO_NAME)
% AnimateMarsFlightpath
% Syntax:
%   AnimateMarsFlightpath(PATH_DEFINED_VECTOR, asteroid_list, A_INDEX_1, A_INDEX_2, A_INDEX_3, t_duration, OUTPUT_VIDEO_NAME)
% Description:
%   Animates a Mars-assist asteroid tour: Earth->Mars->A1->A2->A3.
% Inputs/Outputs:
%   Inputs as shown in Syntax. Writes MP4 video to OUTPUT_VIDEO_NAME.

    A_ID_1 = int2str(asteroid_list(A_INDEX_1).ID);
    A_ID_2 = int2str(asteroid_list(A_INDEX_2).ID);
    A_ID_3 = int2str(asteroid_list(A_INDEX_3).ID);

    A_NAME_1 = asteroid_list(A_INDEX_1).NAME;
    A_NAME_2 = asteroid_list(A_INDEX_2).NAME;
    A_NAME_3 = asteroid_list(A_INDEX_3).NAME;

    MINUTE = 60;
    HOUR = 60 * MINUTE;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;

    MISSION_TIME = PATH_DEFINED_VECTOR.et_launch:2*WEEK:PATH_DEFINED_VECTOR.et_arrive_3;

    t_DIFF_0 = PATH_DEFINED_VECTOR.et_mars - PATH_DEFINED_VECTOR.et_launch;
    t_DIFF_1 = PATH_DEFINED_VECTOR.et_arrive_1 - PATH_DEFINED_VECTOR.et_mars;
    t_DIFF_2 = PATH_DEFINED_VECTOR.et_arrive_2 - PATH_DEFINED_VECTOR.et_stay_1;
    t_DIFF_3 = PATH_DEFINED_VECTOR.et_arrive_3 - PATH_DEFINED_VECTOR.et_stay_2;

    MYU_SUN = cspice_bodvcd(10, 'GM', 10);

    EARTH_LAUNCH_STATE = cspice_spkezr('399', PATH_DEFINED_VECTOR.et_launch, 'ECLIPJ2000', 'NONE', '10');
    MARS_STATE = cspice_spkezr('4', PATH_DEFINED_VECTOR.et_mars, 'ECLIPJ2000', 'NONE', '10');
    A_1_LEAVING_STATE = cspice_spkezr(A_ID_1, PATH_DEFINED_VECTOR.et_stay_1, 'ECLIPJ2000', 'NONE', '10');
    A_2_LEAVING_STATE = cspice_spkezr(A_ID_2, PATH_DEFINED_VECTOR.et_stay_2, 'ECLIPJ2000', 'NONE', '10');

    x_0 = [EARTH_LAUNCH_STATE(1:3); EARTH_LAUNCH_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_launch];
    [X_0, T_0] = TWO_BODY_SIM(t_DIFF_0, x_0, MYU_SUN);

    x_0 = [MARS_STATE(1:3); PATH_DEFINED_VECTOR.v_mars_leave'];
    [X_1, T_1] = TWO_BODY_SIM(t_DIFF_1, x_0, MYU_SUN);

    x_0 = [A_1_LEAVING_STATE(1:3); A_1_LEAVING_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_A1_leave];
    [X_2, T_2] = TWO_BODY_SIM(t_DIFF_2, x_0, MYU_SUN);

    x_0 = [A_2_LEAVING_STATE(1:3); A_2_LEAVING_STATE(4:6) + PATH_DEFINED_VECTOR.delta_v_A2_leave];
    [X_3, T_3] = TWO_BODY_SIM(t_DIFF_3, x_0, MYU_SUN);

    N = length(X_0) + length(X_1) + length(X_2) + length(X_3);
    FPS = N / t_duration;

    v = VideoWriter(OUTPUT_VIDEO_NAME, 'MPEG-4');
    v.FrameRate = FPS;
    open(v);

    videoFigPos = [100 100 1280 720];
    h = figure('Visible', 'on', 'Color', 'w');
    setappdata(h, 'VideoCaptureFigPos', videoFigPos);
    prepareVideoCaptureFigure(h, videoFigPos);
    wb = waitbar(0, 'Starting video encoding...', 'Name', 'Progress');

    ANIMATE_SECTION(MISSION_TIME, X_0, T_0, PATH_DEFINED_VECTOR.et_launch, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);
    ANIMATE_SECTION(MISSION_TIME, X_1, T_1, PATH_DEFINED_VECTOR.et_mars, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);
    ANIMATE_STAY_PATHS(MISSION_TIME, 1, length(T_2), PATH_DEFINED_VECTOR.et_arrive_1, PATH_DEFINED_VECTOR.et_stay_1, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);
    ANIMATE_SECTION(MISSION_TIME, X_2, T_2, PATH_DEFINED_VECTOR.et_stay_1, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);
    ANIMATE_STAY_PATHS(MISSION_TIME, 2, length(T_3), PATH_DEFINED_VECTOR.et_arrive_2, PATH_DEFINED_VECTOR.et_stay_2, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);
    ANIMATE_SECTION(MISSION_TIME, X_3, T_3, PATH_DEFINED_VECTOR.et_stay_2, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb);

    close(v);
    close(wb);
end

function ANIMATE_SECTION(MISSION_TIME, X_i, T_i, et_launch, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb)
    EARTH_CSPICE = cspice_spkezr('399', MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    MARS_CSPICE = cspice_spkezr('4', MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A1_CSPICE = cspice_spkezr(A_ID_1, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A2_CSPICE = cspice_spkezr(A_ID_2, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A3_CSPICE = cspice_spkezr(A_ID_3, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');

    for i = 1:length(T_i)
        prepareVideoCaptureFigure(h, getappdata(h, 'VideoCaptureFigPos'));
        hold on; grid on; grid minor;
        plot3(EARTH_CSPICE(1,:), EARTH_CSPICE(2,:), EARTH_CSPICE(3,:), 'DisplayName', 'Earth', 'Color', 'cyan');
        plot3(MARS_CSPICE(1,:), MARS_CSPICE(2,:), MARS_CSPICE(3,:), 'DisplayName', 'Mars', 'Color', 'magenta');
        plot3(A1_CSPICE(1,:), A1_CSPICE(2,:), A1_CSPICE(3,:), 'DisplayName', A_NAME_1, 'Color', 'r');
        plot3(A2_CSPICE(1,:), A2_CSPICE(2,:), A2_CSPICE(3,:), 'DisplayName', A_NAME_2, 'Color', 'g');
        plot3(A3_CSPICE(1,:), A3_CSPICE(2,:), A3_CSPICE(3,:), 'DisplayName', A_NAME_3, 'Color', 'b');

        scatter3(X_i(i,1), X_i(i,2), X_i(i,3), 'w', 'filled', 'HandleVisibility', 'off');
        plot3(X_i(:,1), X_i(:,2), X_i(:,3), 'DisplayName', 'Trajectory', 'Color', 'w');

        legend(); view(3); lockSpatialPlotScaling(gca, 'ecliptic3d');
        title(cspice_et2utc(T_i(i) + et_launch, 'C', 6));
        xlabel('X (km)'); ylabel('Y (km)'); zlabel('Z (km)');
        hold off; drawnow;
        writeVideo(v, getframe(h));
        waitbar(i / length(T_i), wb, sprintf('SECTION: Encoding Frame %d of %d', i, length(T_i)));
    end
end

function ANIMATE_STAY_PATHS(MISSION_TIME, ASTEROID_STAY_NUMBER, N_time, et_start, et_stop, A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, h, v, wb)
    EARTH_CSPICE = cspice_spkezr('399', MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    MARS_CSPICE = cspice_spkezr('4', MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A1_CSPICE = cspice_spkezr(A_ID_1, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A2_CSPICE = cspice_spkezr(A_ID_2, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');
    A3_CSPICE = cspice_spkezr(A_ID_3, MISSION_TIME, 'ECLIPJ2000', 'NONE', '10');

    t_RANGE = linspace(et_start, et_stop, N_time);
    A_STAY_CSPICE = cspice_spkezr(A_ID_1, t_RANGE, 'ECLIPJ2000', 'NONE', '10');
    if ASTEROID_STAY_NUMBER == 2
        A_STAY_CSPICE = cspice_spkezr(A_ID_2, t_RANGE, 'ECLIPJ2000', 'NONE', '10');
    end

    for i = 1:N_time
        prepareVideoCaptureFigure(h, getappdata(h, 'VideoCaptureFigPos'));
        hold on; grid on; grid minor;
        plot3(EARTH_CSPICE(1,:), EARTH_CSPICE(2,:), EARTH_CSPICE(3,:), 'DisplayName', 'Earth', 'Color', 'cyan');
        plot3(MARS_CSPICE(1,:), MARS_CSPICE(2,:), MARS_CSPICE(3,:), 'DisplayName', 'Mars', 'Color', 'magenta');
        plot3(A1_CSPICE(1,:), A1_CSPICE(2,:), A1_CSPICE(3,:), 'DisplayName', A_NAME_1, 'Color', 'r');
        plot3(A2_CSPICE(1,:), A2_CSPICE(2,:), A2_CSPICE(3,:), 'DisplayName', A_NAME_2, 'Color', 'g');
        plot3(A3_CSPICE(1,:), A3_CSPICE(2,:), A3_CSPICE(3,:), 'DisplayName', A_NAME_3, 'Color', 'b');
        plot3(A_STAY_CSPICE(1,:), A_STAY_CSPICE(2,:), A_STAY_CSPICE(3,:), 'DisplayName', 'Trajectory', 'Color', 'w');

        legend(); view(3); lockSpatialPlotScaling(gca, 'ecliptic3d');
        title(cspice_et2utc(t_RANGE(i), 'C', 6));
        xlabel('X (km)'); ylabel('Y (km)'); zlabel('Z (km)');
        hold off; drawnow;
        writeVideo(v, getframe(h));
        waitbar(i / N_time, wb, sprintf('SECTION: Encoding Frame %d of %d', i, N_time));
    end
end

