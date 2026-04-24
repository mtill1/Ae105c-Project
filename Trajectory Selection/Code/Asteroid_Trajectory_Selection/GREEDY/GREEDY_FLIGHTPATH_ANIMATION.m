function GREEDY_FLIGHTPATH_ANIMATION(PATH_DEFINED_VECTOR, asteroid_list, A_INDEX_1, A_INDEX_2, A_INDEX_3, t_duration, OUTPUT_VIDEO_NAME)
    % date_launch = cspice_et2utc(PATH_DEFINED_VECTOR.et_launch, 'C', 6);
    
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

    MISSION_TIME = PATH_DEFINED_VECTOR(1).et_launch:1*WEEK:PATH_DEFINED_VECTOR(3).et_goal;
    
    t_DIFF_1_1 = PATH_DEFINED_VECTOR(1).et_flyby - PATH_DEFINED_VECTOR(1).et_launch;
    t_DIFF_1_2 = PATH_DEFINED_VECTOR(1).et_goal - PATH_DEFINED_VECTOR(1).et_flyby;

    t_DIFF_2_1 = PATH_DEFINED_VECTOR(2).et_flyby - PATH_DEFINED_VECTOR(2).et_launch;
    t_DIFF_2_2 = PATH_DEFINED_VECTOR(2).et_goal - PATH_DEFINED_VECTOR(2).et_flyby;

    t_DIFF_3_1 = PATH_DEFINED_VECTOR(3).et_flyby - PATH_DEFINED_VECTOR(3).et_launch;
    t_DIFF_3_2 = PATH_DEFINED_VECTOR(3).et_goal - PATH_DEFINED_VECTOR(3).et_flyby;

    
    
    MYU_SUN = cspice_bodvcd(10, 'GM', 10);

    EARTH_LAUNCH_STATE = cspice_spkezr('399', PATH_DEFINED_VECTOR(1).et_launch, ...
        'ECLIPJ2000', 'NONE', '10');
    A1_STATE = cspice_spkezr(A_ID_1, PATH_DEFINED_VECTOR(2).et_launch, ...
        'ECLIPJ2000', 'NONE', '10');
    A2_STATE = cspice_spkezr(A_ID_2, PATH_DEFINED_VECTOR(3).et_launch, ...
        'ECLIPJ2000', 'NONE', '10');
    

    x_0 = [EARTH_LAUNCH_STATE(1:3); ...
        PATH_DEFINED_VECTOR(1).LAMBERT_LAUNCH'];
    [X_1_1, T_1_1] = TWO_BODY_SIM(t_DIFF_1_1, x_0, MYU_SUN);

    if ~isempty(PATH_DEFINED_VECTOR(1).LAMBERT_ARRIVE_OUT)
        x_0 = [X_1_1(end, 1:3), PATH_DEFINED_VECTOR(1).LAMBERT_ARRIVE_OUT]';
        [X_1_2, T_1_2] = TWO_BODY_SIM(t_DIFF_1_2, x_0, MYU_SUN);
    else
        x_0 = X_1_1(end, 1:6);
        [X_1_2, T_1_2] = TWO_BODY_SIM(t_DIFF_1_2, x_0, MYU_SUN);
    end


    x_0 = [A1_STATE(1:3); ...
        PATH_DEFINED_VECTOR(2).LAMBERT_LAUNCH'];
    [X_2_1, T_2_1] = TWO_BODY_SIM(t_DIFF_2_1, x_0, MYU_SUN);

    if ~isempty(PATH_DEFINED_VECTOR(2).LAMBERT_ARRIVE_OUT)
        x_0 = [X_2_1(end, 1:3), PATH_DEFINED_VECTOR(2).LAMBERT_ARRIVE_OUT]';
        [X_2_2, T_2_2] = TWO_BODY_SIM(t_DIFF_2_2, x_0, MYU_SUN);
    else
        x_0 = X_2_1(end, 1:6);
        [X_2_2, T_2_2] = TWO_BODY_SIM(t_DIFF_2_2, x_0, MYU_SUN);
    end

    x_0 = [A2_STATE(1:3); ...
        PATH_DEFINED_VECTOR(3).LAMBERT_LAUNCH'];
    [X_3_1, T_3_1] = TWO_BODY_SIM(t_DIFF_3_1, x_0, MYU_SUN);

    if ~isempty(PATH_DEFINED_VECTOR(3).LAMBERT_ARRIVE_OUT)
        x_0 = [X_3_1(end, 1:3), PATH_DEFINED_VECTOR(3).LAMBERT_ARRIVE_OUT]';
        [X_3_2, T_3_2] = TWO_BODY_SIM(t_DIFF_3_2, x_0, MYU_SUN);
    else
        x_0 = X_3_1(end, 1:6);
        [X_3_2, T_3_2] = TWO_BODY_SIM(t_DIFF_3_2, x_0, MYU_SUN);
    end

    
    

    N = 2 * (length(T_1_1) + length(T_1_2)) + (length(T_2_1) + length(T_2_2)) ...
        + (length(T_3_1) + length(T_3_2));

    FPS = N / t_duration;
    
    v = VideoWriter(OUTPUT_VIDEO_NAME, 'MPEG-4');
    v.FrameRate = FPS;
    open(v);

    h = figure;
    h.WindowState = 'maximized';

    wb = waitbar(0, 'Starting video encoding...', 'Name', 'Progress');

    ANIMATE_SECTION(MISSION_TIME, X_1_1, T_1_1, PATH_DEFINED_VECTOR(1).et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);

    ANIMATE_SECTION(MISSION_TIME, X_1_2, T_1_2, PATH_DEFINED_VECTOR(1).et_flyby, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);
    
    ANIMATE_STAY_PATHS(MISSION_TIME, 1, length(T_1_1) + length(T_1_2), ...
        PATH_DEFINED_VECTOR(1).et_goal, PATH_DEFINED_VECTOR(2).et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);

    ANIMATE_SECTION(MISSION_TIME, X_2_1, T_2_1, PATH_DEFINED_VECTOR(2).et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);

    ANIMATE_SECTION(MISSION_TIME, X_2_2, T_2_2, PATH_DEFINED_VECTOR(2).et_flyby, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);
    
    ANIMATE_STAY_PATHS(MISSION_TIME, 2, length(T_2_1) + length(T_2_2), ...
        PATH_DEFINED_VECTOR(2).et_goal, PATH_DEFINED_VECTOR(3).et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);

    ANIMATE_SECTION(MISSION_TIME, X_3_1, T_3_1, PATH_DEFINED_VECTOR(3).et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);

    ANIMATE_SECTION(MISSION_TIME, X_3_2, T_3_2, PATH_DEFINED_VECTOR(3).et_flyby, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb);
    
    close(v);
    close(wb);
end


function ANIMATE_SECTION(MISSION_TIME, X_i, T_i, et_launch, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb)
    EARTH_CSPICE = cspice_spkezr('399', MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A1_CSPICE = cspice_spkezr(A_ID_1, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A2_CSPICE = cspice_spkezr(A_ID_2, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A3_CSPICE = cspice_spkezr(A_ID_3, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    MARS_CSPICE = cspice_spkezr('4', MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');

    for i = 1:length(T_i)
        clf;
        hold on;
        grid minor;
        
        plot3(EARTH_CSPICE(1, :), EARTH_CSPICE(2, :), EARTH_CSPICE(3, :), "DisplayName", 'Earth', 'Color', 'cyan');
        plot3(MARS_CSPICE(1, :), MARS_CSPICE(2, :), MARS_CSPICE(3, :), "DisplayName", 'Mars', 'Color', 'magenta');
        
        plot3(A1_CSPICE(1, :), A1_CSPICE(2, :), A1_CSPICE(3, :), "DisplayName", A_NAME_1, 'Color', 'r');
        plot3(A2_CSPICE(1, :), A2_CSPICE(2, :), A2_CSPICE(3, :), "DisplayName", A_NAME_2, 'Color', 'g');
        plot3(A3_CSPICE(1, :), A3_CSPICE(2, :), A3_CSPICE(3, :), "DisplayName", A_NAME_3, 'Color', 'b');


        EARTH_CSPICE_CURR = cspice_spkezr('399', T_i(i) + et_launch, ...
            'ECLIPJ2000', 'NONE', '10');
        MARS_CSPICE_CURR = cspice_spkezr('4', T_i(i) + et_launch, ...
            'ECLIPJ2000', 'NONE', '10');
        
        A1_CSPICE_CURR = cspice_spkezr(A_ID_1, T_i(i) + et_launch, ...
            'ECLIPJ2000', 'NONE', '10');
        A2_CSPICE_CURR = cspice_spkezr(A_ID_2, T_i(i) + et_launch, ...
            'ECLIPJ2000', 'NONE', '10');
        A3_CSPICE_CURR = cspice_spkezr(A_ID_3, T_i(i) + et_launch, ...
            'ECLIPJ2000', 'NONE', '10');


        scatter3(EARTH_CSPICE_CURR(1), EARTH_CSPICE_CURR(2), EARTH_CSPICE_CURR(3), 'cyan', 'filled', "HandleVisibility", "off");
        scatter3(MARS_CSPICE_CURR(1), MARS_CSPICE_CURR(2), MARS_CSPICE_CURR(3), 'magenta', 'filled', "HandleVisibility", "off");
        
        scatter3(A1_CSPICE_CURR(1), A1_CSPICE_CURR(2), A1_CSPICE_CURR(3), 'r', 'filled', "HandleVisibility", "off");
        scatter3(A2_CSPICE_CURR(1), A2_CSPICE_CURR(2), A2_CSPICE_CURR(3), 'g', 'filled', "HandleVisibility", "off");
        scatter3(A3_CSPICE_CURR(1), A3_CSPICE_CURR(2), A3_CSPICE_CURR(3), 'b', 'filled', "HandleVisibility", "off");

        scatter3(X_i(i, 1), X_i(i, 2), X_i(i, 3), 'w', 'filled', "HandleVisibility", 'off');

        plot3(X_i(:, 1), X_i(:, 2), X_i(:, 3), 'DisplayName', 'Trajectory', 'Color', 'w');


        legend()
        view(3);
    

        title(cspice_et2utc(T_i(i) + et_launch, 'C', 6));

        hold off;
        drawnow;

        writeVideo(v, getframe(h));
        progress = i / length(T_i);
        waitbar(progress, wb, sprintf('SECTION: Encoding Frame %d of %d', i, length(T_i)));
    end
    
end

function ANIMATE_STAY_PATHS(MISSION_TIME, ASTEROID_STAY_NUMBER, N_time, et_start, et_stop, ...
    A_ID_1, A_ID_2, A_ID_3, A_NAME_1, A_NAME_2, A_NAME_3, ...
    h, v, wb)
    EARTH_CSPICE = cspice_spkezr('399', MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A1_CSPICE = cspice_spkezr(A_ID_1, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A2_CSPICE = cspice_spkezr(A_ID_2, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');
    A3_CSPICE = cspice_spkezr(A_ID_3, MISSION_TIME, ...
        'ECLIPJ2000', 'NONE', '10');

    t_RANGE = linspace(et_start, et_stop, N_time);

    A_STAY_CSPICE = cspice_spkezr(A_ID_1, t_RANGE, ...
        'ECLIPJ2000', 'NONE', '10');
    if ASTEROID_STAY_NUMBER == 2
        A_STAY_CSPICE = cspice_spkezr(A_ID_2, t_RANGE, ...
            'ECLIPJ2000', 'NONE', '10');
    end

    

    for i = 1:N_time
        clf;
        hold on;
        grid minor;
        
        plot3(EARTH_CSPICE(1, :), EARTH_CSPICE(2, :), EARTH_CSPICE(3, :), "DisplayName", 'Earth', 'Color', 'cyan');
        plot3(A1_CSPICE(1, :), A1_CSPICE(2, :), A1_CSPICE(3, :), "DisplayName", A_NAME_1, 'Color', 'r');
        plot3(A2_CSPICE(1, :), A2_CSPICE(2, :), A2_CSPICE(3, :), "DisplayName", A_NAME_2, 'Color', 'g');
        plot3(A3_CSPICE(1, :), A3_CSPICE(2, :), A3_CSPICE(3, :), "DisplayName", A_NAME_3, 'Color', 'b');


        EARTH_CSPICE_CURR = cspice_spkezr('399', t_RANGE(i), ...
            'ECLIPJ2000', 'NONE', '10');
        A1_CSPICE_CURR = cspice_spkezr(A_ID_1, t_RANGE(i), ...
            'ECLIPJ2000', 'NONE', '10');
        A2_CSPICE_CURR = cspice_spkezr(A_ID_2, t_RANGE(i), ...
            'ECLIPJ2000', 'NONE', '10');
        A3_CSPICE_CURR = cspice_spkezr(A_ID_3, t_RANGE(i), ...
            'ECLIPJ2000', 'NONE', '10');


        scatter3(EARTH_CSPICE_CURR(1), EARTH_CSPICE_CURR(2), EARTH_CSPICE_CURR(3), 'cyan', 'filled', "HandleVisibility", "off");
        scatter3(A1_CSPICE_CURR(1), A1_CSPICE_CURR(2), A1_CSPICE_CURR(3), 'r', 'filled', "HandleVisibility", "off");
        scatter3(A2_CSPICE_CURR(1), A2_CSPICE_CURR(2), A2_CSPICE_CURR(3), 'g', 'filled', "HandleVisibility", "off");
        scatter3(A3_CSPICE_CURR(1), A3_CSPICE_CURR(2), A3_CSPICE_CURR(3), 'b', 'filled', "HandleVisibility", "off");
        
        plot3(A_STAY_CSPICE(1, :), A_STAY_CSPICE(2, :), A_STAY_CSPICE(3, :), 'DisplayName', 'Trajectory', 'Color', 'w');

        legend()
        view(3);
    

        title(cspice_et2utc(t_RANGE(i), 'C', 6));

        hold off;
        drawnow;

        writeVideo(v, getframe(h));
        progress = i / N_time;
        waitbar(progress, wb, sprintf('SECTION: Encoding Frame %d of %d', i, N_time));
    end
    
end