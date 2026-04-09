function GRAPH_ASTEROIDS(asteroid_list, t_DURATION, FPS, START_DATE, END_DATE, OUTPUT_VIDEO_NAME)
    et0   = cspice_str2et(START_DATE);
    etf   = cspice_str2et(END_DATE);

    MINUTE = 60;
    HOUR = 60 * MINUTE;
    DAY = 24 * HOUR;
    WEEK = 7 * DAY;
    t_RANGE = et0:4*WEEK:etf;

    N = FPS * t_DURATION;
    K = (length(t_RANGE)/N);
    
    v = VideoWriter(OUTPUT_VIDEO_NAME, 'MPEG-4');
    v.FrameRate = FPS;
    open(v);

    asteroid_cspices = zeros(3, length(asteroid_list), length(t_RANGE));

    for j = 1:length(asteroid_list)
        asteroid_cspice = cspice_spkezr(int2str(asteroid_list(j).ID), t_RANGE, 'ECLIPJ2000', 'NONE', '0');
        asteroid_cspices(:, j, :) = asteroid_cspice(1:3, :);
    end

    h = figure;
    h.WindowState = 'maximized';

    NUM_ASTEROIDS = length(asteroid_list);
    COLORS = hsv(NUM_ASTEROIDS); 

    wb = waitbar(0, 'Starting video encoding...', 'Name', 'Progress');

    for i = 1:N
        clf;
        hold on;
        grid minor;

        position_list = zeros(3, length(asteroid_list));

        for j = 1:length(asteroid_list)
            plot3(squeeze(asteroid_cspices(1, j, :)),...
                squeeze(asteroid_cspices(2, j, :)),...
                squeeze(asteroid_cspices(3, j, :)), ...
                'DisplayName', asteroid_list(j).NAME, ...
                "Color", COLORS(j, :))

            position_list(:, j) = asteroid_cspices(1:3, j, ceil(i * K));
        
        end

        scatter3(position_list(1, :), position_list(2, :), position_list(3, :), 60, COLORS, ...
            "filled", "HandleVisibility", "off");

        title(['Asteroid Trajectories at ', cspice_et2utc(t_RANGE(ceil(i * K)), 'C', 1)]);
        xlabel('X Position (km)');
        ylabel('Y Position (km)');
        zlabel('Z Position (km)');
    
        legend()
        view(3);
    
        hold off;
        drawnow;
        writeVideo(v, getframe(h))
        progress = i / N;
        waitbar(progress, wb, sprintf('Encoding Frame %d of %d', i, N));
    end

    close(v);
    close(wb);

end