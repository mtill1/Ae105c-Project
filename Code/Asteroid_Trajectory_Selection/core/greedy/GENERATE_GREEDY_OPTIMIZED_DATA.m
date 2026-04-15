function asteroid_optimized_data = GENERATE_GREEDY_OPTIMIZED_DATA(asteroid_list, ...
    M, launch_utc_min, launch_utc_max, save_filename)
    wb = waitbar(0, 'Starting Optimization...', 'Name', save_filename);

    NUM_ASTEROIDS = length(asteroid_list);
    TOTAL_OPERATIONS = NUM_ASTEROIDS.^3;

    elapsed_times = zeros(NUM_ASTEROIDS.^3, 1);

    asteroid_optimized_data(NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS, 3) = ...
        struct("dv_launch", Inf, ...
            "dv_arrive", Inf, "dv_goal", Inf, "dv_total", Inf, ...
            "LAMBERT_LAUNCH", Inf, "LAMBERT_ARRIVE_IN", Inf, ...
            "LAMBERT_ARRIVE_OUT", Inf, "LAMBERT_GOAL", Inf, ...
            "FLYBY_BODY", Inf, "et_launch", Inf, ...
            "et_flyby", Inf, "et_goal", Inf);

    ET_LAUNCH_MIN = cspice_str2et(launch_utc_min);
    ET_LAUNCH_MAX = cspice_str2et(launch_utc_max);

    LAUNCH_DATES = [ET_LAUNCH_MIN, ET_LAUNCH_MAX];

    for i = 1:NUM_ASTEROIDS
        for j = 1:NUM_ASTEROIDS
            for k = 1:NUM_ASTEROIDS
                current_operation = k + (j-1) * NUM_ASTEROIDS + (i-1) * NUM_ASTEROIDS.^2;
                tic

                A_ID_1 = (asteroid_list(i).ID);
                A_ID_2 = (asteroid_list(j).ID);
                A_ID_3 = (asteroid_list(k).ID);

                if A_ID_1 == A_ID_2 || A_ID_2 == A_ID_3 || A_ID_3 == A_ID_1
                    elapsed_times(current_operation) = toc;
                    continue
                end

                A_ID_1 = int2str(asteroid_list(i).ID);
                A_ID_2 = int2str(asteroid_list(j).ID);
                A_ID_3 = int2str(asteroid_list(k).ID);
                
                asteroid_optimized_data(i, j, k, :) = ...
                    OPTIMIZE_GREEDY_TIMES(A_ID_1, A_ID_2, A_ID_3, ...
                    LAUNCH_DATES, M);

                elapsed_times(current_operation) = toc;
                ETA = (sum(elapsed_times) * TOTAL_OPERATIONS / current_operation) - sum(elapsed_times);

                progress = current_operation / TOTAL_OPERATIONS;
                waitbar(progress, wb, sprintf('Operation %d of %d (ETA: %.2f minutes)', ...
                    current_operation, TOTAL_OPERATIONS, ETA/60));
            end
        end
    end

    close(wb);

    outDir = fullfile(getRepoRoot(), 'outputs', 'greedy');
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    timestampTag = datestr(now, 'yyyymmdd_HHMMSS');
    save(fullfile(outDir, sprintf('%s_%s.mat', save_filename, timestampTag)), "asteroid_optimized_data");
end
