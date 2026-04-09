function GENERATE_OPTIMIZED_DATA(asteroid_list, m_1, m_2, m_3, launch_utc_min, launch_utc_max)
    wb = waitbar(0, 'Starting Optimization...', 'Name', sprintf('Ms: %d, %d, %d', m_1, m_2, m_3));

    NUM_ASTEROIDS = length(asteroid_list);
    TOTAL_OPERATIONS = NUM_ASTEROIDS.^3;

    elapsed_times = zeros(NUM_ASTEROIDS.^3, 1);

    asteroid_optimized_data(NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS) =...
        struct("delta_v_launch", Inf, ...
        "delta_v_A1_arrive", Inf, "delta_v_A1_leave", Inf, ...
        "delta_v_A2_arrive", Inf, "delta_v_A2_leave", Inf, ...
        "delta_v_A3_arrive", Inf, "delta_v_total", Inf, ...
        "et_launch", Inf, "et_arrive_1", Inf, "et_stay_1", Inf, ...
        "et_arrive_2", Inf, "et_stay_2", Inf, ...
        "et_arrive_3", Inf);

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
                
                asteroid_optimized_data(i, j, k) = OPTIMIZE_TIMES(A_ID_1, A_ID_2, ...
                    A_ID_3, LAUNCH_DATES, m_1, m_2, m_3);
                elapsed_times(current_operation) = toc;
                ETA = (sum(elapsed_times) * TOTAL_OPERATIONS / current_operation) - sum(elapsed_times);

                progress = current_operation / TOTAL_OPERATIONS;
                waitbar(progress, wb, sprintf('Operation %d of %d (ETA: %.2f seconds)', ...
                    current_operation, TOTAL_OPERATIONS, ETA));
            end
        end
    end

    close(wb);
    
    save(sprintf("asteroid_data_%d_%d_%d.mat", m_1, m_2, m_3), "asteroid_optimized_data");
end
