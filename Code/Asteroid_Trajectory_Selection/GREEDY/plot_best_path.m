NUM_ASTEROIDS = length(asteroid_list);




i_mins = -1 * ones(1, 6);
j_mins = -1 * ones(1, 6);
k_mins = -1 * ones(1, 6);

TOTAL_DV_MAP = ones(NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS);

for n = 1:6
    i_min_current = -1;
    j_min_current = -1;
    k_min_current = -1;
    min_delta_v = Inf;
    
    for i = 1:NUM_ASTEROIDS
        for j = 1:NUM_ASTEROIDS
            for k = 1:NUM_ASTEROIDS
                if i == j || j == k || k == i
                    continue
                end

                total_dv = 0;

                for l = 2:3
                    total_dv = total_dv + asteroid_optimized_data(i, j, k, l).dv_total;
                end
                TOTAL_DV_MAP(i, j, k) = total_dv;

                if min_delta_v > total_dv
                    avoid = 0;
                    for p = 1:n
                        if i_mins(p) == i && k_mins(p) == k && k_mins(p) == k
                            avoid = 1;
                        end
                    end

                    if avoid == 1
                        continue;
                    end

                    min_delta_v = total_dv;
                    i_min_current = i;
                    j_min_current = j;
                    k_min_current = k;
                end
            end
        end
    end
    
    i_mins(n) = i_min_current;
    j_mins(n) = j_min_current;
    k_mins(n) = k_min_current;
    % Display the minimum delta_v and corresponding asteroid indices
end


GREEDY_FLIGHTPATH_ANIMATION(asteroid_optimized_data(i_mins(1), j_mins(1), k_mins(1), :), ...
    asteroid_list, i_mins(1), j_mins(1), k_mins(1), 12, "GREEDY_BEST_PATH.mp4")