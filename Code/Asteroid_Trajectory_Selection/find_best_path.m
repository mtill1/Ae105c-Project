NUM_ASTEROIDS = length(asteroid_list);




i_mins = -1 * ones(1, 10);
j_mins = -1 * ones(1, 10);
k_mins = -1 * ones(1, 10);



for n = 1:10
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


    
                if min_delta_v > asteroid_optimized_data(i, j, k).delta_v_total
                    avoid = 0;
                    for p = 1:n
                        if i_mins(p) == i && k_mins(p) == k && k_mins(p) == k
                            avoid = 1;
                        end
                    end

                    if avoid == 1
                        continue;
                    end

                    min_delta_v = asteroid_optimized_data(i, j, k).delta_v_total;
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
    fprintf('Number %d Minimum delta_v: %.2f km/s found for asteroids %s (INDEX %d), %s (INDEX %d), %s (INDEX %d)\n', ...
        n, min_delta_v, asteroid_list(i_mins(n)).NAME, i_mins(n),...
        asteroid_list(j_mins(n)).NAME, j_mins(n),...
        asteroid_list(k_mins(n)).NAME, k_mins(n));
end


min_delta_v = asteroid_optimized_data(i_mins(1), j_mins(1), k_mins(1)).delta_v_total;

% Display the minimum delta_v and corresponding asteroid indices
fprintf('\nMinimum delta_v: %.2f km/s found for asteroids %s (INDEX %d), %s (INDEX %d), %s (INDEX %d)\n', ...
    min_delta_v, asteroid_list(i_mins(1)).NAME, i_mins(1),...
    asteroid_list(j_mins(1)).NAME, j_mins(1),...
    asteroid_list(k_mins(1)).NAME, k_mins(1));


