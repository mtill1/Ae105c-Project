NUM_ASTEROIDS = length(asteroid_list);


min_delta_v = Inf;

i_min = -1;
j_min = -1;
k_min = -1;


for i = 1:NUM_ASTEROIDS
    for j = 1:NUM_ASTEROIDS
        for k = 1:NUM_ASTEROIDS
            if i == j || j == k || k == i
                continue
            end

            if min_delta_v > asteroid_optimized_data(i, j, k).delta_v_total
                min_delta_v = asteroid_optimized_data(i, j, k).delta_v_total;
                i_min = i;
                j_min = j;
                k_min = k;
            end
        end
    end
end
disp(i_min)
FLIGHTPATH_ANIMATION(asteroid_optimized_data(i_min, j_min, k_min), ...
    asteroid_list, i_min, j_min, k_min, 10, "TEST.mp4");

% Display the minimum delta_v and corresponding asteroid indices
fprintf('Minimum delta_v: %.2f found for asteroids %s, %s, %s\n', ...
    min_delta_v, asteroid_list(i_min).NAME, ...
    asteroid_list(j_min).NAME, ...
    asteroid_list(k_min).NAME);
