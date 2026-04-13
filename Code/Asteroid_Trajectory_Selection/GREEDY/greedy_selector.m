

try
    asteroid_list = LOAD_KERNELS("NOTABLE_ASTEROID_BSPs");
catch exception
    close all force;
    fprintf('An error occurred: %s\n', exception.message);
    error("Most likely you need to add NOTABLE_ASTEROID_BSPs to your path" + ...
            " and you must be in the main github folder.");
end

M_VALUES = [0.1, -0.1, 1, -1];
M_length = length(M_VALUES);

MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027';
MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035';

iter_vec = [1, 1, 1, 1, 1, 1];

while iter_vec(end) <= M_length
    

    M = reshape(M_VALUES(iter_vec), 2, 3)';

    save_filename = "greedy_test";

    for i = 1:6
        save_filename = save_filename + sprintf("_%.1f", M_VALUES(iter_vec(i)));
    end

    asteroid_optimized_data = GENERATE_GREEDY_OPTIMIZED_DATA(asteroid_list, ...
        M, MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC, save_filename);

    iter_vec(1) = iter_vec(1) + 1;
    while any(iter_vec > M_length)
        for j = 1:(length(iter_vec)-1)
            if iter_vec(j) > M_length
                iter_vec(j+1) = iter_vec(j+1) + 1;
                iter_vec(j) = 1;
            end
        end
        if iter_vec(end) > M_length
            break
        end
    end
end