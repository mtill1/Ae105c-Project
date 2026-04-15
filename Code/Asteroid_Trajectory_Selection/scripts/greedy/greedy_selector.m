try
    asteroid_list = LOAD_KERNELS("NOTABLE_ASTEROID_BSPs");
catch exception
    close all force;
    fprintf('An error occurred: %s\n', exception.message);
    error("Most likely you need the BSP folder at data/NOTABLE_ASTEROID_BSPs" + ...
            " and you must be in the repo root (so relative paths resolve).");
end

% --- How many full N^3 passes will run? ---
% 'single' (default): one pass — one waitbar from 1 .. NUM_ASTEROIDS^3 (~minutes).
% 'full': every combination of M_VALUES for 6 Lambert knobs = 4^6 = 4096 passes
%         (~4096 × one_pass_time — often many hours to weeks).
GREEDY_M_SEARCH_MODE = 'single';

M_VALUES = [0.1, -0.1, 1, -1];
M_length = length(M_VALUES);

MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027';
MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2035';

% Default: short-way, zero full revolutions on both arcs of each leg (matches 0.1, 0.1 pattern).
M_default = repmat([0.1, 0.1], 3, 1);

if strcmpi(GREEDY_M_SEARCH_MODE, 'full')
    fprintf(['GREEDY_M_SEARCH_MODE = full: %d passes over all asteroid triples ' ...
        '(4^6 combinations). Estimated time ~ %.0f × (one pass duration).\n'], ...
        M_length^6, M_length^6);
    iter_vec = [1, 1, 1, 1, 1, 1];

    while iter_vec(end) <= M_length

        M = reshape(M_VALUES(iter_vec), 2, 3)';

        save_filename = "greedy_test";

        for idx = 1:6
            save_filename = save_filename + sprintf("_%.1f", M_VALUES(iter_vec(idx)));
        end

        asteroid_optimized_data = GENERATE_GREEDY_OPTIMIZED_DATA(asteroid_list, ...
            M, MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC, save_filename);

        iter_vec(1) = iter_vec(1) + 1;
        while any(iter_vec > M_length)
            for j = 1:(length(iter_vec) - 1)
                if iter_vec(j) > M_length
                    iter_vec(j + 1) = iter_vec(j + 1) + 1;
                    iter_vec(j) = 1;
                end
            end
            if iter_vec(end) > M_length
                break
            end
        end
    end
else
    fprintf(['GREEDY_M_SEARCH_MODE = %s: one pass over all asteroid triples ' ...
        '(one waitbar).\n'], GREEDY_M_SEARCH_MODE);
    save_filename = "greedy_default_shortway_m0";
    asteroid_optimized_data = GENERATE_GREEDY_OPTIMIZED_DATA(asteroid_list, ...
        M_default, MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC, save_filename);
end
