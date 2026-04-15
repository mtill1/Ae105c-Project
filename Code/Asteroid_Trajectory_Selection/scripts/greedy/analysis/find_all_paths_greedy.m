NUM_ASTEROIDS = length(asteroid_list);

mat_files = dir(fullfile(getRepoRoot(), 'outputs', 'greedy', '*.mat'));

TOTAL_DV_MAP = ones(NUM_ASTEROIDS, NUM_ASTEROIDS, NUM_ASTEROIDS);

DATA = {'dv_total_mission', 'dv_launch_leg1', 'asteroid 1', 'asteroid 2', 'asteroid 3', 'filename';};
for p = 1:length(mat_files)
    i_mins = -1 * ones(1, 1);
    j_mins = -1 * ones(1, 1);
    k_mins = -1 * ones(1, 1);

    load(fullfile(mat_files(p).folder, mat_files(p).name));
    for n = 1:1
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
                    for l = 1:3
                        total_dv = total_dv + asteroid_optimized_data(i, j, k, l).dv_total;
                    end
                    TOTAL_DV_MAP(i, j, k) = total_dv;
    
                    if min_delta_v > total_dv
                        avoid = 0;
                        for w = 1:n
                            if i_mins(w) == i && j_mins(w) == j && k_mins(w) == k
                                avoid = 1;
                                break;
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
        
        
    end

    i_mins(n) = i_min_current;
        j_mins(n) = j_min_current;
        k_mins(n) = k_min_current;
        % Display the minimum delta_v and corresponding asteroid indices

        DATA(end+1, :) = {min_delta_v, asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).dv_launch,...
            asteroid_list(i_mins(n)).NAME, asteroid_list(j_mins(n)).NAME, asteroid_list(k_mins(n)).NAME, mat_files(p).name};
        fprintf('FILE: %s | Total mission dv (3 legs): %.2f km/s (leg 1 Earth launch dv %.2f) | %s (INDEX %d) | %s (INDEX %d) | %s (INDEX %d)\n', ...
            mat_files(p).name, min_delta_v, asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).dv_launch, asteroid_list(i_mins(n)).NAME, i_mins(n),...
            asteroid_list(j_mins(n)).NAME, j_mins(n),...
            asteroid_list(k_mins(n)).NAME, k_mins(n));
        fprintf(['LAUNCH AT %s (dv %.2f) | FLYBY %s AT %s (dv %.2f) | ARRIVE AT %s (dv %.2f)\n' ...
            'LEAVE AT %s (dv %.2f) | FLYBY %s AT %s (dv %.2f) | ARRIVE AT %s (dv %.2f) \n' ...
            'LEAVE AT %s (dv %.2f) | FLYBY %s AT %s (dv %.2f) | ARRIVE AT %s (dv %.2f) \n\n\n'], ...
            cspice_et2utc(norm(asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).et_launch), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).dv_launch, ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).FLYBY_BODY, ...
            cspice_et2utc(norm(asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).et_flyby), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).dv_arrive, ...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).et_goal), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 1).dv_goal, ...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).et_launch), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).dv_launch, ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).FLYBY_BODY, ...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).et_flyby), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).dv_arrive,...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).et_goal), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 2).dv_goal,...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).et_launch), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).dv_launch,...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).FLYBY_BODY, ...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).et_flyby), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).dv_arrive,...
            cspice_et2utc((asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).et_goal), 'C', 1), ...
            asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n), 3).dv_goal);
end

outCsvDir = fullfile(getRepoRoot(), 'outputs', 'tables');
if ~isfolder(outCsvDir)
    mkdir(outCsvDir);
end
writecell(DATA, fullfile(outCsvDir, 'asteroid_optimized_data_table.csv'));