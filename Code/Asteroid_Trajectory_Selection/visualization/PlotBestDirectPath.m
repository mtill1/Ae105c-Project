outVidDir = fullfile(getRepoRoot(), 'outputs', 'direct', 'videos');
if ~isfolder(outVidDir)
    mkdir(outVidDir);
end
timestampTag = datestr(now, 'yyyymmdd_HHMMSS');

for n = 1:1
    AnimateDirectFlightpath(asteroid_optimized_data(i_mins(n), j_mins(n), k_mins(n)), ...
        asteroid_list, i_mins(n), j_mins(n), k_mins(n), 10, ...
        fullfile(outVidDir, sprintf("%s_%s_%s_DIRECT_BEST_3D_%s.mp4", asteroid_list(i_mins(n)).NAME, ...
        asteroid_list(j_mins(n)).NAME, asteroid_list(k_mins(n)).NAME, timestampTag)));
end