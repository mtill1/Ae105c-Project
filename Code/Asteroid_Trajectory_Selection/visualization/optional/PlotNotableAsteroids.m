clear;
asteroid_list = LOAD_KERNELS("NOTABLE_ASTEROID_BSPs");
outVidDir = fullfile(getRepoRoot(), 'outputs', 'direct', 'videos');
if ~isfolder(outVidDir)
    mkdir(outVidDir);
end
timestampTag = datestr(now, 'yyyymmdd_HHMMSS');
PlotAsteroidOrbits(asteroid_list, 8, 80, 'Jan 1 12:00:00 UTC 2027', ...
    'Dec 31 12:00:00 UTC 2031', ...
    fullfile(outVidDir, sprintf("Jan_1_2027-Dec_31_2031-Notable-Asteroids_%s.mp4", timestampTag)));
