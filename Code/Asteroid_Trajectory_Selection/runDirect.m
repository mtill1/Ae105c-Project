% Runs Direct method (optimize + print best direct tours)
%
% This convenience runner sets Current Folder to the repo root so relative
% paths (data/, outputs/) resolve correctly.
thisFile = which(mfilename);
if isempty(thisFile)
    thisFile = mfilename('fullpath');
end
moduleDir = fileparts(thisFile);            % .../code/Asteroid_Trajectory_Selection
repoRoot  = fileparts(fileparts(moduleDir));% repo root (parent of /code)
cd(repoRoot);

if ~isfolder(fullfile('data','NOTABLE_ASTEROID_BSPs'))
    error("Expected BSP folder at '%s'. Are you in the repo root?", fullfile('data','NOTABLE_ASTEROID_BSPs'));
end

addpath(genpath('code/Asteroid_Trajectory_Selection'));
addpath(genpath(fullfile('data','NOTABLE_ASTEROID_BSPs')));

run('code/Asteroid_Trajectory_Selection/scripts/direct/asteroid_selector.m');
run('code/Asteroid_Trajectory_Selection/scripts/direct/find_best_path.m');
run('code/Asteroid_Trajectory_Selection/visualization/PlotBestDirectPath.m');

