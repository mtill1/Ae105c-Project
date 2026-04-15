function repoRoot = getRepoRoot()
% getRepoRoot  Return repository root folder from module location.
%
% Syntax
%   repoRoot = getRepoRoot()
%
% Description
%   Computes the repo root using this function file path. This is used for
%   stable output paths so scripts do not write under nested script folders.

    thisFile = mfilename('fullpath');
    utilsDir = fileparts(thisFile);        % .../code/Asteroid_Trajectory_Selection/utils
    moduleDir = fileparts(utilsDir);       % .../code/Asteroid_Trajectory_Selection
    codeDir = fileparts(moduleDir);        % .../code
    repoRoot = fileparts(codeDir);         % repo root
end

