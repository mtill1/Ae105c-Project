function lockSpatialPlotScaling(ax, mode)
% lockSpatialPlotScaling  Equalize axis scaling for ECLIPJ2000 trajectory plots.
%
% Syntax
%   lockSpatialPlotScaling
%   lockSpatialPlotScaling(ax)
%   lockSpatialPlotScaling(ax, mode)
%
% Description
%   For 3D views, sets DataAspectRatio so 1 km in X, Y, and Z appears equal on
%   screen (fixes misleading perspective when axis limits differ in magnitude).
%   For view(2) top-down XY, uses axis equal so X and Y km scales match.
%
% Inputs
%   ax   Axes handle; default gca.
%   mode 'ecliptic3d' (default) | 'eclipticxy'

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    if nargin < 2 || isempty(mode)
        mode = 'ecliptic3d';
    end

    modeNorm = lower(char(mode));
    if strcmp(modeNorm, 'eclipticxy')
        axis(ax, 'equal');
        axis(ax, 'tight');
    else
        daspect(ax, [1 1 1]);
        axis(ax, 'tight');
    end
end
