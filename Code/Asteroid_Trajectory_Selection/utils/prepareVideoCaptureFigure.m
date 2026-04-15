function prepareVideoCaptureFigure(figHandle, pixelPos)
% prepareVideoCaptureFigure  Force fixed figure size for consistent VideoWriter frames.
%
% VideoWriter requires every frame image to match the first. Maximized figures and
% clf/layout changes often change the captured bitmap size; this resets a normal
% window to a fixed pixel rectangle before drawing each frame.

    if nargin < 2 || isempty(pixelPos)
        pixelPos = [100 100 1280 720];
    end

    figure(figHandle);
    set(figHandle, 'WindowState', 'normal', 'Units', 'pixels');
    clf(figHandle);
    % Hardcode dark, non-white plot styling for consistent video appearance.
    set(figHandle, 'Position', pixelPos, 'Units', 'pixels', ...
        'Toolbar', 'none', 'Color', [0.12 0.12 0.12], ...
        'DefaultAxesColor', [0.02 0.02 0.02], ...
        'DefaultAxesXColor', [0.95 0.95 0.95], ...
        'DefaultAxesYColor', [0.95 0.95 0.95], ...
        'DefaultAxesZColor', [0.95 0.95 0.95], ...
        'DefaultTextColor', [0.95 0.95 0.95]);
end
