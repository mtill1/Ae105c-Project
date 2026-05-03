function [orbit_fig] = plot_circular_orbits(xr, yr, zr, height, n)
    orbit_fig = figure;
    hold on;
    grid minor;

    axis equal;

    ellipsoid(0, 0, 0, zr, yr, xr, n);

    N = 10;

    for i = 1:N
        plotCircle3D([0, 0, 0], [cos(i * pi / N), sin(i * pi / N), 0], max([xr, yr, zr]) + height);
    end

    title('Orbital Path');
    xlabel('Z-axis');
    ylabel('Y-axis');
    zlabel('X-axis');

    view(3);

    hold off;
end