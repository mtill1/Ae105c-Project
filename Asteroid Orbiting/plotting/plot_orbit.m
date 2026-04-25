function [orbit_fig, velocity_fig] = plot_orbit(t_sol, y_sol, xr, yr, zr, n)
    orbit_fig = figure;
    hold on;
    grid minor;

    axis equal;

    ellipsoid(0, 0, 0, xr, yr, zr, n);
    plot3(y_sol(:, 3), y_sol(:, 2), y_sol(:, 1), "LineWidth", 2);

    title('Orbital Path');
    xlabel('Z-axis');
    ylabel('Y-axis');
    zlabel('X-axis');

    view(3);

    hold off;

    velocity_fig = figure;
    hold on;
    grid minor;
    sgtitle("Velocity Figure")
    axes_names = ["X", "Y", "Z"];
    
    for i = 1:3
        subplot(1, 3, i);
        plot(t_sol, y_sol(:, 3 + i));
        title(sprintf("%s Velocity Component", axes_names(i)));
    end
    
    hold off;
end