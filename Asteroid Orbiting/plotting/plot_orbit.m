function [orbit_fig, velocity_fig] = plot_orbit(t_sol, y_sol, xr, yr, zr, n)
    orbit_fig = figure;
    hold on;
    grid minor;

    scatter3(y_sol(1, 1), y_sol(1, 2), y_sol(1, 3), "DisplayName", "Start")
    scatter3(y_sol(end, 1), y_sol(end, 2), y_sol(end, 3), "DisplayName", "End")
    
    axis equal;
    %xlim([-500, 500])
    %ylim([-500, 500])
    %zlim([-500, 500])
    

    ellipsoid(0, 0, 0, xr, yr, zr, n);
    plot3(y_sol(:, 1), y_sol(:, 2), y_sol(:, 3), "LineWidth", 2);

    title('Orbital Path');
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');

    view(3);
    legend()
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