function [t_sol, y_sol] = orbit_no_control(asteroid, x_0, et_0, et_f)
    [t_sol, y_sol] = ode113(@(t, y) dx_dt_HCW_nonlinear(t, y, asteroid), [et_0, et_f], x_0);
end

function dxdt = dx_dt_HCW_nonlinear(et, x_vec, asteroid)
    pos_asteroid = cspice_spkezr(int2str(asteroid.ID), et, 'ECLIPJ2000', 'NONE', '10');
    myu = cspice_bodvcd(10, 'GM', 6);

    r_asteroid = norm(pos_asteroid(1:3));

    x = x_vec(1);
    y = x_vec(2);
    z = x_vec(3);
    x_d = x_vec(4);
    y_d = x_vec(5);
    z_d = x_vec(6);

    n_asteroid = sqrt(myu ./ (r_asteroid).^3);

    x_ddot = - myu .* (r_asteroid + x) ./ (((r_asteroid + x).^2 + y.^2 + z.^2).^(3/2)) + ...
        myu ./ (r_asteroid.^2) + ...
        2 * n_asteroid .* y_d + ...
        x .* n_asteroid.^2;

    y_ddot = - myu .* y ./ (((r_asteroid + x).^2 + y.^2 + z.^2).^(3/2)) + ...
        - 2 * n_asteroid .* x_d + ...
        y .* n_asteroid.^2;

    z_ddot = - myu .* y ./ (((r_asteroid + x).^2 + y.^2 + z.^2).^(3/2));

    dxdt = [x_d, y_d, z_d, x_ddot, y_ddot, z_ddot]';
    
end