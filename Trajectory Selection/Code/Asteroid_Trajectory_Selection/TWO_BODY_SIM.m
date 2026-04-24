function [X, T] = TWO_BODY_SIM(t_final, x_0, MYU_SUN)
    options = odeset('RelTol', 1e-4, 'AbsTol', 1e-4);
    [T, X] = ode113(@(t, x) dx_TBS_dt(t, x, MYU_SUN), [0, t_final], x_0, options);
end

function x_dot = dx_TBS_dt(~, x, MYU_SUN)
    r_vec = x(1:3)';
    r_mag = norm(r_vec);

    r_ddot = - (MYU_SUN * r_vec ./ (r_mag.^3));
    x_dot = [x(4), x(5), x(6), r_ddot(1), r_ddot(2), r_ddot(3)]';
end