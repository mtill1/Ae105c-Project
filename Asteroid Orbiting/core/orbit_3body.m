function [t_sol, y_sol] = orbit_3body(asteroid, x_0, et_0, et_f, myu_aster)
    [t_sol, y_sol] = ode113(@(t, y) dx_dt_3body(t, y, asteroid, myu_aster), [et_0, et_f], x_0);
end

function dxdt = dx_dt_3body(et, x_vec, asteroid, myu_aster)
    pos_asteroid = cspice_spkezr(int2str(asteroid.ID), et, 'ECLIPJ2000', 'NONE', '10');
    myu_sun = cspice_bodvcd(10, 'GM', 6);

    r_sc_rel_aster = x_vec(1:3) - pos_asteroid(1:3);

    dxdt = 0 .* x_vec;

    r_asteroid = norm(r_sc_rel_aster);
    dxdt(1:3) = x_vec(4:6); % Position derivatives
    dxdt(4:6) = - myu_aster .* r_sc_rel_aster ./ (r_asteroid.^3) - ...
        myu_sun .* x_vec(1:3) ./ (norm(x_vec(1:3)).^3);
end