%struct("delta_v_launch", delta_v_launch, ...
%        "delta_v_A1_arrive", delta_v_A1_arrive, "delta_v_A1_leave", delta_v_A1_leave, ...
%        "delta_v_A2_arrive", delta_v_A2_arrive, "delta_v_A2_leave", delta_v_A2_leave, ...
%        "delta_v_A3_arrive", delta_v_A3_arrive, "delta_v_total", delta_v_total, ...
%        "et_launch", et_launch, "et_arrive_1", et_arrive_1, "et_stay_1", et_stay_1, ...
%        "et_arrive_2", et_arrive_2, "et_stay_2", et_stay_2, ...
%        "et_arrive_3", et_arrive_3)


function GRAPH_ASTEROID_PATH(PATH_DEFINED_VECTOR, A_ID_1, A_ID_2, A_ID_3)
    

end

function [t_f, m_f, X, T, R_max, R_min] = TWO_BODY_SIM(a_Moon, r_start, a_start, f_limit, I_sp, T_max, M0, myu, N_days)
    x_0 = [r_start, 0, 0, 0, -sqrt(myu * ((2/r_start) - (1/a_start))), 0, M0]';

    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
    [T, X] = ode113(@(t, x) dx_TBS_dt(t, x, f_limit, I_sp, T_max, myu), [0, N_days * 24 * 60 * 60], x_0, options);

    R = vecnorm(X(:, 1:3)');

    t_f = min(T(R >= a_Moon));
    m_f = min(X(R <= a_Moon, 7));

    R_max = max(R);
    R_min = min(R);

end

function x_dot = dx_TBS_dt(~, x, f_limit, I_sp, T_max, myu)
    r_vec = x(1:3)';
    r_mag = norm(r_vec);


    v_vec = x(4:6)';
    v_mag = norm(v_vec);

    m = x(7);

    [~, ~, ~, ~, ~, ~, f, ~, ~, ~, ~, ~] = ELORB(r_vec, v_vec, myu);

    g0 = 9.81 * (1e-3);
    
    T_mag = 0;
    % Essentially, it makes sure the f_limit is inbetween -45 and 45 
    if or(clip(f, f_limit, 360 - f_limit) ~= f, f_limit < 0)
        T_mag = T_max;
    end
    
    T_vec = T_mag * v_vec / v_mag;

    r_ddot = - (myu * r_vec ./ (r_mag.^3)) + (T_vec ./ m);

    m_dot = - T_mag / (I_sp * g0);

    x_dot = [x(4), x(5), x(6), r_ddot(1), r_ddot(2), r_ddot(3), m_dot]';
end