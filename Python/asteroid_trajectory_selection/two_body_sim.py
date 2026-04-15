"""Two-body simulation using scipy's solve_ivp (replaces MATLAB ode113)."""

import numpy as np
from scipy.integrate import solve_ivp


def two_body_sim(t_final, x_0, mu_sun):
    """Propagate a two-body trajectory from t=0 to t_final."""

    def dx_dt(t, x):
        r_vec = x[0:3]
        r_mag = np.linalg.norm(r_vec)
        r_ddot = -mu_sun * r_vec / (r_mag ** 3)
        return np.array([x[3], x[4], x[5], r_ddot[0], r_ddot[1], r_ddot[2]])

    sol = solve_ivp(
        dx_dt,
        [0, t_final],
        x_0,
        method='DOP853',
        rtol=1e-4,
        atol=1e-4,
        dense_output=True,
    )

    T = sol.t
    X = sol.y.T  # shape (N, 6), rows are time steps

    return X, T
