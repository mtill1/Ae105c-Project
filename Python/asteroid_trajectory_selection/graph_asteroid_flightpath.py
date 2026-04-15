"""Static 3D plot of the three trajectory legs.

Translates GRAPH_ASTEROID_FLIGHTPATH.m to Python.
"""

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .two_body_sim import two_body_sim


def graph_asteroid_flightpath(path_defined_vector, a_id_1, a_id_2):
    """Plot the three trajectory legs in 3D.

    Parameters
    ----------
    path_defined_vector : dict
        Keys: et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2,
              et_arrive_3, delta_v_launch, delta_v_A1_leave, delta_v_A2_leave.
    a_id_1 : str
        NAIF ID string for asteroid 1.
    a_id_2 : str
        NAIF ID string for asteroid 2.
    """
    pdv = path_defined_vector

    t_1 = pdv['et_arrive_1'] - pdv['et_launch']
    t_2 = pdv['et_arrive_2'] - pdv['et_stay_1']
    t_3 = pdv['et_arrive_3'] - pdv['et_stay_2']

    _, mu_vals = spiceypy.bodvcd(10, 'GM', 10)
    mu_sun = mu_vals[0]

    # --- Leg 1: Earth to Asteroid 1 ---
    earth_launch_state = spiceypy.spkezr('399', pdv['et_launch'],
                                         'ECLIPJ2000', 'NONE', '10')[0]
    x_0 = np.concatenate([earth_launch_state[0:3],
                          earth_launch_state[3:6] + pdv['delta_v_launch']])
    X_1, T_1 = two_body_sim(t_1, x_0, mu_sun)

    # --- Leg 2: Asteroid 1 to Asteroid 2 ---
    a1_leaving_state = spiceypy.spkezr(a_id_1, pdv['et_stay_1'],
                                       'ECLIPJ2000', 'NONE', '10')[0]
    x_0 = np.concatenate([a1_leaving_state[0:3],
                          a1_leaving_state[3:6] + pdv['delta_v_A1_leave']])
    X_2, T_2 = two_body_sim(t_2, x_0, mu_sun)

    # --- Leg 3: Asteroid 2 to Asteroid 3 ---
    a2_leaving_state = spiceypy.spkezr(a_id_2, pdv['et_stay_2'],
                                       'ECLIPJ2000', 'NONE', '10')[0]
    x_0 = np.concatenate([a2_leaving_state[0:3],
                          a2_leaving_state[3:6] + pdv['delta_v_A2_leave']])
    X_3, T_3 = two_body_sim(t_3, x_0, mu_sun)

    # --- Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    ax.plot(X_1[:, 0], X_1[:, 1], X_1[:, 2], 'r', label='Trajectory to A1')
    ax.plot(X_2[:, 0], X_2[:, 1], X_2[:, 2], 'g', label='Trajectory from A1 to A2')
    ax.plot(X_3[:, 0], X_3[:, 1], X_3[:, 2], 'b', label='Trajectory from A2')

    ax.legend()
    ax.view_init(elev=30, azim=-60)  # 3D view
    plt.show()
