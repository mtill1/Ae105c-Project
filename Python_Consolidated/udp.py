"""
udp.py — pagmo User Defined Problem (UDP) classes for trajectory optimization.

Defines optimization problems compatible with pygmo's algorithm ecosystem
(MBH, DE, island model, etc.). Each UDP wraps the existing delta-V computation
pipeline and exposes it as a box-bounded continuous optimization problem.

Three problem variants:
    1. AsteroidTripletUDP        — 6D, pure Lambert legs (baseline)
    2. AsteroidTripletDSM_UDP    — 9D, Lambert legs + mid-leg Deep Space Maneuvers
    3. MarsTripletUDP            — 7D, Mars flyby variant (pure Lambert)

References:
    Izzo, D. (2015). "Revisiting Lambert's problem." CMDA, 121(1):1-15.
    Vasile & De Pascale (2006). "On the Preliminary Design of MGA Trajectories." JGCD.
    pykep: https://esa.github.io/pykep/
"""

import numpy as np

from core import (solve_lambert, propagate_two_body, get_state,
                  compute_flyby_dv, get_mu, get_radius,
                  DAY, YEAR, WEEK, MONTH, MU_SUN,
                  MAX_MISSION_DURATION, unpack_input, unpack_mars_input)


PENALTY_DV = 1e3  # km/s, returned when Lambert fails or constraints violated


# =============================================================================
# LAYER 1: Pure Lambert (6D)
# =============================================================================

class AsteroidTripletUDP:
    """pagmo UDP for optimizing departure/transfer/stay times over 3-asteroid
    rendezvous using Lambert arcs.

    Decision vector (6 variables, in years):
        x = [t_launch_offset, t_transfer_1, t_stay_1,
             t_transfer_2, t_stay_2, t_transfer_3]

    Objective: minimize total impulsive delta-V (excluding Earth departure).
    """

    def __init__(self, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
        self.a_id_1 = a_id_1
        self.a_id_2 = a_id_2
        self.a_id_3 = a_id_3
        self.launch_range = launch_range
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3

        self._lb = np.array([
            0, 2 * WEEK, 3 * MONTH, 2 * WEEK, 3 * MONTH, 2 * WEEK
        ]) / YEAR
        self._ub = np.array([
            launch_range[1] - launch_range[0], 8 * YEAR, YEAR,
            8 * YEAR, YEAR, 8 * YEAR
        ]) / YEAR

    def fitness(self, x):
        et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
            unpack_input(x, self.launch_range)
        if (et_arrive_3 - et_launch) > MAX_MISSION_DURATION:
            return [PENALTY_DV]
        return [_compute_deltav_lambert(
            self.a_id_1, self.a_id_2, self.a_id_3,
            et_launch, et_arrive_1, et_stay_1,
            et_arrive_2, et_stay_2, et_arrive_3,
            self.m_1, self.m_2, self.m_3)]

    def get_bounds(self):
        return (self._lb.tolist(), self._ub.tolist())

    def get_name(self):
        return f"Asteroid Triplet ({self.a_id_1}->{self.a_id_2}->{self.a_id_3})"


# =============================================================================
# LAYER 2: MGA-DSM (9D) — Lambert legs + Deep Space Maneuvers
# =============================================================================

class AsteroidTripletDSM_UDP:
    """pagmo UDP for MGA-DSM trajectory optimization.

    Adds an eta parameter per transfer leg. Each leg is split into two Lambert
    sub-arcs joined by a Deep Space Maneuver at the split point.

    Decision vector (9 variables):
        x = [t_launch_offset, t_transfer_1, t_stay_1,
             t_transfer_2, t_stay_2, t_transfer_3,
             eta_1, eta_2, eta_3]

    where eta_i in [0.01, 0.99] is the fraction of the transfer time at which
    the DSM occurs on leg i.
    """

    def __init__(self, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
        self.a_id_1 = a_id_1
        self.a_id_2 = a_id_2
        self.a_id_3 = a_id_3
        self.launch_range = launch_range
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3

        time_lb = np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR
        time_ub = np.array([launch_range[1]-launch_range[0], 8*YEAR, YEAR,
                            8*YEAR, YEAR, 8*YEAR]) / YEAR
        self._lb = np.concatenate([time_lb, [0.01, 0.01, 0.01]])
        self._ub = np.concatenate([time_ub, [0.99, 0.99, 0.99]])

    def fitness(self, x):
        et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
            unpack_input(x[:6], self.launch_range)
        if (et_arrive_3 - et_launch) > MAX_MISSION_DURATION:
            return [PENALTY_DV]
        eta_1, eta_2, eta_3 = x[6], x[7], x[8]
        return [_compute_deltav_dsm(
            self.a_id_1, self.a_id_2, self.a_id_3,
            et_launch, et_arrive_1, et_stay_1,
            et_arrive_2, et_stay_2, et_arrive_3,
            self.m_1, self.m_2, self.m_3,
            eta_1, eta_2, eta_3)]

    def get_bounds(self):
        return (self._lb.tolist(), self._ub.tolist())

    def get_name(self):
        return f"Asteroid Triplet DSM ({self.a_id_1}->{self.a_id_2}->{self.a_id_3})"


# =============================================================================
# Mars flyby variant (7D, pure Lambert)
# =============================================================================

class MarsTripletUDP:
    """pagmo UDP for Mars flyby trajectory optimization.

    Decision vector (7 variables, in years):
        x = [t_launch_offset, t_mars_transfer, t_transfer_1, t_stay_1,
             t_transfer_2, t_stay_2, t_transfer_3]
    """

    def __init__(self, a_id_1, a_id_2, a_id_3, launch_range,
                 m_1, m_2, m_3, m_mars):
        self.a_id_1 = a_id_1
        self.a_id_2 = a_id_2
        self.a_id_3 = a_id_3
        self.launch_range = launch_range
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.m_mars = m_mars

        self._lb = np.array([0, 3*YEAR, 0.2*YEAR, 3*MONTH,
                             0.2*YEAR, 3*MONTH, 0.2*YEAR]) / YEAR
        self._ub = np.array([launch_range[1]-launch_range[0], 8*YEAR, 4*YEAR,
                             YEAR, 4*YEAR, YEAR, 4*YEAR]) / YEAR

    def fitness(self, x):
        et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
            unpack_mars_input(x, self.launch_range)
        if (et_arrive_3 - et_launch) > MAX_MISSION_DURATION:
            return [PENALTY_DV]
        return [_compute_deltav_mars(
            self.a_id_1, self.a_id_2, self.a_id_3,
            et_launch, et_mars,
            et_arrive_1, et_stay_1, et_arrive_2,
            et_stay_2, et_arrive_3,
            self.m_1, self.m_2, self.m_3, self.m_mars)]

    def get_bounds(self):
        return (self._lb.tolist(), self._ub.tolist())

    def get_name(self):
        return f"Mars Triplet ({self.a_id_1}->{self.a_id_2}->{self.a_id_3})"


# =============================================================================
# INTERNAL: Delta-V computation functions
# =============================================================================

def _compute_deltav_lambert(a_id_1, a_id_2, a_id_3,
                            et_launch, et_arrive_1, et_stay_1,
                            et_arrive_2, et_stay_2, et_arrive_3,
                            m_1, m_2, m_3):
    """Compute total delta-V via 3 Lambert arcs. Returns scalar."""
    earth_r, earth_v = get_state('399', et_launch)
    a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
    a1_dep_r, a1_dep_v = get_state(str(a_id_1), et_stay_1)
    a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
    a2_dep_r, a2_dep_v = get_state(str(a_id_2), et_stay_2)
    a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)

    v1_dep, v1_arr, ef1 = solve_lambert(
        earth_r, a1_arr_r, (et_arrive_1 - et_launch) / DAY, m_1, MU_SUN)
    v2_dep, v2_arr, ef2 = solve_lambert(
        a1_dep_r, a2_arr_r, (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    v3_dep, v3_arr, ef3 = solve_lambert(
        a2_dep_r, a3_arr_r, (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if ef1 != 1 or ef2 != 1 or ef3 != 1:
        return PENALTY_DV

    return (np.linalg.norm(v1_arr - a1_arr_v)
            + np.linalg.norm(v2_dep - a1_dep_v)
            + np.linalg.norm(v2_arr - a2_arr_v)
            + np.linalg.norm(v3_dep - a2_dep_v)
            + np.linalg.norm(v3_arr - a3_arr_v))


def _compute_single_leg_dsm(r_dep, v_dep_body, r_arr, v_arr_body,
                             tof_sec, eta, m, mu):
    """Compute delta-V for a single transfer leg with a Deep Space Maneuver.

    Returns (dv_depart, dv_dsm, dv_arrive) or (PENALTY, 0, 0) on failure.
    """
    tof_days = tof_sec / DAY

    v_dep_lambert, _, ef_full = solve_lambert(r_dep, r_arr, tof_days, m, mu)
    if ef_full != 1:
        return PENALTY_DV, 0.0, 0.0

    dv_depart = np.linalg.norm(v_dep_lambert - v_dep_body)

    t_dsm = eta * tof_sec
    r_dsm, v_dsm_pre = propagate_two_body(r_dep, v_dep_lambert, t_dsm, mu)

    remaining_tof_days = (1.0 - eta) * tof_days
    if remaining_tof_days < 0.1:
        return PENALTY_DV, 0.0, 0.0

    v_dsm_post, v_arr_lambert, ef_dsm = solve_lambert(
        r_dsm, r_arr, remaining_tof_days, 0, mu)
    if ef_dsm != 1:
        return PENALTY_DV, 0.0, 0.0

    dv_dsm = np.linalg.norm(v_dsm_post - v_dsm_pre)
    dv_arrive = np.linalg.norm(v_arr_lambert - v_arr_body)

    return dv_depart, dv_dsm, dv_arrive


def _compute_deltav_dsm(a_id_1, a_id_2, a_id_3,
                         et_launch, et_arrive_1, et_stay_1,
                         et_arrive_2, et_stay_2, et_arrive_3,
                         m_1, m_2, m_3,
                         eta_1, eta_2, eta_3):
    """Compute total delta-V for 3 MGA-DSM legs."""
    earth_r, earth_v = get_state('399', et_launch)
    a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
    a1_dep_r, a1_dep_v = get_state(str(a_id_1), et_stay_1)
    a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
    a2_dep_r, a2_dep_v = get_state(str(a_id_2), et_stay_2)
    a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)

    tof_1 = et_arrive_1 - et_launch
    tof_2 = et_arrive_2 - et_stay_1
    tof_3 = et_arrive_3 - et_stay_2

    _, dv_dsm_1, dv_a1_arrive = _compute_single_leg_dsm(
        earth_r, earth_v, a1_arr_r, a1_arr_v, tof_1, eta_1, m_1, MU_SUN)
    dv_a1_depart, dv_dsm_2, dv_a2_arrive = _compute_single_leg_dsm(
        a1_dep_r, a1_dep_v, a2_arr_r, a2_arr_v, tof_2, eta_2, m_2, MU_SUN)
    dv_a2_depart, dv_dsm_3, dv_a3_arrive = _compute_single_leg_dsm(
        a2_dep_r, a2_dep_v, a3_arr_r, a3_arr_v, tof_3, eta_3, m_3, MU_SUN)

    components = [dv_a1_arrive, dv_dsm_1, dv_a1_depart, dv_dsm_2,
                  dv_a2_arrive, dv_a2_depart, dv_dsm_3, dv_a3_arrive]
    if any(c >= PENALTY_DV for c in components):
        return PENALTY_DV

    return sum(components)


def _compute_deltav_mars(a_id_1, a_id_2, a_id_3,
                          et_launch, et_mars,
                          et_arrive_1, et_stay_1,
                          et_arrive_2, et_stay_2, et_arrive_3,
                          m_1, m_2, m_3, m_mars):
    """Compute total delta-V for Earth->Mars flyby->A1->A2->A3."""
    earth_r, earth_v = get_state('399', et_launch)
    mars_r, mars_v = get_state('4', et_mars)
    a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
    a1_dep_r, a1_dep_v = get_state(str(a_id_1), et_stay_1)
    a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
    a2_dep_r, a2_dep_v = get_state(str(a_id_2), et_stay_2)
    a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)

    mu_mars = get_mu(4)
    safe_radius = get_radius(499) + 200

    v0_dep, v0_arr, ef0 = solve_lambert(
        earth_r, mars_r, -(et_mars - et_launch) / DAY, m_mars, MU_SUN)
    v1_dep, v1_arr, ef1 = solve_lambert(
        mars_r, a1_arr_r, -(et_arrive_1 - et_mars) / DAY, m_1, MU_SUN)
    v2_dep, v2_arr, ef2 = solve_lambert(
        a1_dep_r, a2_arr_r, -(et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    v3_dep, v3_arr, ef3 = solve_lambert(
        a2_dep_r, a3_arr_r, -(et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if ef0 != 1 or ef1 != 1 or ef2 != 1 or ef3 != 1:
        return PENALTY_DV

    dv_launch = np.linalg.norm(v0_dep - earth_v)
    dv_mars = compute_flyby_dv(v0_arr, v1_dep, mars_v, mu_mars, safe_radius)

    return (dv_launch + abs(dv_mars)
            + np.linalg.norm(v1_arr - a1_arr_v)
            + np.linalg.norm(v2_dep - a1_dep_v)
            + np.linalg.norm(v2_arr - a2_arr_v)
            + np.linalg.norm(v3_dep - a2_dep_v)
            + np.linalg.norm(v3_arr - a3_arr_v))
