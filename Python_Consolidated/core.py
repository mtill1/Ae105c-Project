"""
core.py — Foundational utilities for asteroid trajectory selection.

Uses pykep (ESA) for Lambert solving, orbit propagation, and flyby computation.
Uses spiceypy for SPICE kernel management and asteroid ephemerides.

All internal units: km, km/s, km^3/s^2 (SPICE convention).
Conversions to/from SI (pykep) are handled inside wrapper functions.
"""

import os
import glob
import numpy as np
import pykep as pk
import spiceypy

# =============================================================================
# CONSTANTS (in seconds, matching SPICE ephemeris time)
# =============================================================================
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY
MONTH = 30.4375 * DAY       # average month (365.25/12)
YEAR = 365.25 * DAY          # Julian year

MAX_MISSION_DURATION = 14 * YEAR  # hard cap (must stay within BSP coverage to 2050)

# Unit conversion factors
_KM2M = 1e3
_M2KM = 1e-3

# Gravitational parameters in km^3/s^2 (SPICE convention)
MU_SUN = pk.MU_SUN * 1e-9     # m^3/s^2 -> km^3/s^2
MU_EARTH = pk.MU_EARTH * 1e-9
EARTH_RADIUS = pk.EARTH_RADIUS * _M2KM  # m -> km


# =============================================================================
# LAMBERT SOLVER (wrapper around pykep.lambert_problem)
# =============================================================================

def solve_lambert(r1_km, r2_km, tof_days, m, mu_km3s2):
    """Solve Lambert's problem using pykep.

    Parameters
    ----------
    r1_km, r2_km : array (3,) — position vectors [km]
    tof_days : float — time of flight [days]. Negative => long-way (clockwise).
    m : int — number of complete revolutions. Negative => left branch.
    mu_km3s2 : float — gravitational parameter [km^3/s^2]

    Returns
    -------
    V1, V2 : ndarray (3,) — terminal velocities [km/s]
    exitflag : int — 1 = success, -1 = no solution
    """
    r1_m = np.asarray(r1_km, dtype=float) * _KM2M
    r2_m = np.asarray(r2_km, dtype=float) * _KM2M
    tof_sec = abs(tof_days) * DAY
    mu_m3s2 = mu_km3s2 * 1e9

    cw = bool(tof_days < 0)
    multi_revs = abs(int(m))

    try:
        lp = pk.lambert_problem(
            [float(x) for x in r1_m], [float(x) for x in r2_m],
            float(tof_sec), float(mu_m3s2),
            cw, multi_revs,
        )

        v1_all = lp.get_v1()
        v2_all = lp.get_v2()

        if multi_revs == 0:
            idx = 0
        else:
            if m > 0:
                idx = 2 * multi_revs - 1
            else:
                idx = 2 * multi_revs
            if idx >= len(v1_all):
                return np.zeros(3), np.zeros(3), -1

        v1 = np.array(v1_all[idx]) * _M2KM
        v2 = np.array(v2_all[idx]) * _M2KM
        return v1, v2, 1

    except Exception:
        return np.zeros(3), np.zeros(3), -1


def solve_lambert_best(r1_km, r2_km, tof_days, mu_km3s2, max_revs=2):
    """Solve Lambert trying multiple revolutions and both directions, return best.

    Tries m=0,1,...,max_revs for both short-way and long-way, and picks the
    solution with the lowest total delta-v (||V1-V_body1|| + ||V2-V_body2|| proxy:
    we minimize ||V1||+||V2|| as a heuristic since body velocities aren't known here).

    Returns (V1, V2, exitflag) for the best solution found.
    """
    r1_m = np.asarray(r1_km, dtype=float) * _KM2M
    r2_m = np.asarray(r2_km, dtype=float) * _KM2M
    tof_sec = abs(tof_days) * DAY
    mu_m3s2 = mu_km3s2 * 1e9

    best_v1, best_v2, best_cost = np.zeros(3), np.zeros(3), np.inf
    found = False

    for cw in [False, True]:
        try:
            lp = pk.lambert_problem(
                [float(x) for x in r1_m], [float(x) for x in r2_m],
                float(tof_sec), float(mu_m3s2),
                cw, max_revs,
            )
            v1_all = lp.get_v1()
            v2_all = lp.get_v2()
            for idx in range(len(v1_all)):
                v1 = np.array(v1_all[idx]) * _M2KM
                v2 = np.array(v2_all[idx]) * _M2KM
                cost = np.linalg.norm(v1) + np.linalg.norm(v2)
                if cost < best_cost:
                    best_cost = cost
                    best_v1, best_v2 = v1, v2
                    found = True
        except Exception:
            continue

    return best_v1, best_v2, (1 if found else -1)


# =============================================================================
# TWO-BODY PROPAGATION (wrapper around pykep.propagate_lagrangian)
# =============================================================================

def propagate_two_body(r_km, v_km_s, tof_sec, mu_km3s2):
    """Propagate a two-body trajectory to a single final time.

    Returns (r_final_km, v_final_km_s) as numpy arrays.
    """
    r_m = list(np.asarray(r_km, dtype=float) * _KM2M)
    v_ms = list(np.asarray(v_km_s, dtype=float) * _KM2M)
    mu_m3s2 = mu_km3s2 * 1e9

    rf, vf = pk.propagate_lagrangian(r0=r_m, v0=v_ms, tof=float(tof_sec), mu=mu_m3s2)
    return np.array(rf) * _M2KM, np.array(vf) * _M2KM


def two_body_sim(t_final, x_0, mu_km3s2, n_steps=200):
    """Propagate a two-body trajectory and return states at many times.

    Drop-in replacement for the old scipy solve_ivp version.
    Returns (X, T) where X is (N, 6) array [km, km/s] and T is (N,) array [s].
    """
    r_m = list(np.asarray(x_0[0:3], dtype=float) * _KM2M)
    v_ms = list(np.asarray(x_0[3:6], dtype=float) * _KM2M)
    mu_m3s2 = mu_km3s2 * 1e9

    tofs = np.linspace(0, t_final, n_steps).tolist()

    X = np.zeros((len(tofs), 6))
    for i, tof in enumerate(tofs):
        rf, vf = pk.propagate_lagrangian(r0=r_m, v0=v_ms, tof=float(tof), mu=mu_m3s2)
        X[i, 0:3] = np.array(rf) * _M2KM
        X[i, 3:6] = np.array(vf) * _M2KM

    T = np.array(tofs)
    return X, T


# =============================================================================
# FLYBY DELTA-V (wrapper around pykep.fb_dv)
# =============================================================================

def compute_flyby_dv(v_sc_in_km_s, v_sc_out_km_s, v_planet_km_s,
                     mu_planet_km3s2, safe_radius_km):
    """Compute the delta-v needed for a powered gravity assist.

    Parameters
    ----------
    v_sc_in_km_s : array (3,) — spacecraft incoming velocity [km/s]
    v_sc_out_km_s : array (3,) — spacecraft outgoing velocity [km/s]
    v_planet_km_s : array (3,) — planet velocity [km/s]
    mu_planet_km3s2 : float — planet's gravitational parameter [km^3/s^2]
    safe_radius_km : float — minimum flyby periapsis [km]

    Returns
    -------
    dv_km_s : float — required delta-v [km/s], 0 if unpowered flyby suffices
    """
    v_rel_in = (np.asarray(v_sc_in_km_s) - np.asarray(v_planet_km_s)) * _KM2M
    v_rel_out = (np.asarray(v_sc_out_km_s) - np.asarray(v_planet_km_s)) * _KM2M
    mu_m3s2 = mu_planet_km3s2 * 1e9
    safe_r_m = safe_radius_km * _KM2M

    # Try pykep fb_dv first, fall back to manual calculation
    if hasattr(pk, 'fb_dv'):
        dv_ms = pk.fb_dv(list(v_rel_in), list(v_rel_out), mu_m3s2, safe_r_m)
    else:
        # Manual powered flyby delta-v
        v_in_mag = np.linalg.norm(v_rel_in)
        v_out_mag = np.linalg.norm(v_rel_out)
        if v_in_mag < 1e-10 or v_out_mag < 1e-10:
            return 0.0
        cos_delta = np.clip(np.dot(v_rel_in, v_rel_out) / (v_in_mag * v_out_mag), -1, 1)
        delta_des = np.arccos(cos_delta)
        sin_arg = min(1.0, 1.0 / (1.0 + safe_r_m * v_in_mag**2 / mu_m3s2))
        delta_max = 2 * np.arcsin(sin_arg)
        if delta_des <= delta_max and abs(v_in_mag - v_out_mag) < 1e-6:
            return 0.0
        v_p_in = np.sqrt(v_in_mag**2 + 2 * mu_m3s2 / safe_r_m)
        v_p_out = np.sqrt(v_out_mag**2 + 2 * mu_m3s2 / safe_r_m)
        dv_ms = abs(v_p_out - v_p_in)
    return dv_ms * _M2KM


# =============================================================================
# SPICE KERNEL LOADER
# =============================================================================

def load_kernels(bsp_folder_name, generic_kernels_path):
    """Load generic + project-specific SPICE kernels, return asteroid list.

    Returns list of dicts with 'ID' (int), 'NAME' (str) keys.
    """
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'spk', 'planets', 'de430.bsp'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'pck', 'pck00010.tpc'))

    bsp_files = sorted(glob.glob(os.path.join(bsp_folder_name, '*.bsp')))
    asteroid_list = []
    for bsp_path in bsp_files:
        spiceypy.furnsh(bsp_path)
        id_cell = spiceypy.spkobj(bsp_path)
        # Extract integer NAIF IDs from the SpiceCell
        int_ids = [id_cell[i] for i in range(spiceypy.card(id_cell))]
        name = os.path.splitext(os.path.basename(bsp_path))[0]
        asteroid_list.append({'ID': int_ids[0] if int_ids else 0, 'NAME': name})
    return asteroid_list


def get_state(body_id, et):
    """Get heliocentric state [r_km (3,), v_km_s (3,)] at ephemeris time et."""
    state, _ = spiceypy.spkezr(str(body_id), et, 'ECLIPJ2000', 'NONE', '10')
    return np.array(state[0:3]), np.array(state[3:6])


def get_mu(body_id):
    """Get gravitational parameter for a body [km^3/s^2]."""
    _, vals = spiceypy.bodvcd(int(body_id), 'GM', 10)
    return vals[0]


def get_radius(body_id):
    """Get mean radius for a body [km]."""
    _, vals = spiceypy.bodvcd(int(body_id), 'RADII', 10)
    return vals[0]


# =============================================================================
# INPUT UNPACKING
# =============================================================================

def unpack_input(input_vec, launch_range):
    """Unpack 6-element optimization vector into epoch times (seconds)."""
    s = YEAR * input_vec
    et_launch = s[0] + launch_range[0]
    et_arrive_1 = et_launch + s[1]
    et_stay_1 = et_arrive_1 + s[2]
    et_arrive_2 = et_stay_1 + s[3]
    et_stay_2 = et_arrive_2 + s[4]
    et_arrive_3 = et_stay_2 + s[5]
    return et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3


def unpack_mars_input(input_vec, launch_range):
    """Unpack 7-element optimization vector (with Mars flyby) into epoch times."""
    s = YEAR * input_vec
    et_launch = s[0] + launch_range[0]
    et_mars = et_launch + s[1]
    et_arrive_1 = et_mars + s[2]
    et_stay_1 = et_arrive_1 + s[3]
    et_arrive_2 = et_stay_1 + s[4]
    et_stay_2 = et_arrive_2 + s[5]
    et_arrive_3 = et_stay_2 + s[6]
    return et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3


# =============================================================================
# ASTEROID NAME LOOKUP
# =============================================================================

def get_id_from_asteroid_name(asteroid_list, name):
    """Search asteroid_list for matching NAME, return ID or -1."""
    for asteroid in asteroid_list:
        if asteroid['NAME'] == name:
            return asteroid['ID']
    return -1
