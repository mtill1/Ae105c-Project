"""
optimization.py — Delta-v computation, scoring, time optimization, and data generation.

Uses scipy.optimize.differential_evolution for global optimization (replaces
the old grid search + L-BFGS-B approach). Includes two-level optimization
for scaling and beam search for structured path selection.

Combines: compute_path_deltav, compute_mars_path_deltav, score_paths, score_paths_mars,
          optimize_times, optimize_mars_times, generate_optimized_data,
          generate_mars_transfer_optimized, two_level_optimize, beam_search.
"""

import os
import pickle

import numpy as np
import spiceypy
from scipy.optimize import differential_evolution
from tqdm import tqdm

from lowthrust import DEFAULT_M_INIT_KG, DEFAULT_THRUST_N, ISP_ELEC, optimize_lt_leg
from core import (solve_lambert, solve_lambert_best, get_state, get_mu, get_radius,
                  estimate_ballistic_flyby_altitude_km,
                  DAY, YEAR, WEEK, MONTH, MU_SUN,
                  MAX_MISSION_DURATION, unpack_input, unpack_mars_input)

MAX_EARTH_DEPARTURE_DV = float(os.getenv("AE_MAX_EARTH_DEPARTURE_DV", "7.0"))  # km/s cap provided by launch system
LAUNCH_DV_TOL = 5e-4
LAUNCH_DV_GUIDE_TARGET = float(os.getenv("AE_LAUNCH_DV_GUIDE_TARGET", "6.95"))
UNPOWERED_VINF_TOL_KM_S = 1e-3
ELECTRIC_SCORE_PENALTY = 1e3
LT_NSEG = int(os.getenv("AE_LT_NSEG", "20"))
LT_MAX_NFEV = int(os.getenv("AE_LT_MAX_NFEV", "1500"))
LT_REG_WEIGHT = float(os.getenv("AE_LT_REG_WEIGHT", "0.05"))
MAX_LEG_YEARS = float(os.getenv("AE_MAX_LEG_YEARS", "6.0"))
MAX_STAY_YEARS = float(os.getenv("AE_MAX_STAY_YEARS", "3.0"))


def _mission_duration_cap_seconds():
    years = float(os.getenv("AE_MISSION_CAP_YEARS", "30"))
    if years <= 0:
        return np.inf
    return years * YEAR


def _duration_exceeds_cap(et_launch, et_arrive_3):
    cap = _mission_duration_cap_seconds()
    if not np.isfinite(cap):
        return False
    return (float(et_arrive_3) - float(et_launch)) > cap


def _duration_penalty(et_launch, et_arrive_3):
    """Soft penalty to guide search back under mission duration cap."""
    cap = _mission_duration_cap_seconds()
    if not np.isfinite(cap):
        return 0.0
    dur = float(et_arrive_3) - float(et_launch)
    if dur <= cap:
        return 0.0
    return 1e3 + (dur - cap) / YEAR


def _result_is_feasible(res):
    """True if candidate satisfies launch cap and is not a penalty result."""
    dv_total = float(res.get('delta_v_total', ELECTRIC_SCORE_PENALTY))
    if not np.isfinite(dv_total) or dv_total >= ELECTRIC_SCORE_PENALTY:
        return False
    dv_launch_vec = res.get('delta_v_launch', np.array([]))
    if len(dv_launch_vec) == 0:
        return False
    dv_launch = float(np.linalg.norm(dv_launch_vec))
    if not np.isfinite(dv_launch):
        return False
    if 'et_launch' in res and 'et_arrive_3' in res:
        if _duration_exceeds_cap(res['et_launch'], res['et_arrive_3']):
            return False
    return dv_launch <= MAX_EARTH_DEPARTURE_DV + LAUNCH_DV_TOL


def _penalty_result(reason, dv_launch=None, lambert0=np.nan, lambert1=np.nan, lambert2=np.nan):
    out = {
        'delta_v_launch': np.array([]) if dv_launch is None else dv_launch,
        'delta_v_A1_arrive': np.array([]),
        'delta_v_A1_leave': np.array([]),
        'delta_v_A2_arrive': np.array([]),
        'delta_v_A2_leave': np.array([]),
        'delta_v_A3_arrive': np.array([]),
        'delta_v_total': ELECTRIC_SCORE_PENALTY,
        'feasible': False,
        'fail_reason': reason,
        'lambert_leg0_dv_km_s': float(lambert0),
        'lambert_leg1_dv_km_s': float(lambert1),
        'lambert_leg2_dv_km_s': float(lambert2),
    }
    return out


def _launch_cap_penalty(dv_launch_km_s):
    """Graded penalty when launch dv exceeds cap (helps DE converge)."""
    excess = float(dv_launch_km_s) - MAX_EARTH_DEPARTURE_DV
    if excess <= 0:
        return ELECTRIC_SCORE_PENALTY
    return ELECTRIC_SCORE_PENALTY + 250.0 * excess


def _launch_guidance_penalty(dv_launch_km_s):
    """Soft guidance to keep coarse search comfortably below 7 km/s."""
    excess = float(dv_launch_km_s) - LAUNCH_DV_GUIDE_TARGET
    if excess <= 0:
        return 0.0
    # Quadratic penalty pushes DE away from cap boundary.
    return 5000.0 * (excess ** 2)


def _flyby_geometry_penalty(flyby_alt_km, min_alt_km, turn_margin_rad):
    """Graded penalty for flyby geometry violations (altitude / turn angle)."""
    pen = ELECTRIC_SCORE_PENALTY
    if not np.isfinite(flyby_alt_km):
        pen += 600.0
    else:
        alt_deficit = max(0.0, float(min_alt_km) - float(flyby_alt_km))
        # 1 km deficit -> +2 score units
        pen += 2.0 * alt_deficit
    # Negative margin means required turn exceeds ballistic maximum.
    if not np.isfinite(turn_margin_rad):
        pen += 400.0
    elif turn_margin_rad < 0.0:
        # ~57 deg shortfall (1 rad) -> +200 score units
        pen += 200.0 * abs(float(turn_margin_rad))
    return pen


def _electric_leg_surrogate_factor(dv_lambert_km_s, tof_sec, mass_kg, thrust_N=DEFAULT_THRUST_N):
    """Fast surrogate factor for electric legs; inf means thrust-infeasible."""
    if tof_sec <= 0 or mass_kg <= 0:
        return np.inf
    dv_ceiling = thrust_N * tof_sec / mass_kg / 1e3
    if dv_ceiling <= 0:
        return np.inf
    ratio = dv_lambert_km_s / dv_ceiling
    if ratio >= 0.95:
        return np.inf
    base = 1.20 + 0.55 * ratio
    blowup = 0.40 / (1.0 - ratio)
    factor = base + blowup - 0.40
    tof_yr = tof_sec / YEAR
    if tof_yr < 1.0:
        factor *= (2.0 - tof_yr)
    return factor


def _surrogate_mass_after_leg(m_in_kg, dv_km_s, isp_s=ISP_ELEC):
    g0 = 9.80665
    return float(m_in_kg * np.exp(-dv_km_s * 1000.0 / (isp_s * g0)))


def _unpowered_turn_feasible(v_sc_in_km_s, v_sc_out_km_s, v_body_km_s, mu_body_km3_s2, safe_radius_km):
    """True if a strictly ballistic flyby is feasible at/above safe_radius_km."""
    v_rel_in = np.asarray(v_sc_in_km_s, dtype=float) - np.asarray(v_body_km_s, dtype=float)
    v_rel_out = np.asarray(v_sc_out_km_s, dtype=float) - np.asarray(v_body_km_s, dtype=float)
    v_in = np.linalg.norm(v_rel_in)
    v_out = np.linalg.norm(v_rel_out)
    if v_in < 1e-12 or v_out < 1e-12:
        return False
    # Ballistic flyby conserves |v_inf|.
    if abs(v_in - v_out) > UNPOWERED_VINF_TOL_KM_S:
        return False
    cos_delta = np.clip(np.dot(v_rel_in, v_rel_out) / (v_in * v_out), -1.0, 1.0)
    delta_required = float(np.arccos(cos_delta))
    sin_half_max = 1.0 / (1.0 + safe_radius_km * (v_in ** 2) / mu_body_km3_s2)
    sin_half_max = float(np.clip(sin_half_max, 0.0, 1.0))
    delta_max = 2.0 * np.arcsin(sin_half_max)
    return bool(delta_required <= delta_max + 1e-9)


def _unpowered_turn_margin(v_sc_in_km_s, v_sc_out_km_s, v_body_km_s, mu_body_km3_s2, safe_radius_km):
    """Return (delta_max - delta_required) in radians for ballistic flyby."""
    v_rel_in = np.asarray(v_sc_in_km_s, dtype=float) - np.asarray(v_body_km_s, dtype=float)
    v_rel_out = np.asarray(v_sc_out_km_s, dtype=float) - np.asarray(v_body_km_s, dtype=float)
    v_in = np.linalg.norm(v_rel_in)
    v_out = np.linalg.norm(v_rel_out)
    if v_in < 1e-12 or v_out < 1e-12:
        return -np.inf
    if abs(v_in - v_out) > UNPOWERED_VINF_TOL_KM_S:
        return -np.inf
    cos_delta = np.clip(np.dot(v_rel_in, v_rel_out) / (v_in * v_out), -1.0, 1.0)
    delta_required = float(np.arccos(cos_delta))
    sin_half_max = 1.0 / (1.0 + safe_radius_km * (v_in ** 2) / mu_body_km3_s2)
    sin_half_max = float(np.clip(sin_half_max, 0.0, 1.0))
    delta_max = 2.0 * np.arcsin(sin_half_max)
    return float(delta_max - delta_required)


# =============================================================================
# DELTA-V COMPUTATION
# =============================================================================

def compute_path_deltav(a_id_1, a_id_2, a_id_3, et_launch,
                        et_arrive_1, et_stay_1, et_arrive_2,
                        et_stay_2, et_arrive_3, m_1, m_2, m_3):
    try:
        earth_launch_r, earth_launch_v = get_state('399', et_launch)
        a_1_arrive_r, a_1_arrive_v = get_state(str(a_id_1), et_arrive_1)
        a_1_leaving_r, a_1_leaving_v = get_state(str(a_id_1), et_stay_1)
        a_2_arrive_r, a_2_arrive_v = get_state(str(a_id_2), et_arrive_2)
        a_2_leaving_r, a_2_leaving_v = get_state(str(a_id_2), et_stay_2)
        a_3_arrive_r, a_3_arrive_v = get_state(str(a_id_3), et_arrive_3)
    except Exception:
        return _penalty_result('state_lookup_failed')

    earth_lv, a1_arr_lv, ef1 = solve_lambert(
        earth_launch_r, a_1_arrive_r, (et_arrive_1 - et_launch) / DAY, m_1, MU_SUN)
    a1_lv_lv, a2_arr_lv, ef2 = solve_lambert(
        a_1_leaving_r, a_2_arrive_r, (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    a2_lv_lv, a3_arr_lv, ef3 = solve_lambert(
        a_2_leaving_r, a_3_arrive_r, (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if ef1 != 1 or ef2 != 1 or ef3 != 1:
        return _penalty_result('lambert_failed')

    dv_launch = earth_lv - earth_launch_v
    if np.linalg.norm(dv_launch) > MAX_EARTH_DEPARTURE_DV + LAUNCH_DV_TOL:
        return _penalty_result('launch_dv_cap_exceeded', dv_launch=dv_launch)
    dv0_lam = float(np.linalg.norm(a1_arr_lv - earth_lv))
    dv1_lam = float(np.linalg.norm(a2_arr_lv - a1_lv_lv))
    dv2_lam = float(np.linalg.norm(a3_arr_lv - a2_lv_lv))
    # Post-launch mission is electric-only (no impulsive burns after launch):
    # solve each transfer as a low-thrust leg matching body states.
    m_curr = DEFAULT_M_INIT_KG
    lt_1 = optimize_lt_leg(
        earth_launch_r, earth_launch_v + dv_launch, a_1_arrive_r, a_1_arrive_v,
        et_arrive_1 - et_launch, m_init_kg=m_curr, thrust_N=DEFAULT_THRUST_N, isp_s=ISP_ELEC,
        nseg=LT_NSEG, max_nfev=LT_MAX_NFEV, reg_weight=LT_REG_WEIGHT)
    if not lt_1.get('converged', False):
        return _penalty_result('lt_leg0_failed', dv_launch=dv_launch,
                               lambert0=dv0_lam, lambert1=dv1_lam, lambert2=dv2_lam)
    m_curr = float(lt_1['m_final'])

    lt_2 = optimize_lt_leg(
        a_1_leaving_r, a_1_leaving_v, a_2_arrive_r, a_2_arrive_v,
        et_arrive_2 - et_stay_1, m_init_kg=m_curr, thrust_N=DEFAULT_THRUST_N, isp_s=ISP_ELEC,
        nseg=LT_NSEG, max_nfev=LT_MAX_NFEV, reg_weight=LT_REG_WEIGHT)
    if not lt_2.get('converged', False):
        return _penalty_result('lt_leg1_failed', dv_launch=dv_launch,
                               lambert0=dv0_lam, lambert1=dv1_lam, lambert2=dv2_lam)
    m_curr = float(lt_2['m_final'])

    lt_3 = optimize_lt_leg(
        a_2_leaving_r, a_2_leaving_v, a_3_arrive_r, a_3_arrive_v,
        et_arrive_3 - et_stay_2, m_init_kg=m_curr, thrust_N=DEFAULT_THRUST_N, isp_s=ISP_ELEC,
        nseg=LT_NSEG, max_nfev=LT_MAX_NFEV, reg_weight=LT_REG_WEIGHT)
    if not lt_3.get('converged', False):
        return _penalty_result('lt_leg2_failed', dv_launch=dv_launch,
                               lambert0=dv0_lam, lambert1=dv1_lam, lambert2=dv2_lam)

    dv_total = float(lt_1['dv_integral_kms'] + lt_2['dv_integral_kms'] + lt_3['dv_integral_kms'])

    return {'delta_v_launch': dv_launch, 'delta_v_A1_arrive': np.zeros(3),
            'delta_v_A1_leave': np.zeros(3), 'delta_v_A2_arrive': np.zeros(3),
            'delta_v_A2_leave': np.zeros(3), 'delta_v_A3_arrive': np.zeros(3),
            'lt_leg0_dv': float(lt_1['dv_integral_kms']),
            'lt_leg1_dv': float(lt_2['dv_integral_kms']),
            'lt_leg2_dv': float(lt_3['dv_integral_kms']),
            'lt_final_mass_kg': float(lt_3['m_final']),
            'lambert_leg0_dv_km_s': dv0_lam,
            'lambert_leg1_dv_km_s': dv1_lam,
            'lambert_leg2_dv_km_s': dv2_lam,
            'feasible': True,
            'fail_reason': '',
            'delta_v_total': dv_total}


def compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                             et_arrive_1, et_stay_1, et_arrive_2,
                             et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars):
    try:
        earth_r, earth_v = get_state('399', et_launch)
        mars_r, mars_v = get_state('4', et_mars)
        a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
        a1_lv_r, a1_lv_v = get_state(str(a_id_1), et_stay_1)
        a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
        a2_lv_r, a2_lv_v = get_state(str(a_id_2), et_stay_2)
        a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)
    except Exception:
        return {'delta_v_launch': np.array([]), 'v_mars_leave': np.array([]),
                'delta_v_mars': 1e3, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3, 'feasible': False, 'fail_reason': 'state_lookup_failed'}

    mu_mars = get_mu(4)
    mars_radius = get_radius(499)
    min_alt_mars = 200.0
    safe_radius = mars_radius + min_alt_mars  # 200 km altitude

    e_lv, mars_arr_v_lam, ef1 = solve_lambert(earth_r, mars_r, -(et_mars-et_launch)/DAY, m_mars, MU_SUN)
    v_mars_leave, a1_arr_lv, ef2 = solve_lambert(mars_r, a1_arr_r, -(et_arrive_1-et_mars)/DAY, m_1, MU_SUN)
    a1_lv_lv, a2_arr_lv, ef3 = solve_lambert(a1_lv_r, a2_arr_r, -(et_arrive_2-et_stay_1)/DAY, m_2, MU_SUN)
    a2_lv_lv, a3_arr_lv, ef4 = solve_lambert(a2_lv_r, a3_arr_r, -(et_arrive_3-et_stay_2)/DAY, m_3, MU_SUN)

    if ef1 != 1 or ef2 != 1 or ef3 != 1 or ef4 != 1:
        return {'delta_v_launch': np.array([]), 'v_mars_leave': np.array([]),
                'delta_v_mars': 1e3, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3, 'feasible': False, 'fail_reason': 'lambert_failed'}

    dv_launch = e_lv - earth_v
    dv_launch_norm = float(np.linalg.norm(dv_launch))
    if dv_launch_norm > MAX_EARTH_DEPARTURE_DV + LAUNCH_DV_TOL:
        launch_pen = _launch_cap_penalty(dv_launch_norm)
        return {'delta_v_launch': dv_launch, 'v_mars_leave': np.array([]),
                'delta_v_mars': launch_pen, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': launch_pen, 'feasible': False, 'fail_reason': 'launch_dv_cap_exceeded'}
    dv_A1_arr = a1_arr_lv - a1_arr_v
    dv_A1_lv = a1_lv_lv - a1_lv_v
    dv_A2_arr = a2_arr_lv - a2_arr_v
    dv_A2_lv = a2_lv_lv - a2_lv_v
    dv_A3_arr = a3_arr_lv - a3_arr_v
    flyby_alt_km = estimate_ballistic_flyby_altitude_km(
        mars_arr_v_lam, v_mars_leave, mars_v, mu_mars, mars_radius)
    if not np.isfinite(flyby_alt_km) or flyby_alt_km < min_alt_mars:
        return {'delta_v_launch': dv_launch, 'v_mars_leave': v_mars_leave,
                'delta_v_mars': 1e3, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3, 'feasible': False, 'fail_reason': 'flyby_altitude_violation'}
    if not _unpowered_turn_feasible(mars_arr_v_lam, v_mars_leave, mars_v, mu_mars, safe_radius):
        return {'delta_v_launch': dv_launch, 'v_mars_leave': v_mars_leave,
                'delta_v_mars': 1e3, 'delta_v_A1_arrive': np.array([]),
                'delta_v_A1_leave': np.array([]), 'delta_v_A2_arrive': np.array([]),
                'delta_v_A2_leave': np.array([]), 'delta_v_A3_arrive': np.array([]),
                'delta_v_total': 1e3, 'feasible': False, 'fail_reason': 'unpowered_turn_infeasible'}
    dv_mars = 0.0

    # Objective excludes Earth departure Δv (provided by launch system, capped above).
    dv_total = (np.linalg.norm(dv_A1_arr) + np.linalg.norm(dv_A1_lv) + np.linalg.norm(dv_A2_arr)
                + np.linalg.norm(dv_A2_lv) + np.linalg.norm(dv_A3_arr))

    return {'delta_v_launch': dv_launch, 'v_mars_leave': v_mars_leave,
            'delta_v_mars': dv_mars, 'delta_v_A1_arrive': dv_A1_arr,
            'delta_v_A1_leave': dv_A1_lv, 'delta_v_A2_arrive': dv_A2_arr,
            'delta_v_A2_leave': dv_A2_lv, 'delta_v_A3_arrive': dv_A3_arr,
            'delta_v_total': dv_total, 'feasible': True, 'fail_reason': ''}


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_paths(input_vec, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_input(input_vec, launch_range)
    dur_pen = _duration_penalty(et_launch, et_arrive_3)
    if dur_pen > 0:
        return dur_pen
    # Fast electric-only surrogate for DE: launch impulsive cap + electric post-launch legs.
    try:
        earth_launch_r, earth_launch_v = get_state('399', et_launch)
        a_1_arrive_r, a_1_arrive_v = get_state(str(a_id_1), et_arrive_1)
        a_1_leaving_r, a_1_leaving_v = get_state(str(a_id_1), et_stay_1)
        a_2_arrive_r, a_2_arrive_v = get_state(str(a_id_2), et_arrive_2)
        a_2_leaving_r, a_2_leaving_v = get_state(str(a_id_2), et_stay_2)
        a_3_arrive_r, a_3_arrive_v = get_state(str(a_id_3), et_arrive_3)
    except Exception:
        return ELECTRIC_SCORE_PENALTY

    earth_lv, a1_arr_lv, ef1 = solve_lambert(
        earth_launch_r, a_1_arrive_r, (et_arrive_1 - et_launch) / DAY, m_1, MU_SUN)
    a1_lv_lv, a2_arr_lv, ef2 = solve_lambert(
        a_1_leaving_r, a_2_arrive_r, (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    a2_lv_lv, a3_arr_lv, ef3 = solve_lambert(
        a_2_leaving_r, a_3_arrive_r, (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)
    if ef1 != 1 or ef2 != 1 or ef3 != 1:
        return ELECTRIC_SCORE_PENALTY

    dv_launch = earth_lv - earth_launch_v
    dv_launch_norm = float(np.linalg.norm(dv_launch))
    if dv_launch_norm > MAX_EARTH_DEPARTURE_DV + LAUNCH_DV_TOL:
        return _launch_cap_penalty(dv_launch_norm)
    launch_guidance = _launch_guidance_penalty(dv_launch_norm)

    m_curr = DEFAULT_M_INIT_KG
    dv0_lam = float(np.linalg.norm((earth_lv - (earth_launch_v + dv_launch))) + np.linalg.norm(a1_arr_lv - a_1_arrive_v))
    dv1_lam = float(np.linalg.norm(a1_lv_lv - a_1_leaving_v) + np.linalg.norm(a2_arr_lv - a_2_arrive_v))
    dv2_lam = float(np.linalg.norm(a2_lv_lv - a_2_leaving_v) + np.linalg.norm(a3_arr_lv - a_3_arrive_v))
    tof0 = et_arrive_1 - et_launch
    tof1 = et_arrive_2 - et_stay_1
    tof2 = et_arrive_3 - et_stay_2

    f0 = _electric_leg_surrogate_factor(dv0_lam, tof0, m_curr)
    if not np.isfinite(f0):
        return ELECTRIC_SCORE_PENALTY
    dv0 = dv0_lam * f0
    m_curr = _surrogate_mass_after_leg(m_curr, dv0)

    f1 = _electric_leg_surrogate_factor(dv1_lam, tof1, m_curr)
    if not np.isfinite(f1):
        return ELECTRIC_SCORE_PENALTY
    dv1 = dv1_lam * f1
    m_curr = _surrogate_mass_after_leg(m_curr, dv1)

    f2 = _electric_leg_surrogate_factor(dv2_lam, tof2, m_curr)
    if not np.isfinite(f2):
        return ELECTRIC_SCORE_PENALTY
    dv2 = dv2_lam * f2

    return float(dv0 + dv1 + dv2 + launch_guidance)

def score_paths_mars(input_vec, a_id_1, a_id_2, a_id_3, launch_range,
                     m_1, m_2, m_3, m_mars):
    et_launch, et_mars, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3 = \
        unpack_mars_input(input_vec, launch_range)
    dur_pen = _duration_penalty(et_launch, et_arrive_3)
    if dur_pen > 0:
        return dur_pen
    res = compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, et_launch, et_mars,
                                   et_arrive_1, et_stay_1, et_arrive_2,
                                   et_stay_2, et_arrive_3, m_1, m_2, m_3, m_mars)
    dv = float(res['delta_v_total'])
    dv_launch_vec = res.get('delta_v_launch', np.array([]))
    if len(dv_launch_vec) > 0:
        dv += _launch_guidance_penalty(float(np.linalg.norm(dv_launch_vec)))
    return dv


# =============================================================================
# TIME OPTIMIZATION (differential_evolution replaces grid search)
# =============================================================================

def optimize_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    leg_max = max(0.25 * YEAR, MAX_LEG_YEARS * YEAR)
    stay_max = max(0.25 * YEAR, MAX_STAY_YEARS * YEAR)
    bounds = list(zip(
        np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], leg_max, stay_max, leg_max, stay_max, leg_max]) / YEAR))

    res = differential_evolution(
        lambda x: score_paths(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3),
        bounds, maxiter=300, tol=1e-7, seed=42, polish=True, updating='deferred')

    ets = unpack_input(res.x, launch_range)
    result = compute_path_deltav(a_id_1, a_id_2, a_id_3, *ets, m_1, m_2, m_3)
    result.update(dict(zip(['et_launch','et_arrive_1','et_stay_1',
                            'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    if _duration_exceeds_cap(result['et_launch'], result['et_arrive_3']):
        return _penalty_result('mission_duration_cap_exceeded')
    return result


def optimize_times_quick(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3):
    """Fast coarse optimization for two-level search first pass."""
    leg_max = max(0.25 * YEAR, MAX_LEG_YEARS * YEAR)
    stay_max = max(0.25 * YEAR, MAX_STAY_YEARS * YEAR)
    bounds = list(zip(
        np.array([0, 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], leg_max, stay_max, leg_max, stay_max, leg_max]) / YEAR))
    res = differential_evolution(
        lambda x: score_paths(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3),
        bounds, maxiter=30, tol=0.1, seed=42, polish=False, popsize=5, updating='deferred')
    return res.fun


def optimize_mars_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3, m_mars):
    bounds = list(zip(
        np.array([0, 3*YEAR, 0.2*YEAR, 3*MONTH, 0.2*YEAR, 3*MONTH, 0.2*YEAR]) / YEAR,
        np.array([launch_range[1]-launch_range[0], 5*YEAR, 4*YEAR, YEAR, 4*YEAR, YEAR, 4*YEAR]) / YEAR))

    res = differential_evolution(
        lambda x: score_paths_mars(x, a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3, m_mars),
        bounds, maxiter=300, tol=1e-6, seed=42, polish=True, updating='deferred')

    ets = unpack_mars_input(res.x, launch_range)
    result = compute_mars_path_deltav(a_id_1, a_id_2, a_id_3, *ets, m_1, m_2, m_3, m_mars)
    result.update(dict(zip(['et_launch','et_mars','et_arrive_1','et_stay_1',
                            'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    if _duration_exceeds_cap(result['et_launch'], result['et_arrive_3']):
        return {'delta_v_launch': np.array([]), 'delta_v_total': 1e3,
                'feasible': False, 'fail_reason': 'mission_duration_cap_exceeded'}
    return result


# =============================================================================
# GRAVITY ASSIST TRAJECTORIES
# =============================================================================

# Flyby body parameters
FLYBY_BODIES = {
    'moon':  {'id': '301', 'mu_body': 301,  'radii_body': 301, 'min_alt': 100,
              'tof_min': 1*DAY, 'tof_max': 10*DAY},     # lunar flyby: 1-10 days after launch
    'mars':  {'id': '4',   'mu_body': 4,    'radii_body': 499, 'min_alt': 200,
              'tof_min': 0.3*YEAR, 'tof_max': 3*YEAR},   # Mars flyby: months to years
    'earth': {'id': '399', 'mu_body': 399,  'radii_body': 399, 'min_alt': 300,
              'tof_min': 1.0*YEAR, 'tof_max': 3.0*YEAR},  # EGA loop: 1-3 years
}


def compute_path_with_flyby(a_id_1, a_id_2, a_id_3, et_launch, et_flyby,
                            et_arrive_1, et_stay_1, et_arrive_2, et_stay_2,
                            et_arrive_3, flyby_name, m_0, m_1, m_2, m_3):
    """Compute delta-v for Earth -> flyby_body -> A1 -> A2 -> A3.

    Parameters
    ----------
    flyby_name : str — 'moon' or 'mars'
    m_0 : int — Lambert parameter for Earth -> flyby leg
    m_1, m_2, m_3 : int — Lambert parameters for flyby->A1, A1->A2, A2->A3
    """
    fb = FLYBY_BODIES[flyby_name]
    fb_id = fb['id']

    try:
        earth_r, earth_v = get_state('399', et_launch)
        flyby_r, flyby_v = get_state(fb_id, et_flyby)
        a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
        a1_lv_r, a1_lv_v = get_state(str(a_id_1), et_stay_1)
        a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
        a2_lv_r, a2_lv_v = get_state(str(a_id_2), et_stay_2)
        a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)
    except Exception:
        return {'delta_v_total': 1e3, 'delta_v_launch': np.array([]),
                'architecture': flyby_name, 'feasible': False, 'fail_reason': 'state_lookup_failed'}

    mu_flyby = get_mu(fb['mu_body'])
    body_radius = get_radius(fb['radii_body'])
    safe_radius = body_radius + fb['min_alt']

    # Leg 0: Earth -> flyby body. For a REAL Earth gravity assist the spacecraft
    # must (a) fly a visibly different heliocentric orbit (v∞ >= 3 km/s, else it
    # co-orbits with Earth) and (b) get most of the turning from Earth's gravity
    # rather than a burn (powered Δv at periapsis kept small — enforced later).
    if flyby_name == 'earth':
        MIN_EGA_VINF = 3.0  # km/s — matches real-mission heritage (Dawn, MESSENGER, Galileo)
        best_launch_dv = np.inf
        e_lv, fb_arr_lv, ef0 = np.zeros(3), np.zeros(3), -1
        for m_try in (0, 1, -1, 2, -2):
            V1, V2, ef = solve_lambert(earth_r, flyby_r,
                                        (et_flyby - et_launch) / DAY, m_try, MU_SUN)
            if ef != 1:
                continue
            ldv = np.linalg.norm(V1 - earth_v)
            if ldv < MIN_EGA_VINF:
                continue  # degenerate phasing orbit
            if ldv < best_launch_dv:
                best_launch_dv = ldv
                e_lv, fb_arr_lv, ef0 = V1, V2, 1
    else:
        e_lv, fb_arr_lv, ef0 = solve_lambert(earth_r, flyby_r,
                                              (et_flyby - et_launch) / DAY, m_0, MU_SUN)
    # Leg 1: flyby body -> A1
    fb_dep_lv, a1_arr_lv, ef1 = solve_lambert(flyby_r, a1_arr_r,
                                               (et_arrive_1 - et_flyby) / DAY, m_1, MU_SUN)
    # Leg 2: A1 -> A2
    a1_lv_lv, a2_arr_lv, ef2 = solve_lambert(a1_lv_r, a2_arr_r,
                                              (et_arrive_2 - et_stay_1) / DAY, m_2, MU_SUN)
    # Leg 3: A2 -> A3
    a2_lv_lv, a3_arr_lv, ef3 = solve_lambert(a2_lv_r, a3_arr_r,
                                              (et_arrive_3 - et_stay_2) / DAY, m_3, MU_SUN)

    if ef0 != 1 or ef1 != 1 or ef2 != 1 or ef3 != 1:
        return {'delta_v_total': 1e3, 'delta_v_launch': np.array([]),
                'architecture': flyby_name, 'feasible': False, 'fail_reason': 'lambert_failed'}

    dv_launch = e_lv - earth_v
    dv_launch_norm = float(np.linalg.norm(dv_launch))
    if dv_launch_norm > MAX_EARTH_DEPARTURE_DV + LAUNCH_DV_TOL:
        launch_pen = _launch_cap_penalty(dv_launch_norm)
        return {'delta_v_total': launch_pen, 'delta_v_launch': dv_launch,
                'architecture': flyby_name, 'feasible': False, 'fail_reason': 'launch_dv_cap_exceeded'}
    flyby_alt_km = estimate_ballistic_flyby_altitude_km(
        fb_arr_lv, fb_dep_lv, flyby_v, mu_flyby, body_radius)
    turn_margin = _unpowered_turn_margin(fb_arr_lv, fb_dep_lv, flyby_v, mu_flyby, safe_radius)
    if not np.isfinite(flyby_alt_km) or flyby_alt_km < fb['min_alt']:
        geom_pen = _flyby_geometry_penalty(flyby_alt_km, fb['min_alt'], turn_margin)
        return {'delta_v_total': geom_pen, 'delta_v_launch': dv_launch,
                'architecture': flyby_name, 'feasible': False, 'fail_reason': 'flyby_altitude_violation'}
    if turn_margin < 0.0:
        geom_pen = _flyby_geometry_penalty(flyby_alt_km, fb['min_alt'], turn_margin)
        return {'delta_v_total': geom_pen, 'delta_v_launch': dv_launch,
                'architecture': flyby_name, 'feasible': False, 'fail_reason': 'unpowered_turn_infeasible'}
    dv_flyby = 0.0
    dv_A1_arr = a1_arr_lv - a1_arr_v
    dv_A1_lv = a1_lv_lv - a1_lv_v
    dv_A2_arr = a2_arr_lv - a2_arr_v
    dv_A2_lv = a2_lv_lv - a2_lv_v
    dv_A3_arr = a3_arr_lv - a3_arr_v

    # Objective excludes Earth departure Δv (provided by launch system, capped above).
    dv_total = (np.linalg.norm(dv_A1_arr) + np.linalg.norm(dv_A1_lv)
                + np.linalg.norm(dv_A2_arr) + np.linalg.norm(dv_A2_lv)
                + np.linalg.norm(dv_A3_arr))

    return {
        'delta_v_launch': dv_launch, 'delta_v_flyby': dv_flyby,
        'delta_v_A1_arrive': dv_A1_arr, 'delta_v_A1_leave': dv_A1_lv,
        'delta_v_A2_arrive': dv_A2_arr, 'delta_v_A2_leave': dv_A2_lv,
        'delta_v_A3_arrive': dv_A3_arr, 'delta_v_total': dv_total,
        'architecture': flyby_name,
        'flyby_body': fb_id, 'et_flyby': et_flyby,
        'flyby_altitude_km': flyby_alt_km,
        # Heliocentric SC velocity (km/s) at flyby exit — matches patched Lambert Mars/Moon→A1 leg.
        'v_sc_post_flyby': np.asarray(fb_dep_lv, dtype=float),
        'feasible': True, 'fail_reason': '',
    }


def _unpack_flyby_input(input_vec, launch_range, flyby_name):
    """Unpack 7-element vector for flyby trajectory."""
    s = YEAR * input_vec
    et_launch = s[0] + launch_range[0]
    et_flyby = et_launch + s[1]
    et_arrive_1 = et_flyby + s[2]
    et_stay_1 = et_arrive_1 + s[3]
    et_arrive_2 = et_stay_1 + s[4]
    et_stay_2 = et_arrive_2 + s[5]
    et_arrive_3 = et_stay_2 + s[6]
    return et_launch, et_flyby, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2, et_arrive_3


def score_paths_flyby(input_vec, a_id_1, a_id_2, a_id_3, launch_range,
                      flyby_name, m_0, m_1, m_2, m_3):
    """Score function for flyby trajectory."""
    ets = _unpack_flyby_input(input_vec, launch_range, flyby_name)
    dur_pen = _duration_penalty(ets[0], ets[6])
    if dur_pen > 0:
        return dur_pen
    res = compute_path_with_flyby(a_id_1, a_id_2, a_id_3, *ets,
                                  flyby_name, m_0, m_1, m_2, m_3)
    dv = float(res['delta_v_total'])
    dv_launch_vec = res.get('delta_v_launch', np.array([]))
    if len(dv_launch_vec) > 0:
        dv += _launch_guidance_penalty(float(np.linalg.norm(dv_launch_vec)))
    return dv


def optimize_times_flyby(a_id_1, a_id_2, a_id_3, launch_range, flyby_name,
                         m_0=0, m_1=0, m_2=0, m_3=0):
    """Full optimization for a flyby trajectory."""
    fb = FLYBY_BODIES[flyby_name]
    leg_max = max(0.25 * YEAR, MAX_LEG_YEARS * YEAR)
    stay_max = max(0.25 * YEAR, MAX_STAY_YEARS * YEAR)
    bounds = list(zip(
        np.array([0, fb['tof_min'], 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], fb['tof_max'], leg_max, stay_max,
                  leg_max, stay_max, leg_max]) / YEAR))

    res = differential_evolution(
        lambda x: score_paths_flyby(x, a_id_1, a_id_2, a_id_3, launch_range,
                                    flyby_name, m_0, m_1, m_2, m_3),
        bounds, maxiter=300, tol=1e-7, seed=42, polish=True, updating='deferred')

    ets = _unpack_flyby_input(res.x, launch_range, flyby_name)
    result = compute_path_with_flyby(a_id_1, a_id_2, a_id_3, *ets,
                                     flyby_name, m_0, m_1, m_2, m_3)
    result.update(dict(zip(['et_launch','et_flyby','et_arrive_1','et_stay_1',
                            'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
    if _duration_exceeds_cap(result['et_launch'], result['et_arrive_3']):
        return {'delta_v_total': 1e3, 'delta_v_launch': np.array([]),
                'architecture': flyby_name, 'feasible': False,
                'fail_reason': 'mission_duration_cap_exceeded',
                'et_launch': result['et_launch'], 'et_arrive_3': result['et_arrive_3']}
    return result


def optimize_times_flyby_quick(a_id_1, a_id_2, a_id_3, launch_range, flyby_name):
    """Fast coarse evaluation for flyby trajectory."""
    fb = FLYBY_BODIES[flyby_name]
    best_dv = 1e3
    for launch_frac in [0.1, 0.4, 0.7]:
        # Flyby timing depends on body
        if flyby_name == 'moon':
            flyby_tofs = [3*DAY/YEAR, 5*DAY/YEAR]  # 3-5 days for lunar flyby
        elif flyby_name == 'earth':
            flyby_tofs = [1.1, 1.5, 2.0]           # 1-2 yr EGA return loop
        else:
            flyby_tofs = [0.6, 1.2]  # 0.6-1.2 years for Mars flyby
        for fb_tof in flyby_tofs:
            for tof in [1.5, 2.5, 4.0]:
                x = np.array([
                    launch_frac * (launch_range[1] - launch_range[0]) / YEAR,
                    fb_tof, tof, 0.4, tof, 0.4, tof,
                ])
                dv = score_paths_flyby(x, a_id_1, a_id_2, a_id_3, launch_range,
                                       flyby_name, 0, 0, 0, 0)
                if dv < best_dv:
                    best_dv = dv
    return best_dv


def optimize_best_architecture(a_id_1, a_id_2, a_id_3, launch_range,
                               m_1=0, m_2=0, m_3=0, quick=False,
                               allowed_architectures=None):
    """Try direct + Moon flyby + Mars flyby, return the best result.

    Returns the result dict with an extra 'architecture' key:
    'direct', 'moon', or 'mars'.
    """
    results = {}
    if allowed_architectures is None:
        allowed_architectures = ['direct', 'moon', 'mars', 'earth']
    allowed = set(allowed_architectures)

    if quick:
        # Coarse screening — sample evaluations only
        if 'direct' in allowed:
            results['direct'] = optimize_times_quick(a_id_1, a_id_2, a_id_3,
                                                     launch_range, m_1, m_2, m_3)
        for fb_name in ['moon', 'mars', 'earth']:
            if fb_name in allowed:
                results[fb_name] = optimize_times_flyby_quick(a_id_1, a_id_2, a_id_3,
                                                              launch_range, fb_name)
        if not results:
            return 1e3, 'none'
        # Return best dv
        best_arch = min(results, key=results.get)
        return results[best_arch], best_arch
    else:
        # Full optimization
        if 'direct' in allowed:
            direct = optimize_times(a_id_1, a_id_2, a_id_3, launch_range, m_1, m_2, m_3)
            direct['architecture'] = 'direct'
            results['direct'] = direct

        for fb_name in ['moon', 'mars', 'earth']:
            if fb_name not in allowed:
                continue
            try:
                fb_result = optimize_times_flyby(a_id_1, a_id_2, a_id_3,
                                                 launch_range, fb_name, 0, m_1, m_2, m_3)
                results[fb_name] = fb_result
            except Exception:
                results[fb_name] = {'delta_v_total': 1e3, 'architecture': fb_name}
        if not results:
            return {'delta_v_total': 1e3, 'architecture': 'none'}, 'none'

        best_arch = min(results, key=lambda k: results[k]['delta_v_total'])
        return results[best_arch], best_arch


# =============================================================================
# DATA GENERATION — STANDARD (brute force N^3)
# =============================================================================

def generate_optimized_data(asteroid_list, m_1, m_2, m_3,
                            launch_utc_min, launch_utc_max,
                            comp_map=None, required_compositions=None):
    num = len(asteroid_list)
    default = {'delta_v_total': np.inf, 'et_launch': np.inf}
    data = [[[dict(default) for _ in range(num)] for _ in range(num)] for _ in range(num)]

    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    pbar = tqdm(total=num**3, desc=f'Ms: {m_1},{m_2},{m_3}')
    for i in range(num):
        for j in range(num):
            for k in range(num):
                ids = [asteroid_list[x]['ID'] for x in [i, j, k]]
                if ids[0]==ids[1] or ids[1]==ids[2] or ids[2]==ids[0]:
                    pbar.update(1); continue
                if comp_map is not None:
                    if not _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                            comp_map, required_compositions):
                        pbar.update(1); continue
                strs = [str(int(x)) for x in ids]
                data[i][j][k] = optimize_times(*strs, launch_dates, m_1, m_2, m_3)
                pbar.update(1)
    pbar.close()

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    path = f'./optimal_asteroid_paths/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(path, 'wb') as f: pickle.dump(data, f)
    return data


def generate_mars_transfer_optimized(asteroid_list, m_1, m_2, m_3, m_mars,
                                     launch_utc_min, launch_utc_max,
                                     comp_map=None, required_compositions=None):
    num = len(asteroid_list)
    default = {'delta_v_total': np.inf, 'et_launch': np.inf}
    data = [[[dict(default) for _ in range(num)] for _ in range(num)] for _ in range(num)]

    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    pbar = tqdm(total=num**3, desc=f'Ms: {m_mars},{m_1},{m_2},{m_3}')
    for i in range(num):
        for j in range(num):
            for k in range(num):
                ids = [asteroid_list[x]['ID'] for x in [i, j, k]]
                if ids[0]==ids[1] or ids[1]==ids[2] or ids[2]==ids[0]:
                    pbar.update(1); continue
                if comp_map is not None:
                    if not _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                            comp_map, required_compositions):
                        pbar.update(1); continue
                strs = [str(int(x)) for x in ids]
                data[i][j][k] = optimize_mars_times(*strs, launch_dates, m_1, m_2, m_3, m_mars)
                pbar.update(1)
    pbar.close()

    os.makedirs('./optimal_asteroid_paths/mars_transfers', exist_ok=True)
    path = f'./optimal_asteroid_paths/mars_transfers/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(path, 'wb') as f: pickle.dump(data, f)
    return data


# =============================================================================
# COMPOSITION CLASSIFICATION
# =============================================================================

def classify_composition(taxonomy_str):
    """Map a taxonomy string to a composition class: 'C', 'S', or 'X/M'.

    Parameters
    ----------
    taxonomy_str : str — e.g. "Ch — Primitive carbonaceous [SMASSII]"

    Returns
    -------
    str — one of 'C', 'S', 'X/M', or 'Unknown'
    """
    if not taxonomy_str or taxonomy_str == 'Unclassified':
        return 'Unknown'
    # Get first letter of the taxonomy code (before the dash)
    tax = taxonomy_str.split('—')[0].strip().split()[0] if '—' in str(taxonomy_str) else str(taxonomy_str).split()[0]
    first = tax[0].upper()
    if first in ('C', 'B', 'F', 'G', 'D', 'T', 'P'):
        return 'C'
    elif first in ('S', 'Q', 'A', 'V', 'R', 'L', 'K'):
        return 'S'
    elif first in ('M', 'X', 'E'):
        return 'X/M'
    return 'Unknown'


def load_composition_map(tradeoff_csv='asteroid_tradeoff.csv'):
    """Load the tradeoff CSV and build a name -> composition class dict.

    Returns dict mapping UPPERCASE asteroid name -> 'C'/'S'/'X/M'/'Unknown'.
    Names are stripped of leading numbers (e.g. "24 Themis" -> "THEMIS").
    """
    import pandas as pd
    df = pd.read_csv(tradeoff_csv)
    comp_map = {}
    for _, row in df.iterrows():
        # Extract plain name: "24 Themis  (r=99.0km)" -> "THEMIS"
        raw = str(row['Name_DecRadius']).split('(')[0].strip()
        parts = raw.split()
        if parts and parts[0].replace('.', '').isdigit():
            name = ' '.join(parts[1:]).upper()
        else:
            name = raw.upper()
        comp_map[name] = classify_composition(row['Class_Composition_SMASSII'])
    return comp_map


def _triplet_has_diverse_composition(i, j, k, asteroid_list, comp_map, required=None):
    """Check if a triplet has the required composition diversity.

    Parameters
    ----------
    required : set or None — required composition classes, e.g. {'C', 'S', 'X/M'}.
        If None, requires all three asteroids to have DIFFERENT composition classes.
    """
    classes = []
    for idx in [i, j, k]:
        name = asteroid_list[idx]['NAME'].upper()
        cls = comp_map.get(name, 'Unknown')
        classes.append(cls)

    if required is not None:
        return required.issubset(set(classes))
    else:
        # All three must be different
        return len(set(classes)) == 3 and 'Unknown' not in classes


# =============================================================================
# TWO-LEVEL OPTIMIZATION (coarse filter + fine optimization)
# =============================================================================

def two_level_optimize(asteroid_list, m_1, m_2, m_3,
                       launch_utc_min, launch_utc_max,
                       top_n=50, science_scores=None, alpha=1.0,
                       comp_map=None, required_compositions=None,
                       allowed_architectures=None):
    """Coarse-filter all N^3 triplets, then fine-optimize top candidates.

    Parameters
    ----------
    science_scores : dict or None — asteroid NAME -> score (0-10).
    alpha : float — 1.0 = pure delta-v, 0.7 = 70% dv + 30% science.
    comp_map : dict or None — asteroid NAME -> composition class ('C','S','X/M').
        Load with load_composition_map(). If provided, only triplets with the
        required composition diversity are considered.
    required_compositions : set or None — e.g. {'C', 'S', 'X/M'}.
        If None but comp_map is provided, requires all three to be different classes.
    """
    num = len(asteroid_list)
    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_dates = [et_min, et_max]

    # Build task list with optional composition filter
    all_tasks = []
    for i in range(num):
        for j in range(num):
            for k in range(num):
                if len({asteroid_list[i]['ID'], asteroid_list[j]['ID'], asteroid_list[k]['ID']}) < 3:
                    continue
                if comp_map is not None:
                    if not _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                            comp_map, required_compositions):
                        continue
                all_tasks.append((i, j, k))

    n_total = num ** 3
    n_valid = len(all_tasks)
    filter_str = ""
    if comp_map is not None:
        req = required_compositions or {'C', 'S', 'X/M'}
        filter_str = f", filtered to {'+'.join(sorted(req))} compositions"
    print(f"Pass 1: Coarse evaluation ({n_valid} valid triplets out of {n_total}{filter_str})...")

    # Pass 1: coarse — try direct + Moon flyby + Mars flyby for each triplet
    coarse = []
    for i, j, k in tqdm(all_tasks, desc="Coarse"):
        strs = [str(int(asteroid_list[x]['ID'])) for x in [i, j, k]]

        # Try all architectures, keep best
        best_dv, best_arch = optimize_best_architecture(*strs, launch_dates,
                                                        m_1, m_2, m_3, quick=True,
                                                        allowed_architectures=allowed_architectures)
        coarse.append((i, j, k, best_dv, best_arch))

    # Score with optional science weighting
    if science_scores and alpha < 1.0:
        for idx, (i, j, k, dv, arch) in enumerate(coarse):
            sci = sum(science_scores.get(asteroid_list[x]['NAME'], 5.0) for x in [i, j, k])
            coarse[idx] = (i, j, k, alpha * dv + (1-alpha) * (30 - sci), arch)

    coarse.sort(key=lambda x: x[3])
    top = coarse[:top_n]

    # Pass 2: fine — full optimization with best architecture for each
    print(f"\nPass 2: Fine optimization on top {len(top)} candidates...")
    print(f"  Architectures in top {len(top)}: "
          + str({a: sum(1 for t in top if t[4] == a) for a in ['direct', 'moon', 'mars', 'earth']}))

    results = []
    for i, j, k, _, coarse_arch in tqdm(top, desc="Fine"):
        if comp_map is not None:
            # Keep a final guard so no non-diverse triplet can enter results.
            if not _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                    comp_map, required_compositions):
                continue
        strs = [str(int(asteroid_list[x]['ID'])) for x in [i, j, k]]
        result, best_arch = optimize_best_architecture(*strs, launch_dates,
                                                       m_1, m_2, m_3, quick=False,
                                                       allowed_architectures=allowed_architectures)
        result['architecture'] = best_arch
        results.append((i, j, k, result))
    feasible_count = sum(1 for _, _, _, r in results if _result_is_feasible(r))
    if feasible_count == 0:
        print("\nNo feasible solutions satisfy current constraints (including launch dv <= 7 km/s).")
    if science_scores and alpha < 1.0:
        def _objective_with_science(item):
            i, j, k, res = item
            names = [asteroid_list[i]['NAME'], asteroid_list[j]['NAME'], asteroid_list[k]['NAME']]
            sci = sum(science_scores.get(n, 5.0) for n in names)
            infeasible_pen = 1e6 if not _result_is_feasible(res) else 0.0
            return infeasible_pen + alpha * res['delta_v_total'] + (1 - alpha) * (30 - sci)
        results.sort(key=_objective_with_science)
    else:
        results.sort(key=lambda x: (0 if _result_is_feasible(x[3]) else 1,
                                    x[3]['delta_v_total']))

    print(f"\n{'='*90}\nTOP 10 PATHS (with gravity assist options)\n{'='*90}")
    for rank, (i, j, k, res) in enumerate(results[:10], 1):
        n1, n2, n3 = asteroid_list[i]['NAME'], asteroid_list[j]['NAME'], asteroid_list[k]['NAME']
        names = f"{n1} -> {n2} -> {n3}"
        dv = res['delta_v_total']
        ldv = np.linalg.norm(res['delta_v_launch']) if len(res['delta_v_launch']) > 0 else float('nan')
        arch = res.get('architecture', 'direct')
        arch_str = f"  [{arch.upper()}]" if arch != 'direct' else ""
        comp_str = ""
        if comp_map:
            c1 = comp_map.get(n1.upper(), '?')
            c2 = comp_map.get(n2.upper(), '?')
            c3 = comp_map.get(n3.upper(), '?')
            comp_str = f"  [{c1}+{c2}+{c3}]"
        ldv_txt = f"{ldv:.2f}" if np.isfinite(ldv) else "n/a"
        print(f"  #{rank}: {names}  |  dv={dv:.2f}  launch_dv={ldv_txt}{comp_str}{arch_str}")

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    with open(f'./optimal_asteroid_paths/two_level_{m_1}_{m_2}_{m_3}.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results


# =============================================================================
# BEAM SEARCH (structured multi-stage path selection)
# =============================================================================

def beam_search(asteroid_list, launch_utc_min, launch_utc_max,
                beam_width=10, m_1=0, m_2=0, m_3=0,
                science_scores=None, alpha=1.0):
    """Beam search: keep top-K candidates at each leg instead of committing to one.

    Stage 1: Earth -> A_i (all asteroids), keep top K.
    Stage 2: For K survivors, A_i -> A_j, keep top K.
    Stage 3: For K survivors, A_j -> A_k, keep top K.
    """
    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    num = len(asteroid_list)

    def _sci_score(names):
        if not science_scores or alpha >= 1.0:
            return 0
        return -(1-alpha) * sum(science_scores.get(n, 5.0) for n in names)

    def _leg_dv(body1_id, body2_id, et_depart, tof_bounds, m):
        """Optimize a single leg and return (best_dv, et_depart, et_arrive)."""
        bounds = [(tof_bounds[0]/YEAR, tof_bounds[1]/YEAR)]
        def obj(x):
            et_arr = et_depart + x[0] * YEAR
            if et_arr - et_min > MAX_MISSION_DURATION:
                return 1e3
            r1, v1 = get_state(body1_id, et_depart)
            r2, v2 = get_state(body2_id, et_arr)
            vl1, vl2, flag = solve_lambert(r1, r2, (et_arr-et_depart)/DAY, m, MU_SUN)
            if flag != 1:
                return 1e3
            return np.linalg.norm(vl1-v1) + np.linalg.norm(vl2-v2)

        res = differential_evolution(obj, bounds, maxiter=50, tol=0.1,
                                     seed=42, polish=True, popsize=5, updating='deferred')
        et_arrive = et_depart + res.x[0] * YEAR
        return res.fun, et_depart, et_arrive

    # Stage 1
    print("Beam Stage 1: Earth -> each asteroid...")
    stage1 = []
    for i in tqdm(range(num), desc="Stage 1"):
        # Try multiple launch dates across the window
        best_dv = np.inf
        best_ets = None
        for launch_frac in np.linspace(0, 1, 5):
            et_launch = et_min + launch_frac * (et_max - et_min)
            dv, _, et_arr = _leg_dv('399', str(int(asteroid_list[i]['ID'])),
                                     et_launch, (2*WEEK, 5*YEAR), m_1)
            if dv < best_dv:
                best_dv = dv
                best_ets = (et_launch, et_arr)
        stage1.append((i, best_dv, best_ets))

    stage1.sort(key=lambda x: alpha*x[1] + _sci_score([asteroid_list[x[0]]['NAME']]))
    stage1 = stage1[:beam_width]

    # Stage 2
    print(f"Beam Stage 2: {len(stage1)} survivors -> each asteroid...")
    stage2 = []
    for i, dv1, (et_launch, et_arr1) in stage1:
        for j in range(num):
            if j == i: continue
            # Stay at asteroid 1, then depart
            for stay_frac in [0.3, 0.6]:
                et_stay_end = et_arr1 + stay_frac * YEAR
                dv2, _, et_arr2 = _leg_dv(str(int(asteroid_list[i]['ID'])),
                                           str(int(asteroid_list[j]['ID'])),
                                           et_stay_end, (2*WEEK, 5*YEAR), m_2)
                total = dv1 + dv2
                stage2.append((i, j, total, et_launch, et_arr1, et_stay_end, et_arr2))

    stage2.sort(key=lambda x: alpha*x[2] + _sci_score(
        [asteroid_list[x[0]]['NAME'], asteroid_list[x[1]]['NAME']]))
    stage2 = stage2[:beam_width]

    # Stage 3
    print(f"Beam Stage 3: {len(stage2)} survivors -> each asteroid...")
    stage3 = []
    for i, j, dv12, et_launch, et_arr1, et_stay1, et_arr2 in stage2:
        for k in range(num):
            if k == i or k == j: continue
            for stay_frac in [0.3, 0.6]:
                et_stay_end2 = et_arr2 + stay_frac * YEAR
                if et_stay_end2 - et_launch > MAX_MISSION_DURATION:
                    continue
                dv3, _, et_arr3 = _leg_dv(str(int(asteroid_list[j]['ID'])),
                                           str(int(asteroid_list[k]['ID'])),
                                           et_stay_end2, (2*WEEK, 5*YEAR), m_3)
                total = dv12 + dv3
                names = [asteroid_list[x]['NAME'] for x in [i, j, k]]
                stage3.append((i, j, k, total, names,
                               et_launch, et_arr1, et_stay1, et_arr2, et_stay_end2, et_arr3))

    stage3.sort(key=lambda x: alpha*x[3] + _sci_score(x[4]))

    print(f"\n{'='*80}\nBEAM SEARCH RESULTS (K={beam_width})\n{'='*80}")
    for rank, (i, j, k, dv, names, *_) in enumerate(stage3[:10], 1):
        print(f"  #{rank}: {' -> '.join(names)}  |  total dv = {dv:.2f} km/s")

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    with open(f'./optimal_asteroid_paths/beam_K{beam_width}.pkl', 'wb') as f:
        pickle.dump(stage3, f)
    return stage3
