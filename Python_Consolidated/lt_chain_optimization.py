"""LT-after-launch chain optimization.

Mission shape:  Earth → (optional one flyby) → A1 → A2 → A3

Constraints (frozen by `LTChainConfig`):
  • Launch:  impulsive, ≤ 7 km/s, EXCLUDED from objective (launcher's job).
  • Post-launch propulsion: electric-only Sims-Flanagan (Isp 3100 s, 0.30 N cap).
  • Mars/Earth flyby: ballistic only (|v_inf_in| ≈ |v_inf_out|, gravity-only turn).
  • Mission cap: 30 years.
  • Stays: ≥ 3 months at each asteroid.

Objective: minimize total POST-LAUNCH Δv (sum of LT integrated dvs across legs).

Speed: outer DE uses a surrogate (gravity-loss factor) for LT dvs so a single
evaluation is ms-fast. Final winner is re-verified with the real Sims-Flanagan
solver in `verify_lt_chain_full`, which produces the throttle profile per leg
(15 segments × 3 components) for output / plotting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spiceypy
from scipy.optimize import differential_evolution

from core import (audit_flyby_geometry, get_state, solve_lambert,
                   propagate_two_body, MU_SUN, DAY, WEEK, MONTH, YEAR,
                   BALLISTIC_VINF_TOLERANCE_KMS)
from optimization import FLYBY_BODIES
from lowthrust import (optimize_lt_leg, tsiolkovsky_fraction,
                        ISP_ELEC, G0)
from mass_optimization import gravity_loss_factor


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LTChainConfig:
    # Spacecraft
    spacecraft_launch_mass_kg: float = 3000.0   # launcher delivers this much
    lt_chain_initial_mass_kg:  float = 1500.0   # mass at start of LT phase
    isp_elec_s:    float = 3100.0
    thrust_N:      float = 0.30

    # Hard physical constraints
    launch_dv_max_kms: float = 7.0    # impulsive launch ≤ 7 km/s
    mission_max_yr:    float = 30.0
    stay_min_months:   float = 3.0
    lt_leg_min_yr:     float = 2.0    # minimum cruise TOF per LT leg (prevents engine saturation)

    # LT solver settings (per leg)
    lt_nseg:        int   = 15

    # DE
    de_maxiter:    int  = 300
    de_popsize:    int  = 18
    de_seeds:      Tuple[int, ...] = (42, 137, 314, 808)
    de_m_revs:     Tuple[Tuple[int, ...], ...] = (
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0),
    )


DEFAULT_CONFIG = LTChainConfig()


# ============================================================================
# Decision-vector unpacking
# ============================================================================

def _bounds_with_flyby(launch_range_et: List[float],
                        flyby_name: str,
                        cfg: LTChainConfig) -> List[Tuple[float, float]]:
    """Returns list of (lo, hi) in YEARS for the 7-D timing vector.

    Bounds chosen so a typical mid-of-bounds sample respects the mission cap.
    Per-leg upper TOF capped at 8 years to give LT cruise enough time.

    Order: [launch_offset, tof_E2FB, tof_FB2A1, stay_A1, tof_A1A2, stay_A2, tof_A2A3]
    """
    fb = FLYBY_BODIES[flyby_name]
    launch_window_yr = (launch_range_et[1] - launch_range_et[0]) / YEAR
    leg_upper = min(8.0, max(3.0, cfg.mission_max_yr / 4))
    stay_upper = 1.0  # 1-year max stay
    leg_min = cfg.lt_leg_min_yr
    return [
        (0.0,                 launch_window_yr),
        (fb['tof_min']/YEAR,  fb['tof_max']/YEAR),
        (leg_min,             leg_upper),
        (cfg.stay_min_months / 12, stay_upper),
        (leg_min,             leg_upper),
        (cfg.stay_min_months / 12, stay_upper),
        (leg_min,             leg_upper),
    ]


def _bounds_no_flyby(launch_range_et: List[float],
                      cfg: LTChainConfig) -> List[Tuple[float, float]]:
    """6-D for direct (no flyby): same shape minus the E→FB and FB→A1 split."""
    launch_window_yr = (launch_range_et[1] - launch_range_et[0]) / YEAR
    leg_upper = min(8.0, max(3.0, cfg.mission_max_yr / 4))
    stay_upper = 1.0
    leg_min = cfg.lt_leg_min_yr
    return [
        (0.0,                 launch_window_yr),
        (leg_min,             leg_upper),
        (cfg.stay_min_months / 12, stay_upper),
        (leg_min,             leg_upper),
        (cfg.stay_min_months / 12, stay_upper),
        (leg_min,             leg_upper),
    ]


def _unpack_with_flyby(input_vec, launch_range_et):
    s = YEAR * np.asarray(input_vec)
    et_launch = s[0] + launch_range_et[0]
    et_fb     = et_launch + s[1]
    et_a1     = et_fb     + s[2]
    et_s1     = et_a1     + s[3]
    et_a2     = et_s1     + s[4]
    et_s2     = et_a2     + s[5]
    et_a3     = et_s2     + s[6]
    return et_launch, et_fb, et_a1, et_s1, et_a2, et_s2, et_a3


def _unpack_no_flyby(input_vec, launch_range_et):
    s = YEAR * np.asarray(input_vec)
    et_launch = s[0] + launch_range_et[0]
    et_a1     = et_launch + s[1]
    et_s1     = et_a1     + s[2]
    et_a2     = et_s1     + s[3]
    et_s2     = et_a2     + s[4]
    et_a3     = et_s2     + s[5]
    return et_launch, et_a1, et_s1, et_a2, et_s2, et_a3


# ============================================================================
# Surrogate path computation (used inside DE)
# ============================================================================

def _lambert_dv_pair(r0, v0_body, r1, v1_body, et0, et1, m_revs):
    """Returns (V1, V2, dv_dep_kms, dv_arr_kms, ef)."""
    V1, V2, ef = solve_lambert(r0, r1, (et1 - et0) / DAY, m_revs, MU_SUN)
    if ef != 1:
        return None, None, np.inf, np.inf, ef
    return (V1, V2,
             float(np.linalg.norm(V1 - v0_body)),
             float(np.linalg.norm(v1_body - V2)),
             ef)


def compute_path_lt_chain_surrogate(
        a1_id: str, a2_id: str, a3_id: str,
        et_launch: float,
        et_a1_arr: float, et_a1_dep: float,
        et_a2_arr: float, et_a2_dep: float,
        et_a3_arr: float,
        flyby_name: Optional[str],
        et_flyby: Optional[float],
        m_revs: Tuple[int, ...],
        cfg: LTChainConfig = DEFAULT_CONFIG,
        ) -> Dict[str, Any]:
    """Fast surrogate for DE. Lambert solves + gravity-loss factor for LT legs.

    Leg structure with flyby:
      L0: Earth → flyby (impulsive launch dv at start, ballistic afterward)
      L1: flyby → A1 (LT)
      L2: A1 → A2 (LT)
      L3: A2 → A3 (LT)

    Without flyby:
      L0: Earth → A1 (impulsive launch dv at start, LT afterward — modeled as
          one LT leg starting at the body at v_helio = v_Earth + v_inf_launch)
      L1: A1 → A2 (LT)
      L2: A2 → A3 (LT)
    """
    # ---- States ----
    earth_r,  earth_v  = get_state('399', et_launch)
    a1_arr_r, a1_arr_v = get_state(a1_id,  et_a1_arr)
    a1_lv_r,  a1_lv_v  = get_state(a1_id,  et_a1_dep)
    a2_arr_r, a2_arr_v = get_state(a2_id,  et_a2_arr)
    a2_lv_r,  a2_lv_v  = get_state(a2_id,  et_a2_dep)
    a3_arr_r, a3_arr_v = get_state(a3_id,  et_a3_arr)

    fb_r = fb_v = None
    if flyby_name and et_flyby is not None:
        fb = FLYBY_BODIES[flyby_name]
        fb_r, fb_v = get_state(fb['id'], et_flyby)

    # ---- L0: Earth → flyby (or → A1 if direct), impulsive at Earth ----
    if flyby_name:
        V1_e, V2_fb, dvL_dep, _, ef0 = _lambert_dv_pair(
            earth_r, earth_v, fb_r, fb_v, et_launch, et_flyby, m_revs[0])
    else:
        V1_e, V2_fb, dvL_dep, _, ef0 = _lambert_dv_pair(
            earth_r, earth_v, a1_arr_r, a1_arr_v, et_launch, et_a1_arr, m_revs[0])
    if ef0 != 1:
        return {'feasible': False, 'reason': 'lambert_E_fb_fail',
                'post_launch_dv_kms': 1e3}

    launch_dv = dvL_dep
    if launch_dv > cfg.launch_dv_max_kms + 1e-6:
        return {'feasible': False, 'reason': f'launch_dv {launch_dv:.2f} > {cfg.launch_dv_max_kms}',
                'post_launch_dv_kms': 1e3,
                'launch_dv_kms': launch_dv}

    # ---- L1 onward (LT after flyby or direct) ----
    legs_lt: List[Dict[str, Any]] = []
    m_now = cfg.lt_chain_initial_mass_kg

    if flyby_name:
        # Mars/Earth flyby: ballistic check + LT leg flyby→A1
        # Lambert flyby → A1
        V1_fb_dep, V2_a1, dv_dep_fb, dv_arr_a1, ef1 = _lambert_dv_pair(
            fb_r, fb_v, a1_arr_r, a1_arr_v, et_flyby, et_a1_arr, m_revs[1])
        if ef1 != 1:
            return {'feasible': False, 'reason': 'lambert_fb_a1_fail',
                    'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}

        # Ballistic flyby check: |v_inf_in| ≈ |v_inf_out| AND turn ≤ max
        v_inf_in  = V2_fb     - fb_v
        v_inf_out = V1_fb_dep - fb_v
        if abs(np.linalg.norm(v_inf_in) - np.linalg.norm(v_inf_out)) > BALLISTIC_VINF_TOLERANCE_KMS:
            return {'feasible': False, 'reason': 'flyby_powered',
                    'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}

        # Geometric turn feasibility
        from optimization import FLYBY_BODIES as FB
        from core import get_mu, get_radius
        fb = FB[flyby_name]
        mu = get_mu(fb['mu_body']); R = get_radius(fb['radii_body'])
        safe_r = R + fb['min_alt']
        a_, b_ = np.linalg.norm(v_inf_in), np.linalg.norm(v_inf_out)
        cosd = np.clip(np.dot(v_inf_in, v_inf_out) / (a_ * b_), -1, 1)
        delta = np.arccos(cosd)
        s_in  = min(1.0, 1.0 / (1.0 + safe_r * a_**2 / mu))
        s_out = min(1.0, 1.0 / (1.0 + safe_r * b_**2 / mu))
        delta_max = np.arcsin(s_in) + np.arcsin(s_out)
        if delta > delta_max + 1e-6:
            return {'feasible': False, 'reason': 'flyby_geometry',
                    'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}

        # LT surrogate for flyby → A1
        tof_l1 = et_a1_arr - et_flyby
        lambert_dv_l1 = dv_dep_fb + dv_arr_a1
        factor = gravity_loss_factor(lambert_dv_l1, tof_l1, m_now, cfg.thrust_N)
        if not np.isfinite(factor):
            return {'feasible': False, 'reason': 'lt_l1_thrust_ceiling',
                    'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
        dv_lt_l1 = lambert_dv_l1 * factor
        m_after_l1 = m_now * tsiolkovsky_fraction(dv_lt_l1, cfg.isp_elec_s)
        legs_lt.append({'label': f'{flyby_name}→A1',
                         'tof_yr': tof_l1 / YEAR,
                         'lambert_dv_kms': lambert_dv_l1,
                         'dv_lt_kms': dv_lt_l1,
                         'm_in_kg': m_now, 'm_out_kg': m_after_l1,
                         'm_revs': m_revs[1]})
        m_now = m_after_l1
    else:
        # Direct: L0 itself is the LT leg Earth → A1, but launch is impulsive
        # at Earth side. So we already have launch_dv. Treat the LT leg as
        # going from (Earth pos at launch, V1_e) → (A1 pos at arrival, A1 v).
        # Lambert dv_arr_a1 is the velocity mismatch at A1.
        _, _, _, dv_arr_a1, _ = _lambert_dv_pair(
            earth_r, earth_v, a1_arr_r, a1_arr_v,
            et_launch, et_a1_arr, m_revs[0])
        # Wait — we already have V1_e, V2_fb (=V2_a1). dv_arr_a1 = ||V2 - a1_v||
        dv_arr_a1 = float(np.linalg.norm(V2_fb - a1_arr_v))
        tof_l1 = et_a1_arr - et_launch
        # Lambert dv on the leg = arrival mismatch (the launch handled departure)
        lambert_dv_l1 = dv_arr_a1
        factor = gravity_loss_factor(lambert_dv_l1, tof_l1, m_now, cfg.thrust_N)
        if not np.isfinite(factor):
            return {'feasible': False, 'reason': 'lt_l0_thrust_ceiling',
                    'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
        dv_lt_l1 = lambert_dv_l1 * factor
        m_after_l1 = m_now * tsiolkovsky_fraction(dv_lt_l1, cfg.isp_elec_s)
        legs_lt.append({'label': 'Earth→A1',
                         'tof_yr': tof_l1 / YEAR,
                         'lambert_dv_kms': lambert_dv_l1,
                         'dv_lt_kms': dv_lt_l1,
                         'm_in_kg': m_now, 'm_out_kg': m_after_l1,
                         'm_revs': m_revs[0]})
        m_now = m_after_l1

    # ---- L2: A1 → A2 (LT) ----
    leg_idx = 2 if flyby_name else 1
    _, _, dv_dep_a1, dv_arr_a2, ef2 = _lambert_dv_pair(
        a1_lv_r, a1_lv_v, a2_arr_r, a2_arr_v,
        et_a1_dep, et_a2_arr, m_revs[leg_idx])
    if ef2 != 1:
        return {'feasible': False, 'reason': 'lambert_a1_a2_fail',
                'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
    tof_l2 = et_a2_arr - et_a1_dep
    lambert_dv_l2 = dv_dep_a1 + dv_arr_a2
    factor = gravity_loss_factor(lambert_dv_l2, tof_l2, m_now, cfg.thrust_N)
    if not np.isfinite(factor):
        return {'feasible': False, 'reason': 'lt_l2_thrust_ceiling',
                'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
    dv_lt_l2 = lambert_dv_l2 * factor
    m_after_l2 = m_now * tsiolkovsky_fraction(dv_lt_l2, cfg.isp_elec_s)
    legs_lt.append({'label': 'A1→A2', 'tof_yr': tof_l2 / YEAR,
                     'lambert_dv_kms': lambert_dv_l2, 'dv_lt_kms': dv_lt_l2,
                     'm_in_kg': m_now, 'm_out_kg': m_after_l2,
                     'm_revs': m_revs[leg_idx]})
    m_now = m_after_l2

    # ---- L3: A2 → A3 (LT) ----
    leg_idx = 3 if flyby_name else 2
    _, _, dv_dep_a2, dv_arr_a3, ef3 = _lambert_dv_pair(
        a2_lv_r, a2_lv_v, a3_arr_r, a3_arr_v,
        et_a2_dep, et_a3_arr, m_revs[leg_idx])
    if ef3 != 1:
        return {'feasible': False, 'reason': 'lambert_a2_a3_fail',
                'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
    tof_l3 = et_a3_arr - et_a2_dep
    lambert_dv_l3 = dv_dep_a2 + dv_arr_a3
    factor = gravity_loss_factor(lambert_dv_l3, tof_l3, m_now, cfg.thrust_N)
    if not np.isfinite(factor):
        return {'feasible': False, 'reason': 'lt_l3_thrust_ceiling',
                'post_launch_dv_kms': 1e3, 'launch_dv_kms': launch_dv}
    dv_lt_l3 = lambert_dv_l3 * factor
    m_after_l3 = m_now * tsiolkovsky_fraction(dv_lt_l3, cfg.isp_elec_s)
    legs_lt.append({'label': 'A2→A3', 'tof_yr': tof_l3 / YEAR,
                     'lambert_dv_kms': lambert_dv_l3, 'dv_lt_kms': dv_lt_l3,
                     'm_in_kg': m_now, 'm_out_kg': m_after_l3,
                     'm_revs': m_revs[leg_idx]})

    post_launch_dv = sum(L['dv_lt_kms'] for L in legs_lt)
    return {'feasible': True,
             'launch_dv_kms':       launch_dv,
             'post_launch_dv_kms':  post_launch_dv,
             'm_final_kg':          m_after_l3,
             'legs_lt':             legs_lt,
             'flyby_name':          flyby_name,
             'm_revs':              m_revs,
             'epochs': {'et_launch': et_launch, 'et_flyby': et_flyby,
                         'et_a1_arr': et_a1_arr, 'et_a1_dep': et_a1_dep,
                         'et_a2_arr': et_a2_arr, 'et_a2_dep': et_a2_dep,
                         'et_a3_arr': et_a3_arr}}


# ============================================================================
# DE objective wrapper
# ============================================================================

def score_paths_lt_chain(input_vec, a1, a2, a3, launch_range_et,
                          flyby_name: Optional[str],
                          m_revs: Tuple[int, ...],
                          cfg: LTChainConfig = DEFAULT_CONFIG):
    """DE objective with SOFT penalties so DE gets a gradient on infeasible
    points. Returns post_launch_dv if feasible, else 1000 + violation_score.
    """
    try:
        if flyby_name:
            ets = _unpack_with_flyby(input_vec, launch_range_et)
            et_launch, et_fb, et_a1, et_s1, et_a2, et_s2, et_a3 = ets
        else:
            ets = _unpack_no_flyby(input_vec, launch_range_et)
            et_launch, et_a1, et_s1, et_a2, et_s2, et_a3 = ets
            et_fb = None

        # Soft mission-duration penalty
        mission_yr = (et_a3 - et_launch) / YEAR
        mission_violation = max(0.0, mission_yr - cfg.mission_max_yr)

        out = compute_path_lt_chain_surrogate(
            a1, a2, a3, et_launch, et_a1, et_s1, et_a2, et_s2, et_a3,
            flyby_name, et_fb, m_revs, cfg)

        if out['feasible']:
            base = out['post_launch_dv_kms']
            # Even if surrogate says feasible, a small mission-cap violation
            # should add penalty (in case the cap was checked outside).
            return base + 100 * mission_violation

        # Infeasible: build a graduated penalty so DE has gradient
        reason = out.get('reason', '?')
        if 'launch_dv' in reason:
            launch_dv = out.get('launch_dv_kms', 1e3)
            return 1e3 + (launch_dv - cfg.launch_dv_max_kms) * 100
        if reason == 'flyby_powered':
            # Encourage closer to ballistic — score includes how far from match
            return 1e3 + 200 + 50 * mission_violation
        if reason == 'flyby_geometry':
            return 1e3 + 300 + 50 * mission_violation
        if 'thrust_ceiling' in reason:
            # LT thrust limit hit — encourage longer cruise legs
            return 1e3 + 400 + 100 * mission_violation
        if 'lambert' in reason:
            # Lambert non-convergence — quietly hard
            return 1e3 + 500
        return 1e3 + 100 * mission_violation
    except Exception:
        return 1e4


# ============================================================================
# Optimizer wrapper
# ============================================================================

def optimize_lt_chain_triplet(a1_id, a2_id, a3_id,
                                launch_range_et: List[float],
                                flyby_name: Optional[str] = 'mars',
                                cfg: LTChainConfig = DEFAULT_CONFIG,
                                verbose: bool = False) -> Dict[str, Any]:
    """Run DE for one triplet across all m-revs combos and seeds."""
    if flyby_name:
        bounds = _bounds_with_flyby(launch_range_et, flyby_name, cfg)
        n_legs = 4
    else:
        bounds = _bounds_no_flyby(launch_range_et, cfg)
        n_legs = 3

    # Filter m-revs combos to length n_legs
    m_revs_options = [m[:n_legs] for m in cfg.de_m_revs]

    best = None
    for m_revs in m_revs_options:
        for seed in cfg.de_seeds:
            try:
                res = differential_evolution(
                    lambda x: score_paths_lt_chain(
                        x, a1_id, a2_id, a3_id, launch_range_et,
                        flyby_name, m_revs, cfg),
                    bounds, maxiter=cfg.de_maxiter, tol=1e-7, seed=seed,
                    polish=True, popsize=cfg.de_popsize,
                    mutation=(0.5, 1.3), recombination=0.8,
                    updating='deferred')
            except Exception as e:
                if verbose:
                    print(f'    DE m={m_revs} seed={seed} ERROR: {e}')
                continue

            try:
                if flyby_name:
                    ets = _unpack_with_flyby(res.x, launch_range_et)
                    et_launch, et_fb, et_a1, et_s1, et_a2, et_s2, et_a3 = ets
                else:
                    ets = _unpack_no_flyby(res.x, launch_range_et)
                    et_launch, et_a1, et_s1, et_a2, et_s2, et_a3 = ets
                    et_fb = None

                full = compute_path_lt_chain_surrogate(
                    a1_id, a2_id, a3_id, et_launch, et_a1, et_s1, et_a2, et_s2, et_a3,
                    flyby_name, et_fb, m_revs, cfg)
            except Exception:
                continue
            if not full['feasible']:
                continue
            score = full['post_launch_dv_kms']
            if verbose:
                print(f'    m={m_revs} seed={seed:3d}  post-launch dv = '
                      f'{score:.3f}  launch dv = {full["launch_dv_kms"]:.2f}')
            if best is None or score < best['post_launch_dv_kms']:
                full['_seed'] = seed
                best = full
    return best


# ============================================================================
# Verification with full Sims-Flanagan LT solver
# ============================================================================

def verify_lt_chain_full(result: Dict[str, Any],
                          a1_id: str, a2_id: str, a3_id: str,
                          cfg: LTChainConfig = DEFAULT_CONFIG,
                          verbose: bool = False) -> Dict[str, Any]:
    """Re-run each LT leg with full Sims-Flanagan solver. Captures throttles."""
    flyby_name = result['flyby_name']
    eps = result['epochs']
    et_launch = eps['et_launch']
    et_flyby  = eps.get('et_flyby')
    et_a1_arr = eps['et_a1_arr']; et_a1_dep = eps['et_a1_dep']
    et_a2_arr = eps['et_a2_arr']; et_a2_dep = eps['et_a2_dep']
    et_a3_arr = eps['et_a3_arr']
    m_revs = result['m_revs']

    out: Dict[str, Any] = {'flyby_name': flyby_name, 'm_revs': list(m_revs),
                            'launch_dv_kms': result['launch_dv_kms'],
                            'epochs': dict(eps),
                            'verified_legs': [],
                            'feasibility': {}}

    # Capture flyby diagnostics if applicable. Pass m_revs so the audit uses
    # the SAME Lambert branches the surrogate optimized over — otherwise it
    # audits a different (m=0) trajectory than the one actually being flown.
    if flyby_name and et_flyby:
        audit = audit_flyby_geometry(et_launch, et_flyby, et_a1_arr,
                                       a1_id, flyby_name,
                                       m_e_to_fb=m_revs[0],
                                       m_fb_to_a1=m_revs[1])
        out['flyby_audit'] = {
            'feasible':              audit['feasible'],
            'geometric_ok':          audit.get('geometric_ok'),
            'ballistic_ok':          audit.get('ballistic_ok'),
            'v_inf_in_kms':          audit['v_inf_in_kms'],
            'v_inf_out_kms':         audit['v_inf_out_kms'],
            'v_inf_in_vec':          audit['v_inf_in_vec'],
            'v_inf_out_vec':         audit['v_inf_out_vec'],
            'turn_angle_deg':        audit['turn_angle_deg'],
            'turn_max_deg':          audit['turn_max_deg'],
            'periapsis_alt_km':      audit['periapsis_alt_km'],
            'safe_periapsis_alt_km': audit['safe_periapsis_alt_km'],
            'energy_residual_kms':   audit['energy_residual_kms'],
        }

    # Re-run each LT leg with the full solver. We need start/end (r, v) for
    # each LT leg.
    m_now = cfg.lt_chain_initial_mass_kg
    legs_in: List[Tuple[str, str, float, str, float, int]] = []
    if flyby_name:
        # For flyby case: LT leg flyby→A1 starts from Mars exit state.
        # We use the Lambert solution (from the surrogate path) to define the
        # initial v_helio on the leg. (Energy-conservation already passed.)
        legs_in.append((f'{flyby_name}→A1', FLYBY_BODIES[flyby_name]['id'],
                         et_flyby, a1_id, et_a1_arr, m_revs[1]))
        legs_in.append(('A1→A2', a1_id, et_a1_dep, a2_id, et_a2_arr, m_revs[2]))
        legs_in.append(('A2→A3', a2_id, et_a2_dep, a3_id, et_a3_arr, m_revs[3]))
    else:
        legs_in.append(('Earth→A1', '399', et_launch, a1_id, et_a1_arr, m_revs[0]))
        legs_in.append(('A1→A2', a1_id, et_a1_dep, a2_id, et_a2_arr, m_revs[1]))
        legs_in.append(('A2→A3', a2_id, et_a2_dep, a3_id, et_a3_arr, m_revs[2]))

    feasible_all = True

    for label, b0, et0, b1, et1, mrev in legs_in:
        r0, v_body0 = get_state(b0, et0)
        r1, v_body1 = get_state(b1, et1)
        # Initial v of the spacecraft at start of leg
        if 'Earth→' in label:
            # Direct case: spacecraft has launch v_inf at Earth, v_helio = V1_lambert
            V1, V2, ef = solve_lambert(r0, r1, (et1-et0)/DAY, mrev, MU_SUN)
            v0_sc = V1
        elif label.startswith(flyby_name + '→') if flyby_name else False:
            # Post-flyby: v_helio = v_body + v_inf_out (from Lambert flyby→A1)
            V1, V2, ef = solve_lambert(r0, r1, (et1-et0)/DAY, mrev, MU_SUN)
            v0_sc = V1   # equals v_body + v_inf_out
        else:
            # A1→A2 or A2→A3: spacecraft is co-orbiting body0 (v0 = v_body0)
            v0_sc = v_body0

        tof = et1 - et0
        lt_res = optimize_lt_leg(r0, v0_sc, r1, v_body1, tof,
                                   m_init_kg=m_now, thrust_N=cfg.thrust_N,
                                   isp_s=cfg.isp_elec_s, nseg=cfg.lt_nseg,
                                   verbose=verbose)

        nseg = lt_res['nseg']
        u_arr = np.asarray(lt_res['throttles']).reshape(nseg, 3)
        thrust_per_seg_N = np.linalg.norm(u_arr, axis=1) * cfg.thrust_N

        leg_data = {
            'label':          label,
            'et_start':       et0,
            'et_end':          et1,
            'tof_yr':         tof / YEAR,
            'm_revs':         int(mrev),
            'm_in_kg':        m_now,
            'converged':      bool(lt_res['converged']),
            'reason':         lt_res.get('reason', ''),
            'pos_err_km':     lt_res['pos_err_km'],
            'vel_err_kms':    lt_res['vel_err_kms'],
            'dv_integral_kms': lt_res['dv_integral_kms'],
            'm_out_kg':       float(lt_res['m_final']),
            'thrust_profile': {
                'time_yr_from_leg_start': [
                    (k + 0.5) * (tof / YEAR) / nseg for k in range(nseg)],
                'segment_dt_yr':  (tof / YEAR) / nseg,
                'throttle_unit_vector': u_arr.tolist(),         # |u|≤1 nominally
                'thrust_magnitude_N':   thrust_per_seg_N.tolist(),
                'thrust_max_N':         cfg.thrust_N,
            },
        }
        out['verified_legs'].append(leg_data)

        if not lt_res['converged']:
            feasible_all = False
        m_now = float(lt_res['m_final'])

    out['post_launch_dv_kms_full'] = sum(
        L['dv_integral_kms'] for L in out['verified_legs'])
    out['m_final_kg_full'] = m_now
    out['feasibility']['all_legs_converged'] = feasible_all
    return out


# ============================================================================
# Public convenience runner
# ============================================================================

def run_triplet(triplet_names: Tuple[str, str, str],
                 asteroid_list,
                 launch_range_et: List[float],
                 flyby_name: Optional[str] = 'mars',
                 cfg: LTChainConfig = DEFAULT_CONFIG,
                 verify: bool = True,
                 verbose: bool = False) -> Dict[str, Any]:
    """End-to-end: optimize + verify. Returns full result dict."""
    from core import get_id_from_asteroid_name
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n)))
              for n in triplet_names]

    if verbose:
        print(f'Triplet: {" → ".join(triplet_names)}, flyby={flyby_name}')
        print(f'Launch dv ≤ {cfg.launch_dv_max_kms} km/s, '
              f'mission ≤ {cfg.mission_max_yr} yr')

    surrogate = optimize_lt_chain_triplet(
        *a_ids, launch_range_et, flyby_name=flyby_name, cfg=cfg, verbose=verbose)

    if surrogate is None:
        return {'feasible': False, 'reason': 'no_de_convergence'}

    result = {'triplet': list(triplet_names),
               'asteroid_ids': a_ids,
               'flyby_name':    flyby_name,
               'config': {
                   'launch_dv_max_kms':         cfg.launch_dv_max_kms,
                   'mission_max_yr':            cfg.mission_max_yr,
                   'spacecraft_launch_mass_kg': cfg.spacecraft_launch_mass_kg,
                   'lt_chain_initial_mass_kg':  cfg.lt_chain_initial_mass_kg,
                   'isp_elec_s':                cfg.isp_elec_s,
                   'thrust_N':                  cfg.thrust_N,
               },
               'surrogate': surrogate}

    if verify:
        if verbose: print('Running full Sims-Flanagan verification...')
        verified = verify_lt_chain_full(surrogate, *a_ids, cfg=cfg, verbose=verbose)
        result['verified'] = verified
    return result
