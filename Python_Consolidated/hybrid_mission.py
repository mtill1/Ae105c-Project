"""Hybrid mission scoring: evaluate CC / CE / EC / EE architectures per triplet.

For each top impulsive triplet, re-run legs L3 (A1→A2) and L4 (A2→A3) as
either Chemical (impulsive, Isp=320s) or Electric (low-thrust, Isp=3100s).
Chain Tsiolkovsky through all legs, return final delivered mass per arch.

Design: L1 (launch + GA) and L2 (GA → A1) are ALWAYS chemical — the launch
vehicle + periapsis burn need high thrust authority, and L2 is tied to the
GA window. So the trade is only in the two cruise legs L3 and L4.
"""

import numpy as np
from core import get_state, YEAR, DAY
from lowthrust import (optimize_lt_leg, tsiolkovsky_fraction,
                        ISP_CHEM, ISP_ELEC, G0,
                        DEFAULT_M_INIT_KG, DEFAULT_THRUST_N, DEFAULT_NSEG)


def lt_eligible(leg_dv_kms, leg_tof_years, m_kg, thrust_N=DEFAULT_THRUST_N):
    """Pair-closeness pre-screen — can LT even deliver this Δv in the given TOF?"""
    if leg_tof_years < 1.0:  # need at least 1 year for LT to be meaningful
        return False
    if leg_dv_kms > 6.0:  # too large for LT at typical thrust
        return False
    accel_kms2 = thrust_N / m_kg / 1e3
    dv_ceiling = accel_kms2 * leg_tof_years * YEAR
    return dv_ceiling > 1.2 * leg_dv_kms  # 20% margin for gravity losses


def score_arch_chemical_only(res, m_init):
    """All-chemical baseline (CC) — analytical via Tsiolkovsky chain."""
    dv_L1 = (np.linalg.norm(res['delta_v_launch']) +
             abs(float(res.get('delta_v_flyby', 0))))
    dv_L2 = np.linalg.norm(res['delta_v_A1_arrive'])
    dv_L3 = (np.linalg.norm(res['delta_v_A1_leave']) +
             np.linalg.norm(res['delta_v_A2_arrive']))
    dv_L4 = (np.linalg.norm(res['delta_v_A2_leave']) +
             np.linalg.norm(res['delta_v_A3_arrive']))

    m = m_init
    m *= tsiolkovsky_fraction(dv_L1, ISP_CHEM)
    m *= tsiolkovsky_fraction(dv_L2, ISP_CHEM)
    m *= tsiolkovsky_fraction(dv_L3, ISP_CHEM)
    m *= tsiolkovsky_fraction(dv_L4, ISP_CHEM)
    return m, {'dv_L1': dv_L1, 'dv_L2': dv_L2, 'dv_L3': dv_L3, 'dv_L4': dv_L4}


def _solve_leg_lt(res, leg_idx, m_entering, a_ids, thrust_N=DEFAULT_THRUST_N,
                   nseg=DEFAULT_NSEG, verbose=False):
    """Run LT solver on L3 or L4. Returns leg result dict + m_final."""
    if leg_idx == 3:
        et0, et1 = res['et_stay_1'], res['et_arrive_2']
        body0, body1 = a_ids[0], a_ids[1]
    elif leg_idx == 4:
        et0, et1 = res['et_stay_2'], res['et_arrive_3']
        body0, body1 = a_ids[1], a_ids[2]
    else:
        raise ValueError(f"LT only on legs 3 or 4, got {leg_idx}")

    r0, v0 = get_state(body0, et0)
    r1, v1 = get_state(body1, et1)
    tof_sec = et1 - et0

    return optimize_lt_leg(r0, v0, r1, v1, tof_sec,
                            m_init_kg=m_entering, thrust_N=thrust_N,
                            isp_s=ISP_ELEC, nseg=nseg, verbose=verbose)


def evaluate_hybrid(res, a_ids, m_init=DEFAULT_M_INIT_KG,
                    thrust_N=DEFAULT_THRUST_N, nseg=DEFAULT_NSEG,
                    verbose=False):
    """Evaluate all 4 architectures and return best-mass result.

    Returns dict with:
      all_archs: {arch_code: m_final, ...}
      best_arch: 'CC'|'CE'|'EC'|'EE'
      m_best: float (kg)
      m_baseline_CC: float
      improvement_kg: m_best - m_baseline_CC
      lt_leg_results: dict of leg_idx → solver result (if used)
    """
    # Baseline: per-leg chemical Δvs + TOFs
    dv_L1 = (np.linalg.norm(res['delta_v_launch']) +
             abs(float(res.get('delta_v_flyby', 0))))
    dv_L2 = np.linalg.norm(res['delta_v_A1_arrive'])
    dv_L3_chem = (np.linalg.norm(res['delta_v_A1_leave']) +
                   np.linalg.norm(res['delta_v_A2_arrive']))
    dv_L4_chem = (np.linalg.norm(res['delta_v_A2_leave']) +
                   np.linalg.norm(res['delta_v_A3_arrive']))

    tof_L3_yr = (res['et_arrive_2'] - res['et_stay_1']) / YEAR
    tof_L4_yr = (res['et_arrive_3'] - res['et_stay_2']) / YEAR

    # Chain L1 + L2 (always chemical)
    m_after_L1 = m_init     * tsiolkovsky_fraction(dv_L1, ISP_CHEM)
    m_after_L2 = m_after_L1 * tsiolkovsky_fraction(dv_L2, ISP_CHEM)

    archs = {}
    lt_results = {}

    # --- L3 evaluation ---
    m_L3C = m_after_L2 * tsiolkovsky_fraction(dv_L3_chem, ISP_CHEM)

    m_L3E = None
    if lt_eligible(dv_L3_chem, tof_L3_yr, m_after_L2, thrust_N):
        lt3 = _solve_leg_lt(res, 3, m_after_L2, a_ids, thrust_N, nseg, verbose)
        lt_results[3] = lt3
        if lt3['converged']:
            m_L3E = lt3['m_final']
        else:
            if verbose:
                print(f"    L3 LT did not converge: {lt3['reason']}")

    # --- L4 evaluation given each L3 outcome ---
    for arch_L3 in ['C', 'E']:
        if arch_L3 == 'C':
            m_entering_L4 = m_L3C
        else:
            if m_L3E is None:
                continue  # LT infeasible
            m_entering_L4 = m_L3E

        # Chemical L4
        m_final_C = m_entering_L4 * tsiolkovsky_fraction(dv_L4_chem, ISP_CHEM)
        archs[arch_L3 + 'C'] = m_final_C

        # Electric L4 (only if eligible)
        if lt_eligible(dv_L4_chem, tof_L4_yr, m_entering_L4, thrust_N):
            lt4 = _solve_leg_lt(res, 4, m_entering_L4, a_ids, thrust_N, nseg, verbose)
            lt_results[(arch_L3, 4)] = lt4
            if lt4['converged']:
                archs[arch_L3 + 'E'] = lt4['m_final']

    best_arch = max(archs, key=archs.get)
    return {
        'all_archs': archs,
        'best_arch': best_arch,
        'm_best_kg': archs[best_arch],
        'm_baseline_CC_kg': archs.get('CC', 0.0),
        'improvement_kg': archs[best_arch] - archs.get('CC', 0.0),
        'prop_fraction_CC': 1 - archs.get('CC', 0.0) / m_init,
        'prop_fraction_best': 1 - archs[best_arch] / m_init,
        'm_init_kg': m_init, 'thrust_N': thrust_N, 'isp_elec_s': ISP_ELEC,
        'dv_L3_chem': dv_L3_chem, 'dv_L4_chem': dv_L4_chem,
        'tof_L3_yr': tof_L3_yr, 'tof_L4_yr': tof_L4_yr,
        'lt_leg_results': lt_results,
    }
