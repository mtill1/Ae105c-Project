"""Joint timing + propulsion-architecture optimization to maximize delivered mass.

The existing pipeline picks chemical-optimal timings, then layers low-thrust on
top — which forces the LT solver into a geometry it didn't get to shape. This
module re-optimizes timing PER architecture using a fast surrogate for electric
legs (Lambert dv * gravity-loss factor + thrust-feasibility check), with the
objective being final delivered mass (Tsiolkovsky chain through 4 legs).

The heavy LT solver (`lowthrust.optimize_lt_leg`) is reserved for verification
of the top candidates per triplet — see `verify_with_full_lt`.

Architecture codes are 3-letter strings labeling propulsion mode of the 3
inter-body transfer legs (L2 = flyby->A1, L3 = A1->A2, L4 = A2->A3).
L1 (Earth launch + powered flyby periapsis) is always chemical.

Letters: 'C' = chemical (Isp 320s), 'E' = electric (Isp 3100s).
"""

import numpy as np
from scipy.optimize import differential_evolution

from core import (solve_lambert, get_state, get_mu, get_radius,
                  compute_flyby_dv, MU_SUN, DAY, WEEK, MONTH, YEAR,
                  MAX_MISSION_DURATION)
from optimization import FLYBY_BODIES, _unpack_flyby_input
from lowthrust import (tsiolkovsky_fraction, ISP_CHEM, ISP_ELEC, G0,
                        DEFAULT_M_INIT_KG, DEFAULT_THRUST_N, DEFAULT_NSEG,
                        optimize_lt_leg)


# =============================================================================
# SURROGATE: gravity-loss factor for low-thrust legs
# =============================================================================

def gravity_loss_factor(lambert_dv_kms, tof_sec, m_kg, thrust_N):
    """Surrogate multiplier on impulsive Lambert dv to estimate LT integrated dv.

    Calibrated against the observed Sims-Flanagan results in
    ftp_strict_electric_only.pkl:
      - Mars->Fortuna  (LT ratio ≈ 1.74, dv/ceiling ≈ 0.55)
      - Fortuna->Themis (LT ratio ≈ 4.08, dv/ceiling ≈ 0.85)
      - Themis->Psyche  (LT ratio ≈ 1.26, dv/ceiling ≈ 0.42)

    Pattern: factor grows superlinearly as lambert_dv approaches the thrust
    ceiling (T·tof/m). Below 50% of ceiling LT is roughly 1.2-1.5x; above 80%
    it blows up because the engine can't keep up.

    Returns
    -------
    factor : float — multiply lambert_dv by this to get integrated LT dv.
        +inf if leg is infeasible (would need more thrust than available).
    """
    if tof_sec <= 0 or m_kg <= 0:
        return np.inf
    accel_kms2 = thrust_N / m_kg / 1e3
    dv_ceiling_kms = accel_kms2 * tof_sec
    if dv_ceiling_kms <= 0:
        return np.inf

    ratio = lambert_dv_kms / dv_ceiling_kms

    if ratio >= 0.95:
        return np.inf  # infeasible — engine can't deliver

    # Calibrated piecewise model. The denominator (1 - ratio) makes the factor
    # blow up smoothly as lambert_dv approaches dv_ceiling.
    base = 1.20 + 0.55 * ratio        # 1.2 floor, +0.55 per unit of ceiling use
    blowup = 0.40 / (1.0 - ratio)     # superlinear penalty as we approach ceiling
    factor = base + blowup - 0.40     # subtract baseline blowup at ratio=0

    # Short-cruise penalty: if tof < 1 yr, electric is wasteful — penalize.
    tof_yr = tof_sec / YEAR
    if tof_yr < 1.0:
        factor *= (2.0 - tof_yr)  # up to 2x penalty for sub-1yr cruise

    return factor


# =============================================================================
# CORE: mass-aware path evaluation
# =============================================================================

ARCH_CODES = ['CCC', 'CCE', 'CEC', 'CEE', 'ECC', 'ECE', 'EEC', 'EEE']


def compute_path_mass(a_id_1, a_id_2, a_id_3,
                       et_launch, et_flyby, et_arrive_1, et_stay_1,
                       et_arrive_2, et_stay_2, et_arrive_3,
                       flyby_name, m_revs, arch_code,
                       m_init_kg=DEFAULT_M_INIT_KG, thrust_N=DEFAULT_THRUST_N):
    """Evaluate delivered mass for a (timing, architecture) pair.

    Parameters
    ----------
    a_id_1, a_id_2, a_id_3 : str — SPICE IDs for the 3 asteroids
    et_*                   : floats — epochs (SPICE ET seconds past J2000)
    flyby_name             : 'mars' | 'moon' | 'earth'
    m_revs                 : tuple of 4 ints — Lambert revolutions per leg
    arch_code              : 3-char string in ARCH_CODES
    m_init_kg, thrust_N    : spacecraft params

    Returns
    -------
    dict with:
      m_final_kg, dv_total_kms, dv_equiv_kms, feasible (bool),
      per-leg dv (lambert_dv and integrated_dv if electric),
      per-leg mass progression
    """
    fb = FLYBY_BODIES[flyby_name]
    fb_id = fb['id']
    m_0, m_1, m_2, m_3 = m_revs

    # --- States from SPICE ---
    earth_r, earth_v = get_state('399', et_launch)
    flyby_r, flyby_v = get_state(fb_id, et_flyby)
    a1_arr_r, a1_arr_v = get_state(str(a_id_1), et_arrive_1)
    a1_lv_r,  a1_lv_v  = get_state(str(a_id_1), et_stay_1)
    a2_arr_r, a2_arr_v = get_state(str(a_id_2), et_arrive_2)
    a2_lv_r,  a2_lv_v  = get_state(str(a_id_2), et_stay_2)
    a3_arr_r, a3_arr_v = get_state(str(a_id_3), et_arrive_3)

    mu_flyby = get_mu(fb['mu_body'])
    safe_radius = get_radius(fb['radii_body']) + fb['min_alt']

    # --- Lambert solves (4 legs) ---
    e_lv,    fb_arr_lv, ef0 = solve_lambert(earth_r,   flyby_r,   (et_flyby   -et_launch)/DAY, m_0, MU_SUN)
    fb_dep,  a1_arr_lv, ef1 = solve_lambert(flyby_r,   a1_arr_r,  (et_arrive_1-et_flyby )/DAY, m_1, MU_SUN)
    a1_lv_lv,a2_arr_lv, ef2 = solve_lambert(a1_lv_r,   a2_arr_r,  (et_arrive_2-et_stay_1)/DAY, m_2, MU_SUN)
    a2_lv_lv,a3_arr_lv, ef3 = solve_lambert(a2_lv_r,   a3_arr_r,  (et_arrive_3-et_stay_2)/DAY, m_3, MU_SUN)

    if ef0 != 1 or ef1 != 1 or ef2 != 1 or ef3 != 1:
        return _infeasible_result(arch_code, m_init_kg, reason='lambert_fail')

    # --- Per-leg impulsive dv components (km/s) ---
    dv_launch_vec = e_lv - earth_v
    dv_flyby_pow  = compute_flyby_dv(fb_arr_lv, fb_dep, flyby_v, mu_flyby, safe_radius)
    # If the flyby geometry is infeasible (turn exceeds gravity-only max at the
    # safe periapsis altitude), compute_flyby_dv returns the 1000 km/s penalty.
    # Reject the trajectory entirely instead of letting the chain quietly drop
    # m_final to ~0 — the optimizer otherwise keeps these solutions around
    # because exp(-huge_dv/Isp) still produces a non-NaN score.
    if abs(dv_flyby_pow) > 100.0:
        return _infeasible_result(arch_code, m_init_kg, reason='flyby_infeasible')
    dv_A1_arr_vec = a1_arr_lv - a1_arr_v
    dv_A1_lv_vec  = a1_lv_lv  - a1_lv_v
    dv_A2_arr_vec = a2_arr_lv - a2_arr_v
    dv_A2_lv_vec  = a2_lv_lv  - a2_lv_v
    dv_A3_arr_vec = a3_arr_lv - a3_arr_v

    dv_launch_kms  = float(np.linalg.norm(dv_launch_vec))
    dv_flyby_kms   = float(abs(dv_flyby_pow))
    dv_A1_arr_kms  = float(np.linalg.norm(dv_A1_arr_vec))
    dv_A1_lv_kms   = float(np.linalg.norm(dv_A1_lv_vec))
    dv_A2_arr_kms  = float(np.linalg.norm(dv_A2_arr_vec))
    dv_A2_lv_kms   = float(np.linalg.norm(dv_A2_lv_vec))
    dv_A3_arr_kms  = float(np.linalg.norm(dv_A3_arr_vec))

    # --- Group into logical legs (post-separation; launch dv is launcher's job) ---
    # L1: powered flyby only (always chemical; 0 for unpowered Mars/Moon GA)
    # L2: flyby -> A1 (arrival burn only; departure is the unpowered flyby)
    # L3: A1 -> A2 (depart A1 + arrive A2)
    # L4: A2 -> A3 (depart A2 + arrive A3)
    dv_L1 = dv_flyby_kms
    dv_L2 = dv_A1_arr_kms
    dv_L3 = dv_A1_lv_kms + dv_A2_arr_kms
    dv_L4 = dv_A2_lv_kms + dv_A3_arr_kms

    tof_L2 = et_arrive_1 - et_flyby
    tof_L3 = et_arrive_2 - et_stay_1
    tof_L4 = et_arrive_3 - et_stay_2

    # arch_code[0] = L2 mode, arch_code[1] = L3 mode, arch_code[2] = L4 mode
    legs_info = [
        ('L1', dv_L1, 0.0,    'C'),                 # always chemical
        ('L2', dv_L2, tof_L2, arch_code[0]),
        ('L3', dv_L3, tof_L3, arch_code[1]),
        ('L4', dv_L4, tof_L4, arch_code[2]),
    ]

    # --- Tsiolkovsky chain through legs ---
    m = m_init_kg
    leg_results = []
    feasible = True
    dv_total_kms = 0.0

    for label, dv_lam, tof, mode in legs_info:
        if mode == 'C':
            isp = ISP_CHEM
            dv_used = dv_lam
            factor = 1.0
        else:  # electric
            isp = ISP_ELEC
            factor = gravity_loss_factor(dv_lam, tof, m, thrust_N)
            if not np.isfinite(factor):
                feasible = False
                dv_used = np.inf
                m = 0.0
                leg_results.append({
                    'label': label, 'mode': mode, 'isp_s': isp,
                    'dv_lambert_kms': dv_lam, 'factor': factor,
                    'dv_used_kms': dv_used, 'm_in_kg': m, 'm_out_kg': 0.0,
                })
                break
            dv_used = dv_lam * factor

        m_in = m
        m = m * tsiolkovsky_fraction(dv_used, isp)
        dv_total_kms += dv_used
        leg_results.append({
            'label': label, 'mode': mode, 'isp_s': isp,
            'dv_lambert_kms': dv_lam, 'factor': factor,
            'dv_used_kms': dv_used, 'm_in_kg': m_in, 'm_out_kg': m,
        })

    if feasible and m > 0:
        dv_equiv = -ISP_CHEM * G0 * np.log(m / m_init_kg) / 1000.0
    else:
        dv_equiv = 1e3

    return {
        'arch_code': arch_code,
        'm_final_kg': float(m),
        'm_init_kg': float(m_init_kg),
        'dv_total_kms': float(dv_total_kms),
        'dv_equiv_kms': float(dv_equiv),
        'feasible': bool(feasible),
        'leg_results': leg_results,
        # Save raw impulsive components in case caller wants them
        'dv_launch_kms': dv_launch_kms,
        'dv_flyby_kms':  dv_flyby_kms,
        'dv_A1_arr_kms': dv_A1_arr_kms,
        'dv_A1_lv_kms':  dv_A1_lv_kms,
        'dv_A2_arr_kms': dv_A2_arr_kms,
        'dv_A2_lv_kms':  dv_A2_lv_kms,
        'dv_A3_arr_kms': dv_A3_arr_kms,
        # Epochs for downstream verification
        'et_launch': et_launch, 'et_flyby': et_flyby,
        'et_arrive_1': et_arrive_1, 'et_stay_1': et_stay_1,
        'et_arrive_2': et_arrive_2, 'et_stay_2': et_stay_2,
        'et_arrive_3': et_arrive_3,
        'm_revs': m_revs, 'flyby_name': flyby_name, 'thrust_N': thrust_N,
    }


def _infeasible_result(arch_code, m_init, reason='infeasible'):
    return {
        'arch_code': arch_code, 'm_final_kg': 0.0, 'm_init_kg': m_init,
        'dv_total_kms': 1e3, 'dv_equiv_kms': 1e3, 'feasible': False,
        'leg_results': [], 'reason': reason,
    }


# =============================================================================
# DE OBJECTIVE WRAPPER
# =============================================================================

def score_paths_mass(input_vec, a_id_1, a_id_2, a_id_3, launch_range,
                     flyby_name, m_revs, arch_code,
                     m_init_kg=DEFAULT_M_INIT_KG, thrust_N=DEFAULT_THRUST_N):
    """DE objective: minimize -m_final (i.e., maximize delivered mass).

    Returns a large positive number for infeasible / mission-too-long cases.
    """
    ets = _unpack_flyby_input(input_vec, launch_range, flyby_name)
    if ets[6] - ets[0] > MAX_MISSION_DURATION:
        return 1e6
    out = compute_path_mass(a_id_1, a_id_2, a_id_3, *ets,
                             flyby_name, m_revs, arch_code,
                             m_init_kg=m_init_kg, thrust_N=thrust_N)
    if not out['feasible']:
        return 1e6
    # Minimize negative mass = maximize mass. Add small dv_equiv tiebreaker
    # so when masses are equal we prefer lower-dv solutions.
    return -out['m_final_kg'] + 1e-3 * out['dv_equiv_kms']


# =============================================================================
# DE PER ARCHITECTURE
# =============================================================================

def _bounds_flyby(flyby_name, launch_range):
    fb = FLYBY_BODIES[flyby_name]
    return list(zip(
        np.array([0, fb['tof_min'], 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], fb['tof_max'], 5*YEAR, YEAR,
                  5*YEAR, YEAR, 5*YEAR]) / YEAR))


def optimize_for_architecture(a_id_1, a_id_2, a_id_3, launch_range,
                              flyby_name, arch_code,
                              m_revs=(0, 0, 0, 0),
                              m_init_kg=DEFAULT_M_INIT_KG,
                              thrust_N=DEFAULT_THRUST_N,
                              seed=42, maxiter=200, popsize=15):
    """Run DE for a single architecture. Returns the best-mass result dict."""
    bounds = _bounds_flyby(flyby_name, launch_range)
    res = differential_evolution(
        lambda x: score_paths_mass(x, a_id_1, a_id_2, a_id_3, launch_range,
                                   flyby_name, m_revs, arch_code,
                                   m_init_kg=m_init_kg, thrust_N=thrust_N),
        bounds, maxiter=maxiter, tol=1e-6, seed=seed, polish=True,
        popsize=popsize, mutation=(0.5, 1.3), recombination=0.8,
        updating='deferred')

    ets = _unpack_flyby_input(res.x, launch_range, flyby_name)
    out = compute_path_mass(a_id_1, a_id_2, a_id_3, *ets,
                             flyby_name, m_revs, arch_code,
                             m_init_kg=m_init_kg, thrust_N=thrust_N)
    out['_de_x']   = res.x.copy()
    out['_de_fun'] = float(res.fun)
    out['_seed']   = seed
    return out


def pareto_optimize_triplet(a_id_1, a_id_2, a_id_3, launch_range,
                            flyby_name='mars', archs=None,
                            m_revs_options=None, seeds=(42, 137),
                            m_init_kg=DEFAULT_M_INIT_KG,
                            thrust_N=DEFAULT_THRUST_N,
                            maxiter=200, popsize=15, verbose=False):
    """Run mass-objective DE for every architecture, with multi-seed and a small
    Lambert-revolution sweep. Returns list of best result per architecture
    (the Pareto-front view comes from sorting these by dv_equiv vs m_final).

    Parameters
    ----------
    archs            : list of arch_code strings (default: all 8)
    m_revs_options   : list of (m0,m1,m2,m3) tuples to try (default: 4 sensible)
    seeds            : tuple of DE seeds for multi-start
    """
    if archs is None:
        archs = ARCH_CODES
    if m_revs_options is None:
        m_revs_options = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)]

    results_by_arch = {}
    for arch in archs:
        best = None
        for m_revs in m_revs_options:
            for seed in seeds:
                try:
                    out = optimize_for_architecture(
                        a_id_1, a_id_2, a_id_3, launch_range,
                        flyby_name, arch, m_revs=m_revs,
                        m_init_kg=m_init_kg, thrust_N=thrust_N,
                        seed=seed, maxiter=maxiter, popsize=popsize)
                except Exception as e:
                    if verbose:
                        print(f"  {arch} m={m_revs} seed={seed} ERROR: {e}")
                    continue
                if not out['feasible']:
                    continue
                if best is None or out['m_final_kg'] > best['m_final_kg']:
                    best = out
                    if verbose:
                        print(f"  {arch} m={m_revs} seed={seed}: "
                              f"m_final={out['m_final_kg']:.0f} kg "
                              f"dv_eq={out['dv_equiv_kms']:.2f} km/s")
        results_by_arch[arch] = best

    # Filter to feasible
    feasible = {a: r for a, r in results_by_arch.items() if r is not None}
    return feasible


# =============================================================================
# VERIFICATION: refine top candidates with full Sims-Flanagan LT solver
# =============================================================================

def verify_with_full_lt(arch_result, a_id_1, a_id_2, a_id_3,
                        m_init_kg=DEFAULT_M_INIT_KG,
                        thrust_N=DEFAULT_THRUST_N, nseg=DEFAULT_NSEG,
                        verbose=False):
    """Re-evaluate an arch_result with the real LT solver on every electric leg.

    Replaces the surrogate dv with the integrated Sims-Flanagan dv and
    recomputes the Tsiolkovsky chain. Returns a result dict mirroring
    `compute_path_mass` plus a 'verified' key.
    """
    arch_code = arch_result['arch_code']
    if arch_code == 'CCC':
        # Pure chemical — surrogate is exact.
        out = dict(arch_result)
        out['verified'] = True
        return out

    et_flyby    = arch_result['et_flyby']
    et_arrive_1 = arch_result['et_arrive_1']
    et_stay_1   = arch_result['et_stay_1']
    et_arrive_2 = arch_result['et_arrive_2']
    et_stay_2   = arch_result['et_stay_2']
    et_arrive_3 = arch_result['et_arrive_3']
    flyby_name  = arch_result['flyby_name']
    fb = FLYBY_BODIES[flyby_name]

    # Reconstruct chained masses leg by leg using the real LT solver where elec.
    m = m_init_kg
    refined_legs = []
    feasible = True

    # ---- L1: chemical (powered flyby only; launch is launcher's job) ----
    dv_L1 = arch_result['dv_flyby_kms']
    m_in = m; m = m * tsiolkovsky_fraction(dv_L1, ISP_CHEM)
    refined_legs.append({'label': 'L1', 'mode': 'C', 'dv_kms': dv_L1,
                          'm_in_kg': m_in, 'm_out_kg': m})

    leg_specs = [
        ('L2', arch_code[0],
            fb['id'],     et_flyby,    str(a_id_1), et_arrive_1,
            arch_result['dv_A1_arr_kms']),
        ('L3', arch_code[1],
            str(a_id_1), et_stay_1,    str(a_id_2), et_arrive_2,
            arch_result['dv_A1_lv_kms'] + arch_result['dv_A2_arr_kms']),
        ('L4', arch_code[2],
            str(a_id_2), et_stay_2,    str(a_id_3), et_arrive_3,
            arch_result['dv_A2_lv_kms'] + arch_result['dv_A3_arr_kms']),
    ]
    for label, mode, body0, et0, body1, et1, dv_lam in leg_specs:
        if not feasible:
            refined_legs.append({'label': label, 'mode': mode, 'skipped': True})
            continue
        if mode == 'C':
            m_in = m
            m = m * tsiolkovsky_fraction(dv_lam, ISP_CHEM)
            refined_legs.append({'label': label, 'mode': 'C', 'dv_kms': dv_lam,
                                  'm_in_kg': m_in, 'm_out_kg': m})
        else:  # electric — run the real solver
            r0, v0 = get_state(body0, et0)
            r1, v1 = get_state(body1, et1)
            tof = et1 - et0
            lt = optimize_lt_leg(r0, v0, r1, v1, tof,
                                 m_init_kg=m, thrust_N=thrust_N,
                                 isp_s=ISP_ELEC, nseg=nseg, verbose=False)
            if not lt['converged']:
                feasible = False
                refined_legs.append({'label': label, 'mode': 'E',
                                      'converged': False, 'reason': lt['reason']})
                continue
            m_in = m
            m = float(lt['m_final'])
            refined_legs.append({'label': label, 'mode': 'E',
                                  'dv_kms': float(lt['dv_integral_kms']),
                                  'm_in_kg': m_in, 'm_out_kg': m,
                                  'converged': True,
                                  'pos_err_km': lt['pos_err_km'],
                                  'vel_err_kms': lt['vel_err_kms']})

    if feasible and m > 0:
        dv_equiv = -ISP_CHEM * G0 * np.log(m / m_init_kg) / 1000.0
    else:
        dv_equiv = 1e3

    out = dict(arch_result)
    out['verified']            = True
    out['verified_feasible']   = feasible
    out['verified_m_final_kg'] = float(m)
    out['verified_dv_equiv_kms'] = float(dv_equiv)
    out['verified_legs']       = refined_legs
    if verbose:
        print(f"  arch {arch_code}: surrogate m={arch_result['m_final_kg']:.0f} kg, "
              f"verified m={m:.0f} kg "
              f"({'OK' if feasible else 'INFEASIBLE'})")
    return out
