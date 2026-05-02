"""Low-thrust leg solver — Sims-Flanagan direct transcription via scipy.

Splits a heliocentric leg into N segments. Each segment has:
  - Half-segment Kepler coast (propagate_lagrangian)
  - Impulsive kick bounded by max_dv_seg = T · Δt / m
  - Half-segment Kepler coast

Decision vars: throttle unit-vectors u_i ∈ [-1,1]^3 per segment (|u_i| ≤ 1).
Objective: match target state at end while minimizing propellant.

No pygmo dependency — uses scipy.optimize.least_squares with a composite
residual that enforces boundary matching and penalizes excess throttle.
"""

import numpy as np
from scipy.optimize import least_squares

import pykep as pk
from core import MU_SUN, DAY, YEAR

_KM2M, _M2KM = 1e3, 1e-3
G0 = 9.80665
ISP_CHEM = 320.0
ISP_ELEC = 3100.0

# Default spacecraft wet mass at Earth departure
DEFAULT_M_INIT_KG = 3000.0
DEFAULT_THRUST_N  = 0.30
MAX_THRUST_N      = 0.30  # 300 mN hard cap
DEFAULT_NSEG      = 15


def tsiolkovsky_fraction(dv_kms, isp_s):
    """m_final / m_initial for one impulsive Δv at given Isp."""
    return float(np.exp(-dv_kms * 1e3 / (isp_s * G0)))


def _propagate_km(r_km, v_kms, tof_sec):
    """Kepler propagation wrapper keeping our km/s units."""
    r_m  = [float(x) * _KM2M for x in r_km]
    v_ms = [float(x) * _KM2M for x in v_kms]
    rf, vf = pk.propagate_lagrangian(r0=r_m, v0=v_ms,
                                     tof=float(tof_sec),
                                     mu=MU_SUN * 1e9)
    return np.array(rf) * _M2KM, np.array(vf) * _M2KM


def _forward_propagate(r0, v0, m0, throttles, tof_sec, thrust_N, isp_s, nseg):
    """Forward-integrate N segments with Sims-Flanagan impulse kicks.

    Returns (r_f, v_f, m_f, dv_total) — final state + final mass + Σ|Δv_i|.
    """
    dt = tof_sec / nseg
    r, v, m = np.asarray(r0, float), np.asarray(v0, float), float(m0)
    dv_total = 0.0
    for i in range(nseg):
        u = np.asarray(throttles[3 * i: 3 * i + 3])
        # Half-coast
        r, v = _propagate_km(r, v, dt / 2)
        # Impulse: δv_max = T·dt / m  (converted to km/s)
        dv_max_kms = thrust_N * dt / m / 1e3
        dv_vec = u * dv_max_kms  # in km/s
        v = v + dv_vec
        # Mass update per Tsiolkovsky
        dv_mag = np.linalg.norm(dv_vec)
        m = m * np.exp(-dv_mag * 1e3 / (isp_s * G0))
        dv_total += dv_mag
        # Second half-coast
        r, v = _propagate_km(r, v, dt / 2)
    return r, v, m, dv_total


def _seed_from_lambert(r0, v0, r1, v1, tof_sec, nseg):
    """Throttle seed: Lambert solution split across segments.

    Use Lambert to find the two-body trajectory connecting (r0, r1) in tof.
    The boundary-matching Δvs become a proxy for each segment's impulse."""
    from core import solve_lambert
    V1, V2, ef = solve_lambert(r0, r1, tof_sec / DAY, 0, MU_SUN)
    if ef != 1:
        return np.zeros(3 * nseg)
    dv_dep = V1 - np.asarray(v0)
    dv_arr = np.asarray(v1) - V2
    # Apply departure dv in the first half of segments, arrival in the last half
    seed = np.zeros((nseg, 3))
    split = nseg // 2
    if split > 0:
        seed[:split] = dv_dep / split / 5.0   # scale down — not all of Δv needed as LT
        seed[split:] = dv_arr / (nseg - split) / 5.0
    # Normalize into [-1,1] range assuming this is a fraction of per-segment max
    maxabs = np.max(np.abs(seed)) if np.max(np.abs(seed)) > 0 else 1.0
    if maxabs > 1:
        seed = seed / maxabs * 0.8
    return seed.flatten()


def optimize_lt_leg(r0_km, v0_kms, r1_km, v1_kms, tof_sec,
                    m_init_kg=DEFAULT_M_INIT_KG, thrust_N=DEFAULT_THRUST_N,
                    isp_s=ISP_ELEC, nseg=DEFAULT_NSEG, verbose=False,
                    max_nfev=1500, reg_weight=0.05):
    """Optimize a low-thrust leg.

    Parameters
    ----------
    r0_km, v0_kms : array(3,) — initial heliocentric state, km / km/s
    r1_km, v1_kms : array(3,) — target heliocentric state
    tof_sec       : float     — time of flight, seconds
    m_init_kg     : float     — initial spacecraft mass, kg
    thrust_N      : float     — continuous thrust magnitude, Newtons
    isp_s         : float     — specific impulse, seconds
    nseg          : int       — Sims-Flanagan segment count

    Returns
    -------
    dict with keys: m_final, dv_integral_kms, throttles, pos_err_km, vel_err_kms,
                     converged (bool), reason (str)
    """
    r0 = np.asarray(r0_km, float); v0 = np.asarray(v0_kms, float)
    r1 = np.asarray(r1_km, float); v1 = np.asarray(v1_kms, float)

    if thrust_N < 0:
        return {'converged': False, 'reason': f'invalid thrust_N={thrust_N:.6f} N (must be >= 0)',
                'm_final': 0.0, 'dv_integral_kms': np.inf}
    if thrust_N > MAX_THRUST_N:
        return {'converged': False, 'reason': f'thrust cap exceeded: {thrust_N:.6f} N > {MAX_THRUST_N:.2f} N',
                'm_final': 0.0, 'dv_integral_kms': np.inf}

    # Feasibility: can thrust deliver the needed Δv in the given TOF?
    accel_kms2 = thrust_N / m_init_kg / 1e3
    dv_ceiling_kms = accel_kms2 * tof_sec
    lambert_dv_est = np.linalg.norm(r1 - r0) / tof_sec  # crude km/s

    def residuals(throttles):
        rf, vf, _, _ = _forward_propagate(r0, v0, m_init_kg, throttles,
                                           tof_sec, thrust_N, isp_s, nseg)
        pos_err = (rf - r1) / 1e6        # scale to ~O(1): 1e6 km ≈ 0.007 AU
        vel_err = (vf - v1) * 10          # 1 km/s ≈ 10 units → comparable weight
        # Regularization: penalize throttle magnitude to prefer fuel-optimal solns.
        # Weight scaled relative to mismatch residuals (which are O(1) at error
        # boundaries 1e6 km / 0.1 km/s).  Too weak -> over-throttling.
        reg = reg_weight * throttles
        # Hard cap: physical engine can't exceed thrust_N.  Solver bounds are a
        # cube [-1,1]^3 but the engine constraint is |u| <= 1 (unit ball).
        # Add a one-sided smooth penalty that activates when |u_i| > 1 per segment.
        u_vecs = throttles.reshape(-1, 3)
        u_mags = np.linalg.norm(u_vecs, axis=1)
        over   = np.maximum(0.0, u_mags - 1.0)  # 0 if feasible, else excess
        thrust_pen = 50.0 * over  # weight large enough to dominate when over-budget
        return np.concatenate([pos_err, vel_err, reg, thrust_pen])

    seed = _seed_from_lambert(r0, v0, r1, v1, tof_sec, nseg)
    bounds = (-np.ones(3 * nseg), np.ones(3 * nseg))

    try:
        res = least_squares(residuals, seed, bounds=bounds,
                            method='trf', max_nfev=max_nfev,
                            xtol=1e-7, ftol=1e-7, verbose=0)
    except Exception as e:
        return {'converged': False, 'reason': f'solver_exception: {e}',
                'm_final': 0.0, 'dv_integral_kms': np.inf}

    throttles = res.x
    rf, vf, mf, dv_int = _forward_propagate(r0, v0, m_init_kg, throttles,
                                             tof_sec, thrust_N, isp_s, nseg)
    pos_err_km  = float(np.linalg.norm(rf - r1))
    vel_err_kms = float(np.linalg.norm(vf - v1))

    # Convergence criteria: within 1% of 1 AU position, 0.1 km/s velocity
    converged = (pos_err_km < 1.5e6) and (vel_err_kms < 0.15)
    reason = ('ok' if converged
              else f'mismatch pos={pos_err_km:.2e} km, vel={vel_err_kms:.3f} km/s')

    if verbose:
        print(f"  LT leg: TOF={tof_sec/YEAR:.2f}yr  "
              f"dv_ceil={dv_ceiling_kms:.2f}  "
              f"dv_int={dv_int:.2f}  m_f={mf:.0f}  "
              f"pos_err={pos_err_km:.1e}  vel_err={vel_err_kms:.3f}  "
              f"{'✓' if converged else '✗'}")

    return {
        'converged': converged, 'reason': reason,
        'm_final': mf, 'dv_integral_kms': dv_int,
        'throttles': throttles, 'nseg': nseg,
        'pos_err_km': pos_err_km, 'vel_err_kms': vel_err_kms,
        'isp_s': isp_s, 'thrust_N': thrust_N, 'm_init_kg': m_init_kg,
    }
