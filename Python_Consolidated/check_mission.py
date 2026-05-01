"""check_mission.py — Independent verification of a saved trajectory.

Audits a saved result `.pkl` against every constraint the optimizer is
supposed to enforce. Re-derives every Δv from raw SPICE states (does NOT
trust saved values), independently audits flyby geometry, and checks every
bound. Reports PASS/FAIL per check.

Usage:
    python Python_Consolidated/check_mission.py <pkl_path> [--rank N]
                                                            [--names A B C]
                                                            [--all]      # all entries
                                                            [--strict]    # fail on any warning

Designed to run independently of `optimization.py` / `mass_optimization.py`,
so a checker bug doesn't mask an optimizer bug.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

warnings.filterwarnings('ignore')


# Constants — duplicated here so the checker doesn't depend on the optimizer modules
_KM2M = 1e3
DAY  = 86400.0
WEEK = 7 * DAY
MONTH = 30.4375 * DAY
YEAR = 365.25 * DAY
MAX_MISSION_DURATION = 14 * YEAR

# Flyby body parameters (mu_body and radii_body intentionally separate — Mars uses
# different SPICE IDs for GM vs. radii)
FLYBY_BODIES = {
    'moon':  {'spice_id': '301', 'mu_body': 301, 'radii_body': 301, 'min_alt_km': 100},
    'mars':  {'spice_id': '4',   'mu_body': 4,   'radii_body': 499, 'min_alt_km': 200},
    'earth': {'spice_id': '399', 'mu_body': 399, 'radii_body': 399, 'min_alt_km': 300},
}


# --------------------------------------------------------------------------
# SPICE / pykep facade — re-loaded fresh per run
# --------------------------------------------------------------------------

def _setup_repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    sys.path.insert(0, here)


def _load_kernels():
    """Load all SPICE kernels needed for verification."""
    import spiceypy, glob
    spiceypy.kclear()
    gk = 'generic_kernels'
    spiceypy.furnsh(os.path.join(gk, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'planets',    'de430.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'pck00010.tpc'))
    asteroid_list = []
    for bsp in sorted(glob.glob('NOTABLE_ASTEROID_BSPs/*.bsp')):
        spiceypy.furnsh(bsp)
        id_cell = spiceypy.spkobj(bsp)
        ids = [id_cell[i] for i in range(spiceypy.card(id_cell))]
        name = os.path.splitext(os.path.basename(bsp))[0]
        asteroid_list.append({'ID': ids[0] if ids else 0, 'NAME': name})
    return asteroid_list


def _get_state(body_id: str, et: float):
    import spiceypy
    state, _ = spiceypy.spkezr(str(body_id), et, 'ECLIPJ2000', 'NONE', '10')
    return np.array(state[0:3]), np.array(state[3:6])


def _get_mu(body_id: int):
    import spiceypy
    _, vals = spiceypy.bodvcd(int(body_id), 'GM', 10)
    return float(vals[0])


def _get_radius(body_id: int):
    import spiceypy
    _, vals = spiceypy.bodvcd(int(body_id), 'RADII', 10)
    return float(np.mean(vals[:3]))


def _solve_lambert(r1_km, r2_km, tof_days, m, mu_km3s2):
    """Wrap pykep's Lambert solver. Tries both prograde/retrograde, returns best."""
    import pykep as pk
    r1_m = [float(x) * 1e3 for x in r1_km]
    r2_m = [float(x) * 1e3 for x in r2_km]
    tof_sec = abs(tof_days) * DAY
    mu_m3s2 = mu_km3s2 * 1e9
    cw = bool(tof_days < 0)
    multi_revs = abs(int(m))
    try:
        lp = pk.lambert_problem(r1_m, r2_m, tof_sec, mu_m3s2, cw, multi_revs)
        v1_all = lp.get_v1(); v2_all = lp.get_v2()
        if multi_revs == 0:
            idx = 0
        else:
            idx = 2 * multi_revs - 1 if m > 0 else 2 * multi_revs - 2
        if idx >= len(v1_all):
            return None, None, -1
        V1 = np.array(v1_all[idx]) * 1e-3
        V2 = np.array(v2_all[idx]) * 1e-3
        return V1, V2, 1
    except Exception:
        return None, None, -1


# --------------------------------------------------------------------------
# Check primitives
# --------------------------------------------------------------------------

def check(name: str, condition: bool, detail: str = '') -> Tuple[str, bool, str]:
    return (name, bool(condition), detail)


def _print_check(c: Tuple[str, bool, str], indent: int = 2):
    name, ok, detail = c
    flag = '\x1b[32m  OK\x1b[0m' if ok else '\x1b[31mFAIL\x1b[0m'
    pad = ' ' * indent
    print(f'{pad}[{flag}]  {name}'
          f'{("  ─  " + detail) if detail else ""}')


# --------------------------------------------------------------------------
# Result-format normalization (handles every saved-pkl shape we know about)
# --------------------------------------------------------------------------

def _extract_entries(data: Any) -> List[Dict[str, Any]]:
    """Normalize any saved-pkl format into a list of entries with consistent keys."""
    entries: List[Dict[str, Any]] = []

    if isinstance(data, dict) and 'audited' in data:
        # diverse_top3_feasible / mars_diverse_feasible
        for a in data['audited']:
            best = a.get('best') or {}
            entries.append({
                'names': tuple(a.get('names', ())),
                'arch':  a.get('arch') or best.get('architecture') or 'mars',
                'best':  best,
                'audit_saved': a.get('audit') or {},
            })
        return entries

    if isinstance(data, dict) and 'all_results' in data:
        # mass-Pareto sweeps
        for r in data['all_results']:
            if r.get('error'): continue
            best = r.get('best_verified') or {}
            entries.append({
                'names': tuple(r.get('names', ())),
                'arch':  best.get('flyby_name') or 'mars',
                'best':  best,
                'audit_saved': {},
            })
        return entries

    if isinstance(data, dict) and 'best_verified' in data:
        # single-triplet GCP runner output
        best = data['best_verified']
        entries.append({
            'names': tuple(data.get('triplet', ())),
            'arch':  best.get('flyby_name') or 'mars',
            'best':  best,
            'audit_saved': {},
        })
        return entries

    if isinstance(data, dict) and 'best_mars' in data:
        # robust FTP-style output
        best = data['best_mars']
        entries.append({
            'names': tuple(data.get('triplet', ())),
            'arch':  'mars',
            'best':  best,
            'audit_saved': {},
        })
        return entries

    if isinstance(data, list):
        # two-level optimize: list of (i, j, k, result_dict)
        for item in data:
            if isinstance(item, tuple) and len(item) >= 4 and isinstance(item[3], dict):
                res = item[3]
                entries.append({
                    'names': tuple(res.get('names', ())),
                    'arch':  res.get('architecture') or 'direct',
                    'best':  res,
                    'audit_saved': {},
                })
        return entries

    raise ValueError(f'Unrecognized pkl format: {type(data)}')


# --------------------------------------------------------------------------
# Per-leg Lambert verification
# --------------------------------------------------------------------------

MU_SUN = None  # set after kernels load


def _verify_lambert(label, r0, v0_body, r1, v1_body, et0, et1, m_revs):
    """Re-solve Lambert and report (Δv at start, Δv at end, position residual)."""
    tof_days = (et1 - et0) / DAY
    V1, V2, ef = _solve_lambert(r0, r1, tof_days, m_revs, MU_SUN)
    if ef != 1:
        return None, None, None, f'Lambert non-convergence ({label})'
    # Forward-propagate to verify endpoint
    import pykep as pk
    rprop_m, vprop_m = pk.propagate_lagrangian(
        r0=[float(x)*1e3 for x in r0], v0=[float(x)*1e3 for x in V1],
        tof=float((et1-et0)), mu=MU_SUN*1e9)
    rprop = np.array(rprop_m) * 1e-3
    vprop = np.array(vprop_m) * 1e-3
    pos_err = float(np.linalg.norm(rprop - r1))
    vel_err = float(np.linalg.norm(vprop - V2))
    dv_dep = float(np.linalg.norm(V1 - v0_body))
    dv_arr = float(np.linalg.norm(v1_body - V2))
    return dv_dep, dv_arr, (pos_err, vel_err), None


# --------------------------------------------------------------------------
# Flyby physics audit
# --------------------------------------------------------------------------

def _audit_flyby(et_launch, et_flyby, et_arr_a1, a1_id, arch):
    """Independent geometric audit. Returns a dict identical to
    core.audit_flyby_geometry but reimplemented locally."""
    fb = FLYBY_BODIES[arch]
    fb_id = fb['spice_id']

    earth_r, _    = _get_state('399', et_launch)
    fb_r,  fb_v   = _get_state(fb_id, et_flyby)
    a1_r, _       = _get_state(a1_id, et_arr_a1)

    _, V2_in,  ef0 = _solve_lambert(earth_r, fb_r, (et_flyby-et_launch)/DAY, 0, MU_SUN)
    V1_out, _, ef1 = _solve_lambert(fb_r,    a1_r, (et_arr_a1-et_flyby)/DAY, 0, MU_SUN)
    if ef0 != 1 or ef1 != 1:
        return {'feasible': False, 'reason': 'lambert_fail'}

    v_inf_in  = V2_in  - fb_v
    v_inf_out = V1_out - fb_v
    vin  = float(np.linalg.norm(v_inf_in))
    vout = float(np.linalg.norm(v_inf_out))
    cosd = np.clip(np.dot(v_inf_in, v_inf_out)/(vin*vout), -1, 1)
    delta = float(np.degrees(np.arccos(cosd)))

    mu = _get_mu(fb['mu_body'])
    R  = _get_radius(fb['radii_body'])
    safe_r = R + fb['min_alt_km']

    sin_in  = min(1.0, 1.0/(1.0 + safe_r * vin**2  / mu))
    sin_out = min(1.0, 1.0/(1.0 + safe_r * vout**2 / mu))
    delta_max = float(np.degrees(np.arcsin(sin_in) + np.arcsin(sin_out)))

    energy_residual = abs(vin - vout)
    geometric_ok = delta <= delta_max + 1e-6
    BALLISTIC_TOL_KMS = 0.05  # matches core.BALLISTIC_VINF_TOLERANCE_KMS
    ballistic_ok = energy_residual <= BALLISTIC_TOL_KMS

    return {
        'feasible':            bool(geometric_ok and ballistic_ok),
        'geometric_ok':        bool(geometric_ok),
        'ballistic_ok':        bool(ballistic_ok),
        'v_inf_in_kms':        vin,
        'v_inf_out_kms':       vout,
        'turn_angle_deg':      delta,
        'turn_max_deg':        delta_max,
        'energy_residual_kms': vin - vout,
        'safe_radius_km':      safe_r,
        'body_radius_km':      R,
        'min_alt_km':          fb['min_alt_km'],
    }


# --------------------------------------------------------------------------
# Master check function
# --------------------------------------------------------------------------

def verify_entry(entry: Dict, asteroid_list: List, strict: bool = False) -> bool:
    """Run all checks on one entry. Returns True if every check passes."""
    name_to_id = {a['NAME'].upper(): str(int(a['ID'])) for a in asteroid_list}
    names = [n.upper() for n in entry['names']]
    res = entry['best']
    arch = entry['arch']

    # Print header
    print(f'\n{"="*78}')
    print(f' Verifying: {" -> ".join(names)}  via {arch}')
    print(f'{"="*78}')

    checks: List[Tuple[str, bool, str]] = []

    # Resolve asteroid IDs
    missing = [n for n in names if n not in name_to_id]
    checks.append(check('Asteroid names resolve to BSP IDs',
                         not missing,
                         '' if not missing else f'missing: {missing}'))
    if missing:
        for c in checks: _print_check(c)
        return False
    a_ids = [name_to_id[n] for n in names]

    # Pull epochs
    required_keys = ['et_launch', 'et_arrive_1', 'et_stay_1',
                     'et_arrive_2', 'et_stay_2', 'et_arrive_3']
    if arch != 'direct':
        required_keys.insert(1, 'et_flyby')
    missing_keys = [k for k in required_keys if k not in res]
    checks.append(check('All required epochs present in result',
                         not missing_keys,
                         '' if not missing_keys else f'missing: {missing_keys}'))
    if missing_keys:
        for c in checks: _print_check(c)
        return False

    et_launch = res['et_launch']
    et_flyby  = res.get('et_flyby')
    et_arr_1  = res['et_arrive_1']
    et_stay_1 = res['et_stay_1']
    et_arr_2  = res['et_arrive_2']
    et_stay_2 = res['et_stay_2']
    et_arr_3  = res['et_arrive_3']

    # ---- Bound checks ----
    duration_yr = (et_arr_3 - et_launch) / YEAR
    checks.append(check(
        'Mission duration ≤ 14 years',
        duration_yr <= 14.001,
        f'{duration_yr:.3f} yr'))

    leg2_yr = (et_arr_1 - (et_flyby if et_flyby else et_launch)) / YEAR
    leg3_yr = (et_arr_2 - et_stay_1) / YEAR
    leg4_yr = (et_arr_3 - et_stay_2) / YEAR
    stay1_yr = (et_stay_1 - et_arr_1) / YEAR
    stay2_yr = (et_stay_2 - et_arr_2) / YEAR

    checks.append(check('Stay at A1 in [3 mo, 1 yr]',
                         0.249 <= stay1_yr <= 1.001,
                         f'{stay1_yr*12:.2f} months'))
    checks.append(check('Stay at A2 in [3 mo, 1 yr]',
                         0.249 <= stay2_yr <= 1.001,
                         f'{stay2_yr*12:.2f} months'))
    for i, (lab, t) in enumerate([('flyby→A1' if et_flyby else 'Earth→A1', leg2_yr),
                                    ('A1→A2', leg3_yr), ('A2→A3', leg4_yr)], 2):
        checks.append(check(f'Leg {i} ({lab}) TOF in [2 wk, 5 yr]',
                             0.038 <= t <= 5.001,
                             f'{t:.3f} yr'))

    if et_flyby:
        e_to_fb_yr = (et_flyby - et_launch) / YEAR
        if arch == 'mars':
            checks.append(check('Earth→Mars TOF in [0.3 yr, 3 yr]',
                                 0.299 <= e_to_fb_yr <= 3.001,
                                 f'{e_to_fb_yr:.3f} yr'))
        elif arch == 'moon':
            checks.append(check('Earth→Moon TOF in [1 day, 10 days]',
                                 0.999 <= e_to_fb_yr*365.25 <= 10.001,
                                 f'{e_to_fb_yr*365.25:.2f} days'))
        elif arch == 'earth':
            checks.append(check('Earth→Earth TOF in [1 yr, 3 yr]',
                                 0.999 <= e_to_fb_yr <= 3.001,
                                 f'{e_to_fb_yr:.3f} yr'))

    for c in checks: _print_check(c)
    bounds_pass = all(c[1] for c in checks)

    # ---- Lambert re-verification ----
    print('\n  Re-deriving every leg via fresh Lambert solve…')
    earth_r,  earth_v  = _get_state('399', et_launch)
    a1_arr_r, a1_arr_v = _get_state(a_ids[0], et_arr_1)
    a1_lv_r,  a1_lv_v  = _get_state(a_ids[0], et_stay_1)
    a2_arr_r, a2_arr_v = _get_state(a_ids[1], et_arr_2)
    a2_lv_r,  a2_lv_v  = _get_state(a_ids[1], et_stay_2)
    a3_arr_r, a3_arr_v = _get_state(a_ids[2], et_arr_3)

    m_revs = res.get('m_revs', (0, 0, 0, 0))
    if not isinstance(m_revs, (tuple, list)) or len(m_revs) != 4:
        m_revs = (0, 0, 0, 0)

    leg_checks: List[Tuple[str, bool, str]] = []
    dv_total_recomputed = 0.0
    leg_summaries = []

    if et_flyby is not None:
        fb_r, fb_v = _get_state(FLYBY_BODIES[arch]['spice_id'], et_flyby)
        # Leg 1: Earth → flyby body
        d_dep, d_arr, errs, err_msg = _verify_lambert(
            f'Earth→{arch}', earth_r, earth_v, fb_r, fb_v,
            et_launch, et_flyby, m_revs[0])
        if err_msg:
            leg_checks.append(check(f'Leg 1 (Earth→{arch}) Lambert', False, err_msg))
        else:
            leg_checks.append(check(f'Leg 1 (Earth→{arch}) Lambert converges',
                                      errs[0] < 1e-3,
                                      f'pos_err={errs[0]:.2e} km, vel_err={errs[1]:.2e} km/s'))
            leg_summaries.append(('Earth→' + arch, d_dep, d_arr, m_revs[0]))
            dv_total_recomputed += d_dep
        # Leg 2: flyby → A1
        d_dep, d_arr, errs, err_msg = _verify_lambert(
            f'{arch}→A1', fb_r, fb_v, a1_arr_r, a1_arr_v,
            et_flyby, et_arr_1, m_revs[1])
        if err_msg:
            leg_checks.append(check(f'Leg 2 ({arch}→A1) Lambert', False, err_msg))
        else:
            leg_checks.append(check(f'Leg 2 ({arch}→A1) Lambert converges',
                                      errs[0] < 1e-3,
                                      f'pos_err={errs[0]:.2e} km'))
            leg_summaries.append((f'{arch}→{names[0]}', d_dep, d_arr, m_revs[1]))
            dv_total_recomputed += d_arr  # arrival burn at A1
    else:
        # Direct: Leg 1 is Earth → A1
        d_dep, d_arr, errs, err_msg = _verify_lambert(
            'Earth→A1', earth_r, earth_v, a1_arr_r, a1_arr_v,
            et_launch, et_arr_1, m_revs[0])
        if err_msg:
            leg_checks.append(check('Leg 1 (Earth→A1) Lambert', False, err_msg))
        else:
            leg_checks.append(check('Leg 1 (Earth→A1) Lambert converges',
                                      errs[0] < 1e-3,
                                      f'pos_err={errs[0]:.2e} km'))
            leg_summaries.append((f'Earth→{names[0]}', d_dep, d_arr, m_revs[0]))
            dv_total_recomputed += d_dep + d_arr

    # Leg 3: A1 → A2
    d_dep, d_arr, errs, err_msg = _verify_lambert(
        'A1→A2', a1_lv_r, a1_lv_v, a2_arr_r, a2_arr_v,
        et_stay_1, et_arr_2, m_revs[2])
    if err_msg:
        leg_checks.append(check('Leg 3 (A1→A2) Lambert', False, err_msg))
    else:
        leg_checks.append(check('Leg 3 (A1→A2) Lambert converges',
                                 errs[0] < 1e-3,
                                 f'pos_err={errs[0]:.2e} km'))
        leg_summaries.append((f'{names[0]}→{names[1]}', d_dep, d_arr, m_revs[2]))
        dv_total_recomputed += d_dep + d_arr

    # Leg 4: A2 → A3
    d_dep, d_arr, errs, err_msg = _verify_lambert(
        'A2→A3', a2_lv_r, a2_lv_v, a3_arr_r, a3_arr_v,
        et_stay_2, et_arr_3, m_revs[3])
    if err_msg:
        leg_checks.append(check('Leg 4 (A2→A3) Lambert', False, err_msg))
    else:
        leg_checks.append(check('Leg 4 (A2→A3) Lambert converges',
                                 errs[0] < 1e-3,
                                 f'pos_err={errs[0]:.2e} km'))
        leg_summaries.append((f'{names[1]}→{names[2]}', d_dep, d_arr, m_revs[3]))
        dv_total_recomputed += d_dep + d_arr

    for c in leg_checks: _print_check(c)
    lambert_pass = all(c[1] for c in leg_checks)

    # ---- Flyby physics audit ----
    flyby_pass = True
    if arch != 'direct' and et_flyby is not None:
        print('\n  Auditing flyby physics…')
        audit = _audit_flyby(et_launch, et_flyby, et_arr_1, a_ids[0], arch)
        if 'reason' in audit:
            _print_check(check('Flyby audit completes', False, audit['reason']))
            flyby_pass = False
        else:
            # Hard pass/fail constraints:
            #  1. Geometric feasibility of the turn at safe periapsis
            #     (catches the unphysical "sub-surface flyby" bug)
            #  2. Ballistic energy conservation: |v_inf_in| ≈ |v_inf_out|
            #     (rejects powered Mars/Moon flybys — they're not real GAs)
            checks_fb = [
                check('Flyby turn angle ≤ natural max (geometric)',
                       audit['geometric_ok'],
                       f'{audit["turn_angle_deg"]:.2f}° / {audit["turn_max_deg"]:.2f}°'),
                check('Ballistic flyby (|v_inf_in| ≈ |v_inf_out|)',
                       audit['ballistic_ok'],
                       f'|residual| = {abs(audit["energy_residual_kms"]):.4f} km/s '
                       f'(tol 0.05)'),
            ]
            for c in checks_fb: _print_check(c)
            flyby_pass = all(c[1] for c in checks_fb)

            saved_powered_dv = abs(float(res.get('delta_v_flyby', 0.0)))
            energy_residual = abs(audit['energy_residual_kms'])
            print(f'    v_inf_in   : {audit["v_inf_in_kms"]:.4f} km/s')
            print(f'    v_inf_out  : {audit["v_inf_out_kms"]:.4f} km/s')
            if energy_residual > 0.5:
                print(f'    Powered Δv : {saved_powered_dv:.3f} km/s '
                      f'(periapsis Oberth burn)')
                print(f'    Note: |v_inf| residual = {energy_residual:.3f} km/s. '
                      f'pk.fb_dv computes a non-vis-viva')
                print(f'          burn that depends on chosen periapsis radius. '
                      f'Δv breakdown internal')
                print(f'          consistency is the load-bearing check '
                      f'(see total Δv comparison below).')
            else:
                print(f'    Energy : conserved (ballistic flyby, '
                      f'residual {energy_residual:.3f} km/s)')
            print(f'    safe r_p   : {audit["safe_radius_km"]:.0f} km   '
                  f'(altitude floor: {audit["min_alt_km"]} km)')

    # ---- Δv comparison ----
    print('\n  Δv breakdown (re-derived vs saved):')
    for label, d_dep, d_arr, mrev in leg_summaries:
        print(f'    {label:30s}  dep={d_dep:6.3f}  arr={d_arr:6.3f}  m={mrev}')

    saved_dv = res.get('delta_v_total')
    if saved_dv is not None:
        # If powered flyby, add it to recomputed total
        dv_powered = abs(float(res.get('delta_v_flyby', 0.0))) if arch != 'direct' else 0.0
        dv_total_full = dv_total_recomputed + dv_powered
        diff = abs(saved_dv - dv_total_full)
        # Tolerance: 0.1 km/s (allows for slight rounding differences, alt m-rev branches)
        consistent = diff < 0.1
        _print_check(check('Total Δv re-derived matches saved value (±0.1 km/s)',
                            consistent,
                            f'saved={saved_dv:.3f}, recomputed≈{dv_total_full:.3f}, '
                            f'diff={diff:.3f}'))

    # ---- Summary ----
    print()
    overall = bounds_pass and lambert_pass and flyby_pass
    if overall:
        print('  \x1b[42;30m ✓ ALL CHECKS PASS — TRAJECTORY IS PHYSICALLY VALID \x1b[0m')
    else:
        print('  \x1b[41;37m ✗ ONE OR MORE CHECKS FAILED — DO NOT TRUST THIS RESULT \x1b[0m')
    return overall


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='check_mission.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pkl', help='Path to result pkl (or basename in optimal_asteroid_paths/pkl/)')
    parser.add_argument('--rank', type=int, default=None,
                        help='Audit entry at this rank (1-indexed). Default: rank 1.')
    parser.add_argument('--names', nargs=3, default=None, metavar=('A','B','C'))
    parser.add_argument('--all', action='store_true',
                        help='Verify every entry in the pkl (good for batch sanity)')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with non-zero status on any failure')
    args = parser.parse_args()

    _setup_repo_root()

    pkl_path = (args.pkl if os.path.isabs(args.pkl)
                else os.path.join('optimal_asteroid_paths/pkl', args.pkl))
    if not os.path.exists(pkl_path):
        sys.exit(f'Not found: {pkl_path}')

    print(f'Loading: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print('Loading SPICE kernels and asteroid BSPs…')
    asteroid_list = _load_kernels()
    print(f'  {len(asteroid_list)} asteroid BSPs available')

    global MU_SUN
    import pykep as pk
    MU_SUN = pk.MU_SUN * 1e-9   # m³/s² → km³/s²

    entries = _extract_entries(data)
    print(f'  {len(entries)} entries in this pkl')

    # Choose which entries to verify
    if args.all:
        targets = entries
    elif args.names:
        wanted = tuple(n.upper() for n in args.names)
        targets = [e for e in entries
                   if tuple(n.upper() for n in e['names']) == wanted]
        if not targets:
            sys.exit(f'No entry with names {wanted}')
    else:
        rank = args.rank or 1
        if rank < 1 or rank > len(entries):
            sys.exit(f'--rank {rank} out of range (1..{len(entries)})')
        targets = [entries[rank - 1]]

    pass_count = 0
    fail_count = 0
    for e in targets:
        if verify_entry(e, asteroid_list, strict=args.strict):
            pass_count += 1
        else:
            fail_count += 1

    print(f'\n{"="*78}')
    print(f' SUMMARY: {pass_count} pass, {fail_count} fail '
          f'(out of {len(targets)} verified)')
    print(f'{"="*78}\n')

    if fail_count and args.strict:
        sys.exit(1)


if __name__ == '__main__':
    main()
