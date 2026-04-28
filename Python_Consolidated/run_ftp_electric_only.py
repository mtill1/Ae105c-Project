"""Electric-propulsion-only evaluation for FORTUNA -> THEMIS -> PSYCHE.

Takes the saved best Mars-GA impulsive trajectory and re-scores it under
electric-only architectures.  All deep-space maneuvers are reflown with
Sims-Flanagan low-thrust legs at NSTAR Isp; Earth launch v_inf is
provided by the launch vehicle, not by spacecraft propellant.

Compares:
  CCC  : all chemical (baseline)
  CCE  : L2,L3 chemical, L4 electric
  CEC  : L2 chem, L3 elec, L4 chem
  CEE  : L2 chem, L3+L4 elec
  EEE  : L2+L3+L4 all electric  <-- the "electric only" case
"""

import os
import pickle
import numpy as np
import spiceypy

from core import (load_kernels, get_id_from_asteroid_name,
                  get_state, DAY, YEAR)
from lowthrust import (optimize_lt_leg, tsiolkovsky_fraction,
                       ISP_CHEM, ISP_ELEC, G0,
                       DEFAULT_M_INIT_KG, DEFAULT_THRUST_N, DEFAULT_NSEG)


PKL_IN  = 'optimal_asteroid_paths/pkl/fortuna_themis_psyche_mars_robust.pkl'
PKL_OUT = 'optimal_asteroid_paths/pkl/ftp_electric_only.pkl'

TRIPLET = ('FORTUNA', 'THEMIS', 'PSYCHE')

# Spacecraft choices (Dawn-class)
M_INIT_KG  = 1500.0
THRUST_N   = 0.30
NSEG       = 25     # finer grid than default — these are 4-yr legs
NSEG_RETRY = 40


def fmt_et(et):
    return spiceypy.et2utc(et, 'C', 0)


def leg_chemical_dv(res, leg_idx):
    """Δv (km/s) for the chemical-only version of leg L1..L4."""
    if leg_idx == 1:
        # Launch + Mars flyby powered burn (flyby_powered ~0 for free GA)
        return (np.linalg.norm(res['delta_v_launch']) +
                abs(float(res.get('delta_v_flyby', 0.0))))
    if leg_idx == 2:
        return float(np.linalg.norm(res['delta_v_A1_arrive']))
    if leg_idx == 3:
        return (np.linalg.norm(res['delta_v_A1_leave']) +
                np.linalg.norm(res['delta_v_A2_arrive']))
    if leg_idx == 4:
        return (np.linalg.norm(res['delta_v_A2_leave']) +
                np.linalg.norm(res['delta_v_A3_arrive']))
    raise ValueError(leg_idx)


def leg_endpoints(res, a_ids, leg_idx):
    """(et0, et1, body0, body1) for each leg (L2..L4)."""
    if leg_idx == 2:
        return res['et_flyby'],   res['et_arrive_1'], '4',     a_ids[0]
    if leg_idx == 3:
        return res['et_stay_1'],  res['et_arrive_2'], a_ids[0], a_ids[1]
    if leg_idx == 4:
        return res['et_stay_2'],  res['et_arrive_3'], a_ids[1], a_ids[2]
    raise ValueError(leg_idx)


def solve_leg_lt(res, a_ids, leg_idx, m_entering, thrust_N, nseg, verbose=True):
    et0, et1, b0, b1 = leg_endpoints(res, a_ids, leg_idx)
    r0, v0 = get_state(b0, et0)
    r1, v1 = get_state(b1, et1)
    tof_sec = et1 - et0
    out = optimize_lt_leg(r0, v0, r1, v1, tof_sec,
                          m_init_kg=m_entering, thrust_N=thrust_N,
                          isp_s=ISP_ELEC, nseg=nseg, verbose=False)
    if not out['converged']:
        # Retry with finer segmentation
        out2 = optimize_lt_leg(r0, v0, r1, v1, tof_sec,
                               m_init_kg=m_entering, thrust_N=thrust_N,
                               isp_s=ISP_ELEC, nseg=NSEG_RETRY, verbose=False)
        if out2['converged']:
            out = out2
    out['leg_idx'] = leg_idx
    out['tof_yr'] = tof_sec / YEAR
    out['m_entering_kg'] = m_entering
    if verbose:
        flag = 'OK' if out['converged'] else 'NO'
        print(f'    L{leg_idx} LT: TOF={out["tof_yr"]:5.2f}yr  '
              f'm_in={m_entering:7.1f}kg  m_out={out["m_final"]:7.1f}kg  '
              f'dv_int={out["dv_integral_kms"]:5.2f}  '
              f'pos_err={out["pos_err_km"]:.2e}  '
              f'vel_err={out["vel_err_kms"]:.3f}  [{flag}]')
    return out


def run_arch(arch_code, res, a_ids, m_init, thrust_N, nseg, verbose=True):
    """Architecture code is XYZ where each char in {C,E} for legs L2,L3,L4.

    Launch (L1) is always chemical via the launch vehicle.
    """
    print(f'\n--- Architecture {arch_code} ---')
    # L1: launch vehicle delivers the v_inf at Earth.  This is launch-vehicle
    # propellant, not spacecraft propellant -- so the spacecraft mass at start
    # of L2 = m_init (full).  The flyby-powered Δv (chemical) is part of L1.
    dv_flyby_powered = abs(float(res.get('delta_v_flyby', 0.0)))
    m = m_init
    if dv_flyby_powered > 1e-3:
        m *= tsiolkovsky_fraction(dv_flyby_powered, ISP_CHEM)

    leg_results = {1: {'dv': leg_chemical_dv(res, 1),
                       'mode': 'C-launch', 'm_in': m_init, 'm_out': m,
                       'isp': ISP_CHEM}}

    for i, mode in zip([2, 3, 4], arch_code):
        dv_chem = leg_chemical_dv(res, i)
        if mode == 'C':
            m_out = m * tsiolkovsky_fraction(dv_chem, ISP_CHEM)
            leg_results[i] = {'dv': dv_chem, 'mode': 'C', 'isp': ISP_CHEM,
                              'm_in': m, 'm_out': m_out, 'converged': True}
            if verbose:
                print(f'    L{i} C : Δv={dv_chem:5.2f} km/s  '
                      f'm_in={m:7.1f}  m_out={m_out:7.1f} kg')
            m = m_out
        else:
            lt = solve_leg_lt(res, a_ids, i, m, thrust_N, nseg, verbose=verbose)
            if lt['converged']:
                m_out = lt['m_final']
                leg_results[i] = {'dv_int': lt['dv_integral_kms'], 'mode': 'E',
                                  'isp': ISP_ELEC, 'm_in': m, 'm_out': m_out,
                                  'converged': True, 'pos_err_km': lt['pos_err_km'],
                                  'vel_err_kms': lt['vel_err_kms']}
                m = m_out
            else:
                # Fallback: Tsiolkovsky with impulsive Δv at electric Isp.
                # This is optimistic (real LT pays gravity-loss penalty) but
                # gives a usable upper bound on m_final when the SF solver fails.
                m_out = m * tsiolkovsky_fraction(dv_chem, ISP_ELEC)
                leg_results[i] = {'dv': dv_chem, 'mode': 'E-fallback',
                                  'isp': ISP_ELEC, 'm_in': m, 'm_out': m_out,
                                  'converged': False, 'reason': lt['reason']}
                if verbose:
                    print(f'    L{i} E-fallback: SF did not converge, '
                          f'using Tsiolkovsky upper bound: m_out={m_out:.1f} kg')
                m = m_out

    m_final = m
    prop_used = m_init - m_final
    prop_frac = prop_used / m_init
    return {'arch': arch_code, 'm_final_kg': m_final,
            'm_prop_kg': prop_used, 'prop_fraction': prop_frac,
            'leg_results': leg_results}


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs',
                                 '/Users/rebnoob/Documents/ae105/generic_kernels')

    with open(PKL_IN, 'rb') as f:
        saved = pickle.load(f)
    res = saved['best_mars']

    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n))) for n in TRIPLET]

    print(f'Trajectory: {" -> ".join(TRIPLET)} (Mars GA, impulsive)')
    print(f'  Launch       : {fmt_et(res["et_launch"])}')
    print(f'  Mars flyby   : {fmt_et(res["et_flyby"])}')
    print(f'  Arrive A1    : {fmt_et(res["et_arrive_1"])}')
    print(f'  Depart A1    : {fmt_et(res["et_stay_1"])}')
    print(f'  Arrive A2    : {fmt_et(res["et_arrive_2"])}')
    print(f'  Depart A2    : {fmt_et(res["et_stay_2"])}')
    print(f'  Arrive A3    : {fmt_et(res["et_arrive_3"])}')
    print(f'  Total impulsive Δv: {res["delta_v_total"]:.2f} km/s\n')

    print('Per-leg impulsive Δv (km/s):')
    for i in (1, 2, 3, 4):
        et0, et1 = (
            (res['et_launch'], res['et_flyby']) if i == 1 else
            (res['et_flyby'], res['et_arrive_1']) if i == 2 else
            (res['et_stay_1'], res['et_arrive_2']) if i == 3 else
            (res['et_stay_2'], res['et_arrive_3']))
        print(f'  L{i}: Δv = {leg_chemical_dv(res, i):5.2f} km/s   '
              f'TOF = {(et1-et0)/YEAR:5.2f} yr')

    print(f'\nSpacecraft: m_init={M_INIT_KG} kg, thrust={THRUST_N} N, '
          f'Isp_chem={ISP_CHEM} s, Isp_elec={ISP_ELEC} s')

    archs_to_try = ['CCC', 'CCE', 'CEC', 'CEE', 'ECC', 'ECE', 'EEC', 'EEE']
    results = []
    for arch in archs_to_try:
        r = run_arch(arch, res, a_ids, M_INIT_KG, THRUST_N, NSEG, verbose=True)
        results.append(r)

    # Summary
    print('\n' + '=' * 72)
    print('Mass-budget summary  (m_init = 1500 kg, electric Isp = 3100 s)')
    print('=' * 72)
    print(f'{"Arch":<6} {"m_final [kg]":>14} {"m_prop [kg]":>14} '
          f'{"prop frac":>11}    {"all converged?":>16}')
    print('-' * 72)
    for r in sorted(results, key=lambda x: -x['m_final_kg']):
        all_ok = all(lr.get('converged', True) for lr in r['leg_results'].values()
                     if lr['mode'].startswith('E'))
        flag = 'yes' if all_ok else 'fallback used'
        print(f'{r["arch"]:<6} {r["m_final_kg"]:>14.1f} {r["m_prop_kg"]:>14.1f} '
              f'{r["prop_fraction"]:>11.3f}    {flag:>16}')

    # Save
    out_dir = os.path.dirname(PKL_OUT)
    os.makedirs(out_dir, exist_ok=True)
    with open(PKL_OUT, 'wb') as f:
        pickle.dump({'triplet': TRIPLET, 'mars_traj': res,
                     'm_init_kg': M_INIT_KG, 'thrust_N': THRUST_N,
                     'isp_elec_s': ISP_ELEC, 'isp_chem_s': ISP_CHEM,
                     'archs': results}, f)
    print(f'\nSaved -> {PKL_OUT}')


if __name__ == '__main__':
    main()
