"""Constraint-compliance audit for ppt_lt_chain_close_approach.pkl.

Runs every project-level constraint check from CONSTRAINTS_AND_OUTPUTS.md.
Independent of the SF audit (`verify_ppt_lt_chain_full.py`) — focuses on
project / mission-level limits rather than physics re-derivation.

Usage:
    python check_constraints.py [path/to/ppt_lt_chain_close_approach.pkl]
"""
import os
import pickle
import sys
import numpy as np
import spiceypy

YEAR  = 365.25 * 86400
DAY   = 86400
MONTH = 30.4375 * DAY


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_pkl = os.path.join(here, '..', 'optimization',
                                 'ppt_lt_chain_close_approach.pkl')
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else default_pkl
    print(f'Loading: {pkl_path}')

    # Need leapseconds for any UTC formatting later (not strictly required for the checks).
    for guess in ['/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls',
                   os.path.join(here, '..', '..', 'generic_kernels', 'lsk', 'naif0012.tls')]:
        if os.path.exists(guess):
            spiceypy.furnsh(guess); break

    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    v = d['verified']; eps = v['epochs']; cfg = d['config']

    checks = []
    def chk(name, ok, detail=''):
        checks.append((name, ok, detail))

    # ---- Mission shape ----
    chk('Direct architecture (no flyby)', d.get('flyby') is None,
         f'flyby={d.get("flyby")}')
    chk('Triplet = PARTHENOPE/PSYCHE/THEMIS',
         set(d.get('ordering', d.get('best_ordering', []))) ==
         {'PARTHENOPE','PSYCHE','THEMIS'},
         str(d.get('ordering')))

    # ---- Launch dv ----
    ld = v['launch_dv_kms']
    chk('Launch dv ≤ 7.0 km/s (impulsive)', ld <= 7.001, f'{ld:.4f} km/s')

    # ---- Spacecraft mass ----
    chk('Launch mass = 3000 kg', cfg['spacecraft_launch_mass_kg'] == 3000.0)
    chk('LT chain initial mass = 1500 kg', cfg['lt_chain_initial_mass_kg'] == 1500.0)
    chk('Final mass > 0', v['m_final_kg_full'] > 0, f'{v["m_final_kg_full"]:.1f} kg')

    # ---- Engine ----
    chk('Isp = 3100 s',          cfg['isp_elec_s'] == 3100.0)
    chk('Thrust cap = 0.30 N',   cfg['thrust_N']   == 0.30)
    chk('No flyby physics needed', v.get('flyby_name') is None)

    # ---- Mission duration ----
    dur = (eps['et_a3_arr'] - eps['et_launch']) / YEAR
    chk('Mission duration ≤ 30 yr', dur <= 30.001, f'{dur:.2f} yr')

    # ---- Stays ----
    stay1_mo = (eps['et_a1_dep'] - eps['et_a1_arr']) / MONTH
    stay2_mo = (eps['et_a2_dep'] - eps['et_a2_arr']) / MONTH
    chk('Stay PARTHENOPE in [3 mo, 12 mo]', 3.0-1e-3 <= stay1_mo <= 12.0+1e-3, f'{stay1_mo:.2f} mo')
    chk('Stay PSYCHE in [3 mo, 12 mo]',     3.0-1e-3 <= stay2_mo <= 12.0+1e-3, f'{stay2_mo:.2f} mo')

    # ---- LT leg TOF (relaxed: ≥ 1 yr per the project decision) ----
    for L in v['verified_legs']:
        chk(f'{L["label"]:18s} TOF ≥ 1 yr (engine OK in practice)',
             L['tof_yr'] >= 0.999, f'{L["tof_yr"]:.2f} yr')
        chk(f'{L["label"]:18s} TOF ≤ 8 yr', L['tof_yr'] <= 8.001,
             f'{L["tof_yr"]:.2f} yr')

    # ---- Convergence ----
    for L in v['verified_legs']:
        chk(f'{L["label"]:18s} pos_err < 1.5e6 km', L['pos_err_km'] < 1.5e6,
             f'{L["pos_err_km"]:.1e} km')
        chk(f'{L["label"]:18s} vel_err < 0.15 km/s', L['vel_err_kms'] < 0.15,
             f'{L["vel_err_kms"]:.2e} km/s')
        chk(f'{L["label"]:18s} converged flag', L['converged'])

    # ---- Throttle bounds ----
    for L in v['verified_legs']:
        u = np.array(L['thrust_profile']['throttle_unit_vector'])
        max_u = float(np.linalg.norm(u, axis=1).max())
        chk(f'{L["label"]:18s} max |u| ≤ 1 (engine cap)', max_u <= 1.001,
             f'max |u| = {max_u:.4f}')

    # ---- Thrust magnitude ≤ 0.30 N ----
    for L in v['verified_legs']:
        tmax = max(L['thrust_profile']['thrust_magnitude_N'])
        chk(f'{L["label"]:18s} thrust ≤ 0.30 N', tmax <= 0.300001,
             f'peak {tmax*1000:.1f} mN')

    # ---- Mass strictly decreases ----
    masses = [L['m_in_kg'] for L in v['verified_legs']] + [v['verified_legs'][-1]['m_out_kg']]
    chk('Mass strictly decreases through chain',
         all(masses[i+1] < masses[i] for i in range(len(masses)-1)),
         'chain = ' + ' > '.join(f'{m:.1f}' for m in masses))

    # ---- Σ leg dv == post_launch_dv ----
    sum_dv = sum(L['dv_integral_kms'] for L in v['verified_legs'])
    chk('Σ leg dv == post_launch_dv_kms_full',
         abs(sum_dv - v['post_launch_dv_kms_full']) < 1e-3,
         f'sum={sum_dv:.4f}  saved={v["post_launch_dv_kms_full"]:.4f}')

    # ---- All legs converged flag ----
    chk("verified['feasibility']['all_legs_converged'] is True",
         v['feasibility']['all_legs_converged'])

    # ---- Print + summary ----
    print('=' * 70)
    print('CONSTRAINT-COMPLIANCE AUDIT — close-approach trajectory')
    print('=' * 70)
    n_pass = n_fail = 0
    for name, ok, detail in checks:
        flag = '\033[32m  OK\033[0m' if ok else '\033[31mFAIL\033[0m'
        if ok: n_pass += 1
        else:  n_fail += 1
        line = f'  [{flag}]  {name}'
        if detail: line += f'   ─  {detail}'
        print(line)

    print()
    print(f'OVERALL: {n_pass}/{n_pass+n_fail} pass, {n_fail} fail')
    if n_fail == 0:
        print('✓ All project-level constraints satisfied.')
        return 0
    else:
        print('✗ Constraint violations detected.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
