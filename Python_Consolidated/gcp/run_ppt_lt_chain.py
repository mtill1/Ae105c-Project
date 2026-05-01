#!/usr/bin/env python3
"""GCP runner: PARTHENOPE → PSYCHE → THEMIS, LT-after-launch chain.

Tries ALL 6 orderings of {PARTHENOPE, PSYCHE, THEMIS} and picks the one
with the lowest post-launch Δv. Each ordering is optimized with Mars flyby
AND direct (no flyby) architectures.

Constraint setup (see MISSION_PARTHENOPE_PSYCHE_THEMIS.md §0):
  • Launch: impulsive ≤ 7 km/s, EXCLUDED from objective
  • Post-launch: electric-only Sims-Flanagan, Isp 3100 s, thrust ≤ 0.30 N
  • Mars flyby: ballistic only
  • LT leg min TOF: 2 years (prevents engine saturation)
  • Mission cap: 30 years (bounded above by BSP coverage)
  • Stays: ≥ 3 months
  • Spacecraft: 3000 kg launch, LT chain m_init = 1500 kg

Output: per-leg integrated Δv, throttle-vs-time profiles (15 seg × 3 comp
per leg), Mars flyby v_∞ vectors / turn / altitude, full date/Δv breakdown.
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import time

import numpy as np
import spiceypy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


def main():
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    os.chdir(project_root)

    from core import load_kernels, YEAR
    from lt_chain_optimization import LTChainConfig, run_triplet

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids', flush=True)

    et_min = spiceypy.str2et('Jan 1 12:00:00 UTC 2027')
    et_max = spiceypy.str2et('Dec 31 12:00:00 UTC 2035')
    launch_range = [et_min, et_max]

    cfg = LTChainConfig(
        spacecraft_launch_mass_kg=3000.0,
        lt_chain_initial_mass_kg=1500.0,
        isp_elec_s=3100.0,
        thrust_N=0.30,
        launch_dv_max_kms=7.0,
        mission_max_yr=30.0,
        stay_min_months=3.0,
        lt_leg_min_yr=2.0,
        lt_nseg=15,
        de_maxiter=400, de_popsize=22,
        de_seeds=(13, 42, 91, 137, 314, 808),
        de_m_revs=((0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0), (1,1,0,0)),
    )

    print('\nLT-chain config:', flush=True)
    for k, v in cfg.__dict__.items():
        print(f'  {k:30s}: {v}', flush=True)
    print(f'  launch window: 2027-01-01 → 2035-12-31', flush=True)

    # All 6 orderings of {PARTHENOPE, PSYCHE, THEMIS}
    base_triplet = ('PARTHENOPE', 'PSYCHE', 'THEMIS')
    orderings = list(itertools.permutations(base_triplet))
    flyby_options = ['mars', None]   # mars flyby and direct

    print(f'\n=== Trying {len(orderings)} orderings × '
          f'{len(flyby_options)} architectures = '
          f'{len(orderings)*len(flyby_options)} combinations ===\n', flush=True)

    all_results = []
    best_overall = None

    for ordering in orderings:
        for flyby in flyby_options:
            label = f'{" → ".join(ordering)} ({"Mars" if flyby else "direct"})'
            print(f'--- {label} ---', flush=True)
            t0 = time.time()

            result = run_triplet(ordering, asteroid_list, launch_range,
                                  flyby_name=flyby, cfg=cfg,
                                  verify=False, verbose=True)

            elapsed = time.time() - t0
            surr = result.get('surrogate')

            if surr is None:
                print(f'    No feasible solution ({elapsed:.0f}s)', flush=True)
                all_results.append({
                    'ordering': list(ordering), 'flyby': flyby,
                    'feasible': False, 'elapsed_s': elapsed})
                continue

            print(f'    launch_dv={surr["launch_dv_kms"]:.2f}  '
                  f'post_launch_dv={surr["post_launch_dv_kms"]:.2f}  '
                  f'm_final={surr["m_final_kg"]:.0f} kg  ({elapsed:.0f}s)',
                  flush=True)

            entry = {
                'ordering': list(ordering), 'flyby': flyby,
                'feasible': True, 'elapsed_s': elapsed,
                'surrogate': surr,
            }
            all_results.append(entry)

            if (best_overall is None
                    or surr['post_launch_dv_kms'] < best_overall['surrogate']['post_launch_dv_kms']):
                best_overall = entry

    # ======================================================================
    # Verify best with full Sims-Flanagan
    # ======================================================================
    if best_overall is None:
        print('\nNo feasible solution found across all orderings.', flush=True)
        return

    winner_ordering = tuple(best_overall['ordering'])
    winner_flyby    = best_overall['flyby']
    print(f'\n{"="*78}', flush=True)
    print(f' BEST ORDERING: {" → ".join(winner_ordering)} '
          f'({"Mars" if winner_flyby else "direct"})', flush=True)
    print(f' Surrogate post-launch dv: '
          f'{best_overall["surrogate"]["post_launch_dv_kms"]:.3f} km/s', flush=True)
    print(f'{"="*78}', flush=True)

    print(f'\nRunning full Sims-Flanagan verification on winner...', flush=True)
    t0 = time.time()
    from core import get_id_from_asteroid_name
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n)))
              for n in winner_ordering]
    from lt_chain_optimization import verify_lt_chain_full
    verified = verify_lt_chain_full(best_overall['surrogate'], *a_ids, cfg=cfg,
                                      verbose=True)
    print(f'Verification took {time.time()-t0:.1f}s', flush=True)

    # ======================================================================
    # Report
    # ======================================================================
    eps = verified['epochs']
    print(f'\n{"="*78}', flush=True)
    print(f' VERIFIED RESULT', flush=True)
    print(f'{"="*78}', flush=True)
    print(f'  Ordering               : {" → ".join(winner_ordering)}', flush=True)
    print(f'  Architecture           : {"Mars ballistic GA" if winner_flyby else "direct"}',
          flush=True)
    print(f'  Launch dv (excl obj)   : {verified["launch_dv_kms"]:.4f} km/s', flush=True)
    print(f'  Post-launch dv (LT)    : {verified["post_launch_dv_kms_full"]:.4f} km/s',
          flush=True)
    print(f'  Final mass             : {verified["m_final_kg_full"]:.1f} kg '
          f'of {cfg.lt_chain_initial_mass_kg:.0f}', flush=True)
    print(f'  All legs converged     : {verified["feasibility"]["all_legs_converged"]}',
          flush=True)

    print('\n  Mission timeline:', flush=True)
    for label, key in [('Earth launch', 'et_launch'),
                         ('Mars flyby', 'et_flyby'),
                         (f'Arrive {winner_ordering[0]}', 'et_a1_arr'),
                         (f'Depart {winner_ordering[0]}', 'et_a1_dep'),
                         (f'Arrive {winner_ordering[1]}', 'et_a2_arr'),
                         (f'Depart {winner_ordering[1]}', 'et_a2_dep'),
                         (f'Arrive {winner_ordering[2]}', 'et_a3_arr')]:
        et = eps.get(key)
        if et is None: continue
        print(f'    {label:25s}: {spiceypy.et2utc(et, "C", 0)}', flush=True)
    dur = (eps['et_a3_arr'] - eps['et_launch']) / YEAR
    print(f'    Total mission duration   : {dur:.2f} yr', flush=True)

    if 'flyby_audit' in verified:
        fa = verified['flyby_audit']
        print('\n  Mars GA diagnostics:', flush=True)
        print(f'    v_inf_in vector  : '
              f'[{fa["v_inf_in_vec"][0]:+.4f}, {fa["v_inf_in_vec"][1]:+.4f}, '
              f'{fa["v_inf_in_vec"][2]:+.4f}] km/s', flush=True)
        print(f'    v_inf_out vector : '
              f'[{fa["v_inf_out_vec"][0]:+.4f}, {fa["v_inf_out_vec"][1]:+.4f}, '
              f'{fa["v_inf_out_vec"][2]:+.4f}] km/s', flush=True)
        print(f'    |v_inf|          : {fa["v_inf_in_kms"]:.4f} / '
              f'{fa["v_inf_out_kms"]:.4f} km/s   '
              f'({"BALLISTIC" if fa.get("ballistic_ok") else "POWERED"})',
              flush=True)
        print(f'    Turn angle       : {fa["turn_angle_deg"]:.3f}°  '
              f'(max {fa["turn_max_deg"]:.3f}°)', flush=True)
        print(f'    Periapsis alt    : {fa.get("periapsis_alt_km", 0):,.0f} km',
              flush=True)

    print('\n  Low-thrust legs:', flush=True)
    for L in verified['verified_legs']:
        print(f'    {L["label"]:18s}  TOF={L["tof_yr"]:5.2f} yr  '
              f'dv_int={L["dv_integral_kms"]:6.3f} km/s  '
              f'm: {L["m_in_kg"]:6.1f} → {L["m_out_kg"]:6.1f} kg  '
              f'pos_err={L["pos_err_km"]:.1e} km  '
              f'{"OK" if L["converged"] else "FAIL"}', flush=True)

    print('\n  Thrust profile per leg:', flush=True)
    for L in verified['verified_legs']:
        tp = L['thrust_profile']
        mags = np.array(tp['thrust_magnitude_N'])
        print(f'    {L["label"]:18s}  '
              f'mean={1000*mags.mean():6.1f} mN  '
              f'peak={1000*mags.max():6.1f} mN  '
              f'duty={(mags > 0.05*tp["thrust_max_N"]).mean()*100:5.1f}%',
              flush=True)

    # Comparison table for all orderings
    print(f'\n{"="*78}', flush=True)
    print(f' ALL ORDERINGS COMPARISON', flush=True)
    print(f'{"="*78}', flush=True)
    feasible_entries = sorted(
        [r for r in all_results if r['feasible']],
        key=lambda r: r['surrogate']['post_launch_dv_kms'])
    print(f'  {"ordering":45s}  {"arch":>6s}  {"launch":>7s}  '
          f'{"post_dv":>8s}  {"m_kg":>6s}', flush=True)
    for r in feasible_entries:
        s = r['surrogate']
        print(f'  {"→".join(r["ordering"]):45s}  '
              f'{"Mars" if r["flyby"] else "direct":>6s}  '
              f'{s["launch_dv_kms"]:7.2f}  '
              f'{s["post_launch_dv_kms"]:8.2f}  '
              f'{s["m_final_kg"]:6.0f}', flush=True)

    # ======================================================================
    # Save
    # ======================================================================
    out_path = os.path.join(project_root,
        'optimal_asteroid_paths/pkl/ppt_lt_chain.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'triplet_set': sorted(base_triplet),
            'best_ordering': list(winner_ordering),
            'best_flyby': winner_flyby,
            'config': cfg.__dict__,
            'surrogate': best_overall['surrogate'],
            'verified': verified,
            'all_orderings': all_results,
        }, f)
    print(f'\nSaved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
