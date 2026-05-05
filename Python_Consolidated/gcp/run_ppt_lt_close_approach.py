#!/usr/bin/env python3
"""GCP runner: PPT LT chain — exploit Psyche-Themis close approach (2041-04-25).

Same launch date as the v2 result (2029-07-28), same ordering
(PARTHENOPE → PSYCHE → THEMIS), same architecture (direct, no Mars flyby).

Difference: bias DE to land Themis arrival near the 0.144 AU close approach
on 2041-04-25. Also relax lt_leg_min_yr from 2.0 to 1.0 yr so a short
close-approach transfer is in the search space.
"""
from __future__ import annotations

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

    from core import load_kernels, get_id_from_asteroid_name, get_state, YEAR, DAY
    from lt_chain_optimization import (LTChainConfig, score_paths_lt_chain,
                                          _bounds_no_flyby, _unpack_no_flyby,
                                          compute_path_lt_chain_surrogate,
                                          verify_lt_chain_full)
    from scipy.optimize import differential_evolution

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids', flush=True)

    # ----------------------------------------------------------------------
    # Fixed launch — match v2 winner exactly
    # ----------------------------------------------------------------------
    LAUNCH_ET = spiceypy.str2et('Jul 28 21:06:34 UTC 2029')
    launch_range = [LAUNCH_ET, LAUNCH_ET + DAY]
    print(f'Launch FIXED at {spiceypy.et2utc(LAUNCH_ET, "C", 0)}', flush=True)

    # ----------------------------------------------------------------------
    # Psyche-Themis close approach target
    # ----------------------------------------------------------------------
    TARGET_THEMIS_ARR = spiceypy.str2et('Apr 25 15:09:19 UTC 2041')
    ARR_WINDOW_YR = 1.5
    print(f'Target Themis arrival: {spiceypy.et2utc(TARGET_THEMIS_ARR, "C", 0)}',
          flush=True)
    print(f'  (Psyche-Themis close approach: 0.144 AU sep, 4.21 km/s v_rel)',
          flush=True)
    print(f'Soft window: ±{ARR_WINDOW_YR} yr', flush=True)

    # ----------------------------------------------------------------------
    # Locked ordering & architecture (PPT direct = v2 winner)
    # ----------------------------------------------------------------------
    ordering = ('PARTHENOPE', 'PSYCHE', 'THEMIS')
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n)))
              for n in ordering]
    flyby_name = None  # direct
    print(f'Ordering: {" → ".join(ordering)} (direct)', flush=True)

    # ----------------------------------------------------------------------
    # Config: relax lt_leg_min_yr globally so DE can pick short Psyche→Themis
    # ----------------------------------------------------------------------
    cfg = LTChainConfig(
        spacecraft_launch_mass_kg=3000.0,
        lt_chain_initial_mass_kg=1500.0,
        isp_elec_s=3100.0,
        thrust_N=0.30,
        launch_dv_max_kms=7.0,
        mission_max_yr=30.0,
        stay_min_months=3.0,
        lt_leg_min_yr=1.0,                          # relaxed: SF solver handles 1-yr legs
        lt_nseg=15,
        de_maxiter=600,                             # was 400
        de_popsize=30,                              # was 22
        de_seeds=(13, 42, 91, 137, 314, 808, 2024, 7777, 12345, 99),
        de_m_revs=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)),
    )
    print('\nConfig:', flush=True)
    for k, v in cfg.__dict__.items():
        print(f'  {k:30s}: {v}', flush=True)

    bounds = _bounds_no_flyby(launch_range, cfg)
    print(f'\nDecision-vector bounds (years):', flush=True)
    labels = ['launch_offset', 'tof_E→A1', 'stay_A1', 'tof_A1→A2', 'stay_A2', 'tof_A2→A3']
    for label, (lo, hi) in zip(labels, bounds):
        print(f'  {label:18s}: [{lo:.3f}, {hi:.3f}]', flush=True)

    # ----------------------------------------------------------------------
    # Custom objective: original score + soft penalty on Themis arrival
    # outside the close-approach window
    # ----------------------------------------------------------------------
    def score_close_approach(x, m_revs):
        base = score_paths_lt_chain(x, a_ids[0], a_ids[1], a_ids[2],
                                     launch_range, flyby_name, m_revs, cfg)
        ets = _unpack_no_flyby(x, launch_range)
        et_a3 = ets[-1]
        delta_yr = abs(et_a3 - TARGET_THEMIS_ARR) / YEAR
        if delta_yr > ARR_WINDOW_YR:
            base = base + 100 * (delta_yr - ARR_WINDOW_YR)
        return base

    # ----------------------------------------------------------------------
    # Build seeded init populations: most biased toward the close-approach
    # window, plus the v2 known-good point as anchor, plus uniform spread.
    # scipy DE expects init values in parameter space (years), not normalized.
    # ----------------------------------------------------------------------
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # v2 known-good point (Earth→Parthenope=3.75, stay=0.53, →Psyche=6.91,
    # stay=1.0, →Themis=4.6) — anchors a feasible region.
    v2_anchor = np.array([0.0, 3.75, 0.53, 6.91, 1.0, 4.60])

    def biased_init(rng, popsize, dim):
        """init for DE: 60% biased toward close approach, 30% uniform, 10% v2 anchor."""
        N = popsize
        pop_yr = np.zeros((N, dim))
        target_yr_after_launch = (TARGET_THEMIS_ARR - LAUNCH_ET) / YEAR  # ≈ 11.74

        n_anchor = max(2, N // 10)              # ~10% v2 anchor (jittered)
        n_bias   = (N - n_anchor) * 6 // 10     # ~60% close-approach biased
        n_unif   = N - n_anchor - n_bias        # remainder uniform random

        # 1) anchor near v2 winner with small jitter
        for i in range(n_anchor):
            jitter = rng.normal(0, 0.10, size=dim) * (hi - lo)
            cand = np.clip(v2_anchor + jitter, lo, hi)
            pop_yr[i] = cand

        # 2) close-approach biased: pick TOF for last leg ≥ 2 yr (constraint),
        # but bias toward short end so arrival lands near close approach.
        for i in range(n_anchor, n_anchor + n_bias):
            tof_a2a3 = rng.uniform(2.0, 3.5)
            stay_a2  = rng.uniform(cfg.stay_min_months/12, 1.0)
            stay_a1  = rng.uniform(cfg.stay_min_months/12, 1.0)
            remaining = target_yr_after_launch - tof_a2a3 - stay_a2 - stay_a1
            # split remaining between tof_E→A1 and tof_A1→A2
            f = rng.uniform(0.28, 0.55)
            tof_e_a1  = remaining * f
            tof_a1_a2 = remaining * (1 - f)
            cand = np.array([
                rng.uniform(0, 0.001),
                tof_e_a1, stay_a1,
                tof_a1_a2, stay_a2,
                tof_a2a3])
            pop_yr[i] = np.clip(cand, lo, hi)

        # 3) uniform random fill
        for i in range(n_anchor + n_bias, N):
            cand = lo + rng.uniform(0, 1, size=dim) * (hi - lo)
            pop_yr[i] = cand

        return pop_yr

    # ----------------------------------------------------------------------
    # Run DE for each m_revs combo and seed
    # ----------------------------------------------------------------------
    print(f'\n=== Running DE ({len(cfg.de_m_revs)} m_revs × '
          f'{len(cfg.de_seeds)} seeds = '
          f'{len(cfg.de_m_revs)*len(cfg.de_seeds)} runs) ===\n', flush=True)

    best = None
    n_runs = 0
    t_start = time.time()

    for m_revs in cfg.de_m_revs:
        for seed in cfg.de_seeds:
            n_runs += 1
            rng = np.random.default_rng(seed)
            init_pop = biased_init(rng, cfg.de_popsize * len(bounds), len(bounds))

            try:
                t0 = time.time()
                res = differential_evolution(
                    lambda x: score_close_approach(x, m_revs),
                    bounds, maxiter=cfg.de_maxiter, tol=1e-7, seed=seed,
                    polish=True, popsize=cfg.de_popsize,
                    mutation=(0.5, 1.3), recombination=0.8,
                    init=init_pop, updating='deferred')
            except Exception as e:
                print(f'  m={m_revs} seed={seed:5d}  ERROR: {e}', flush=True)
                continue

            ets = _unpack_no_flyby(res.x, launch_range)
            et_launch, et_a1, et_s1, et_a2, et_s2, et_a3 = ets
            full = compute_path_lt_chain_surrogate(
                a_ids[0], a_ids[1], a_ids[2],
                et_launch, et_a1, et_s1, et_a2, et_s2, et_a3,
                flyby_name, None, m_revs, cfg)

            elapsed = time.time() - t0
            if not full['feasible']:
                reason = full.get('reason', '?')
                print(f'  m={m_revs} seed={seed:5d}  infeasible ({reason})  '
                      f'score={res.fun:.2f}  ({elapsed:.0f}s)', flush=True)
                continue

            arr_str = spiceypy.et2utc(et_a3, "C", 0)
            tof_a2a3_yr = (et_a3 - et_s2) / YEAR
            print(f'  m={m_revs} seed={seed:5d}  '
                  f'launch_dv={full["launch_dv_kms"]:.2f}  '
                  f'post_dv={full["post_launch_dv_kms"]:.3f}  '
                  f'arr={arr_str}  '
                  f'tof_A2A3={tof_a2a3_yr:.2f}yr  '
                  f'({elapsed:.0f}s)', flush=True)

            if best is None or full['post_launch_dv_kms'] < best['post_launch_dv_kms']:
                full['_seed'] = seed
                best = full

    print(f'\nTotal DE time: {time.time()-t_start:.0f}s ({n_runs} runs)', flush=True)

    if best is None:
        print('\nNo feasible solution found.', flush=True)
        return

    # ----------------------------------------------------------------------
    # Verify with full Sims-Flanagan
    # ----------------------------------------------------------------------
    print(f'\n{"="*78}', flush=True)
    print(f' SURROGATE BEST', flush=True)
    print(f'{"="*78}', flush=True)
    print(f'  Ordering         : {" → ".join(ordering)}', flush=True)
    print(f'  Architecture     : direct', flush=True)
    print(f'  m_revs           : {best["m_revs"]}', flush=True)
    print(f'  Launch dv (excl) : {best["launch_dv_kms"]:.4f} km/s', flush=True)
    print(f'  Post-launch dv   : {best["post_launch_dv_kms"]:.4f} km/s', flush=True)
    print(f'  Final mass (sur.): {best["m_final_kg"]:.1f} kg', flush=True)

    eps = best['epochs']
    for label, key in [('Earth launch', 'et_launch'),
                         ('Arrive PARTHENOPE', 'et_a1_arr'),
                         ('Depart PARTHENOPE', 'et_a1_dep'),
                         ('Arrive PSYCHE', 'et_a2_arr'),
                         ('Depart PSYCHE', 'et_a2_dep'),
                         ('Arrive THEMIS', 'et_a3_arr')]:
        et = eps.get(key)
        if et is None: continue
        print(f'    {label:20s}: {spiceypy.et2utc(et, "C", 0)}', flush=True)

    print(f'\nRunning full Sims-Flanagan verification...', flush=True)
    t0 = time.time()
    verified = verify_lt_chain_full(best, *a_ids, cfg=cfg, verbose=True)
    print(f'Verification took {time.time()-t0:.1f}s', flush=True)

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------
    print(f'\n{"="*78}', flush=True)
    print(f' VERIFIED RESULT (close-approach exploit)', flush=True)
    print(f'{"="*78}', flush=True)
    print(f'  Ordering             : {" → ".join(ordering)}', flush=True)
    print(f'  Architecture         : direct (no Mars flyby)', flush=True)
    print(f'  Launch dv (excl obj) : {verified["launch_dv_kms"]:.4f} km/s', flush=True)
    print(f'  Post-launch dv (LT)  : {verified["post_launch_dv_kms_full"]:.4f} km/s',
          flush=True)
    print(f'  Final mass           : {verified["m_final_kg_full"]:.1f} kg '
          f'of {cfg.lt_chain_initial_mass_kg:.0f}', flush=True)
    print(f'  All legs converged   : {verified["feasibility"]["all_legs_converged"]}',
          flush=True)

    eps = verified['epochs']
    print('\n  Mission timeline:', flush=True)
    for label, key in [('Earth launch', 'et_launch'),
                         ('Arrive PARTHENOPE', 'et_a1_arr'),
                         ('Depart PARTHENOPE', 'et_a1_dep'),
                         ('Arrive PSYCHE', 'et_a2_arr'),
                         ('Depart PSYCHE', 'et_a2_dep'),
                         ('Arrive THEMIS', 'et_a3_arr')]:
        et = eps.get(key)
        if et is None: continue
        print(f'    {label:20s}: {spiceypy.et2utc(et, "C", 0)}', flush=True)
    dur = (eps['et_a3_arr'] - eps['et_launch']) / YEAR
    print(f'    Total duration       : {dur:.2f} yr', flush=True)

    # Compute Psyche-Themis separation at Themis arrival
    rp, _ = get_state(a_ids[1], eps['et_a3_arr'])
    rt, _ = get_state(a_ids[2], eps['et_a3_arr'])
    sep_au = float(np.linalg.norm(np.array(rp) - np.array(rt))) / 1.496e8
    print(f'    Psyche-Themis sep at Themis arrival: {sep_au:.3f} AU', flush=True)
    days_off = (eps['et_a3_arr'] - TARGET_THEMIS_ARR) / DAY
    print(f'    Themis arrival vs close approach: {days_off:+.0f} days', flush=True)

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

    # ----------------------------------------------------------------------
    # Compare with v2 baseline
    # ----------------------------------------------------------------------
    print(f'\n{"="*78}', flush=True)
    print(f' COMPARISON vs v2 (saved baseline)', flush=True)
    print(f'{"="*78}', flush=True)
    v2_path = 'optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl'
    if os.path.exists(v2_path):
        with open(v2_path, 'rb') as f:
            v2 = pickle.load(f)
        v2_post = v2['verified']['post_launch_dv_kms_full']
        v2_dur = (v2['verified']['epochs']['et_a3_arr']
                   - v2['verified']['epochs']['et_launch']) / YEAR
        new_post = verified['post_launch_dv_kms_full']
        new_dur = dur
        print(f'  {"":25s}  {"v2 baseline":>15s}  {"close-appr":>15s}  {"delta":>10s}',
              flush=True)
        print(f'  {"post-launch dv (km/s)":25s}  {v2_post:15.3f}  {new_post:15.3f}  '
              f'{new_post-v2_post:+10.3f}', flush=True)
        print(f'  {"mission duration (yr)":25s}  {v2_dur:15.2f}  {new_dur:15.2f}  '
              f'{new_dur-v2_dur:+10.2f}', flush=True)
    else:
        print(f'  v2 baseline not found at {v2_path}', flush=True)

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    out_path = os.path.join(project_root,
        'optimal_asteroid_paths/pkl/ppt_lt_chain_close_approach.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'experiment': 'PPT LT chain — Psyche-Themis close approach exploit',
            'ordering': list(ordering),
            'flyby': flyby_name,
            'launch_et_fixed': float(LAUNCH_ET),
            'target_themis_arr_et': float(TARGET_THEMIS_ARR),
            'arr_window_yr': ARR_WINDOW_YR,
            'config': cfg.__dict__,
            'surrogate': best,
            'verified': verified,
        }, f)
    print(f'\nSaved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
