#!/usr/bin/env python3
"""Joint mass + Δv optimization across top-50 triplets, all 8 architectures.

For each triplet from results_69ast_ega_real.pkl:
  1. Run mass-objective DE for each of 8 propulsion architectures
     (CCC, CCE, CEC, CEE, ECC, ECE, EEC, EEE) — re-optimizing timing per arch.
  2. Multi-seed (3 seeds) + multi-Lambert-revolution sweep (4 m-revs combos).
  3. Pick the architecture with the highest delivered mass.
  4. Verify the top-3 winners per triplet with the real LT solver.
  5. Save results and print ranked summary tables.

Designed to run on a 12-vCPU GCP VM. Wall time estimate: ~30 min, ~$0.20.
"""
import sys
import os
import time
import pickle
import warnings
import multiprocessing as mp

import numpy as np

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


# =============================================================================
# WORKER INIT (each pool worker re-loads SPICE kernels)
# =============================================================================

def _init_worker():
    import spiceypy, glob
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    gk = os.path.join(project_root, 'generic_kernels')
    spiceypy.furnsh(os.path.join(gk, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'planets',    'de430.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'pck00010.tpc'))
    for bsp in sorted(glob.glob(
            os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs', '*.bsp'))):
        spiceypy.furnsh(bsp)


# =============================================================================
# PER-TRIPLET TASK
# =============================================================================

def _eval_triplet(args):
    """Worker: optimize one triplet across all 8 architectures, with verification."""
    rank, i, j, k, ids, names, launch_range, flyby_name, m_init_kg, thrust_N = args
    from mass_optimization import (pareto_optimize_triplet, verify_with_full_lt,
                                    ARCH_CODES)

    t0 = time.time()
    try:
        results = pareto_optimize_triplet(
            *ids, launch_range, flyby_name=flyby_name,
            archs=ARCH_CODES,
            m_revs_options=[(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)],
            seeds=(42, 137, 314),
            maxiter=200, popsize=15,
            m_init_kg=m_init_kg, thrust_N=thrust_N)

        if not results:
            return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names,
                    'error': 'all archs infeasible',
                    'elapsed_s': time.time() - t0}

        # Sort by m_final (descending) and verify top 3
        sorted_archs = sorted(results.items(),
                              key=lambda kv: -kv[1]['m_final_kg'])
        verified = []
        for arch, surrogate_result in sorted_archs[:3]:
            try:
                v = verify_with_full_lt(surrogate_result, *ids,
                                         m_init_kg=m_init_kg, thrust_N=thrust_N)
                verified.append(v)
            except Exception as e:
                verified.append({**surrogate_result,
                                  'verified': False, 'verify_error': str(e)})

        # Pick the best verified result
        best_verified = None
        for v in verified:
            if v.get('verified_feasible', False):
                m = v.get('verified_m_final_kg', 0.0)
                if best_verified is None or m > best_verified.get('verified_m_final_kg', 0.0):
                    best_verified = v
        if best_verified is None:
            best_verified = verified[0]  # fall back to the surrogate winner

        return {
            'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names,
            'all_archs': results,
            'verified_top3': verified,
            'best_verified': best_verified,
            'elapsed_s': time.time() - t0,
        }
    except Exception as e:
        import traceback
        return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names,
                'error': str(e), 'traceback': traceback.format_exc(),
                'elapsed_s': time.time() - t0}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    pkl_dir = os.path.join(project_root, 'optimal_asteroid_paths', 'pkl')

    # Triplet pool: top-50 from the EGA-real run (the 9.40 km/s family)
    SOURCE_PKL = os.path.join(pkl_dir, 'results_69ast_ega_real.pkl')
    if not os.path.exists(SOURCE_PKL):
        SOURCE_PKL = os.path.join(pkl_dir, 'results_69ast_ga.pkl')
    print(f"Loading triplet pool from: {SOURCE_PKL}", flush=True)
    with open(SOURCE_PKL, 'rb') as f:
        seed_results = pickle.load(f)
    print(f"Loaded {len(seed_results)} triplets", flush=True)

    TOP_N = 50
    seed_results = seed_results[:TOP_N]

    # Asteroid registry
    from core import load_kernels
    import spiceypy
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))

    # Launch window matches the rest of the pipeline
    et_min = spiceypy.str2et('Jan 1 12:00:00 UTC 2027')
    et_max = spiceypy.str2et('Dec 31 12:00:00 UTC 2035')
    launch_range = [et_min, et_max]

    # Spacecraft params (Dawn-like)
    M_INIT_KG = 1500.0
    THRUST_N  = 0.30
    FLYBY     = 'mars'   # the EGA-real seed pool uses Mars GA

    # Build task list
    tasks = []
    for rank, entry in enumerate(seed_results, 1):
        if isinstance(entry, tuple) and len(entry) == 4:
            i, j, k, _res = entry
        else:
            # some pickles store list-of-dicts
            i = entry.get('i'); j = entry.get('j'); k = entry.get('k')
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        ids   = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        tasks.append((rank, i, j, k, ids, names, launch_range, FLYBY,
                      M_INIT_KG, THRUST_N))

    N_WORKERS = min(mp.cpu_count(), len(tasks))
    print(f"\nMass + Δv joint optimization", flush=True)
    print(f"  Source pool   : {SOURCE_PKL}", flush=True)
    print(f"  Triplets      : {len(tasks)}", flush=True)
    print(f"  Architectures : 8 per triplet", flush=True)
    print(f"  m-revs combos : 4 per arch", flush=True)
    print(f"  DE seeds      : 3 per arch (multi-start)", flush=True)
    print(f"  DE budget     : maxiter=200, popsize=15", flush=True)
    print(f"  Verification  : full Sims-Flanagan on top-3 archs per triplet", flush=True)
    print(f"  Workers       : {N_WORKERS}", flush=True)
    print(f"  Spacecraft    : {M_INIT_KG:.0f} kg, {THRUST_N} N, {FLYBY} flyby", flush=True)
    print(f"  Window        : 2027-01-01 to 2035-12-31", flush=True)
    print(flush=True)
    print("-" * 110, flush=True)
    print(f"{'#':>3}  {'triplet':35s}  {'arch':4s}  "
          f"{'surr_m':>7s}  {'ver_m':>7s}  {'dv_eq':>6s}  t(s)", flush=True)
    print("-" * 110, flush=True)

    t_start = time.time()
    all_results = []
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for out in pool.imap_unordered(_eval_triplet, tasks, chunksize=1):
            all_results.append(out)
            if 'error' in out:
                print(f"  #{out['rank']:<3d} ERROR: {out['error'][:70]}",
                      flush=True)
                continue
            n = out['names']
            tag = f"{n[0][:11]}->{n[1][:11]}->{n[2][:11]}"
            best_surr = max(out['all_archs'].values(),
                            key=lambda r: r['m_final_kg'])
            ver = out.get('best_verified', {})
            ver_m = ver.get('verified_m_final_kg', 0.0) if ver.get('verified_feasible') else 0.0
            ver_dv = ver.get('verified_dv_equiv_kms', 1e3) if ver.get('verified_feasible') else 1e3
            print(f"  #{out['rank']:<3d} {tag:35s}  {best_surr['arch_code']:4s}  "
                  f"{best_surr['m_final_kg']:7.1f}  {ver_m:7.1f}  {ver_dv:6.2f}  "
                  f"{out['elapsed_s']:4.0f}", flush=True)

    t_total = time.time() - t_start
    print("-" * 110, flush=True)
    print(f"\nDone in {t_total:.0f}s ({t_total/60:.1f} min)", flush=True)

    valid = [r for r in all_results if 'error' not in r and r.get('best_verified')]

    # ---- Ranking by verified delivered mass ----
    print("\n" + "=" * 110, flush=True)
    print(" TOP 15 BY VERIFIED DELIVERED MASS (LT-refined)", flush=True)
    print("=" * 110, flush=True)
    print(f"{'rank':>4s}  {'triplet':37s}  {'arch':4s}  {'m_kg':>8s}  "
          f"{'dv_eq':>6s}  {'src_rank':>8s}", flush=True)
    valid_v = [r for r in valid
               if r['best_verified'].get('verified_feasible', False)]
    valid_v.sort(key=lambda r: -r['best_verified']['verified_m_final_kg'])
    for rank, r in enumerate(valid_v[:15], 1):
        v = r['best_verified']
        n = r['names']
        tag = f"{n[0][:11]}->{n[1][:11]}->{n[2][:11]}"
        print(f"  {rank:2d}  {tag:37s}  {v['arch_code']:4s}  "
              f"{v['verified_m_final_kg']:8.1f}  "
              f"{v['verified_dv_equiv_kms']:6.2f}  "
              f"{r['rank']:8d}", flush=True)

    # ---- Ranking by lowest verified dv_equiv (== highest mass) ----
    print("\n" + "=" * 110, flush=True)
    print(" TOP 15 BY VERIFIED Δv-EQUIVALENT (lower = more delivered mass)", flush=True)
    print("=" * 110, flush=True)
    valid_v2 = sorted(valid_v,
                      key=lambda r: r['best_verified']['verified_dv_equiv_kms'])
    print(f"{'rank':>4s}  {'triplet':37s}  {'arch':4s}  {'dv_eq':>6s}  "
          f"{'m_kg':>8s}", flush=True)
    for rank, r in enumerate(valid_v2[:15], 1):
        v = r['best_verified']
        n = r['names']
        tag = f"{n[0][:11]}->{n[1][:11]}->{n[2][:11]}"
        print(f"  {rank:2d}  {tag:37s}  {v['arch_code']:4s}  "
              f"{v['verified_dv_equiv_kms']:6.2f}  "
              f"{v['verified_m_final_kg']:8.1f}", flush=True)

    # ---- Save full results ----
    ts = int(time.time())
    out_pkl = os.path.join(project_root, 'optimal_asteroid_paths',
                            'pkl', f'results_mass_pareto_{ts}.pkl')
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'source_pkl': SOURCE_PKL,
            'm_init_kg': M_INIT_KG, 'thrust_N': THRUST_N, 'flyby': FLYBY,
            'launch_range_utc': ('2027-01-01', '2035-12-31'),
            'all_results': all_results,
        }, f)
    print(f"\nSaved: {out_pkl}", flush=True)
