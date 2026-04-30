#!/usr/bin/env python3
"""Mars-flyby-only, composition-diverse, physically-feasible Δv optimization.

Differs from main.py --feasible (which tries direct + Moon + Mars + Earth):
this runner restricts every candidate to Mars gravity assist. Useful when the
mission architecture is fixed (Mars launch window dominates other trades) and
you want to know the cleanest feasible Mars-GA path.

Pipeline:
  Stage 1 — coarse pass: every C+S+X/M triplet, Mars-only, sampling-based pre-screen.
  Stage 2 — full DE (300 iter, popsize=18, multi-seed) on the top 50.
  Stage 3 — independent geometric audit of every flyby. Only feasible
            results are reported as final.

Patched core.compute_flyby_dv returns 1000 km/s penalty for any flyby
geometry whose required turn angle exceeds the natural max at the safe
periapsis altitude — so DE steers away from physically impossible solutions.
The post-hoc audit using audit_flyby_geometry confirms.
"""
import sys
import os
import time
import pickle
import warnings
import multiprocessing as mp

import numpy as np
import spiceypy

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


# =============================================================================
# WORKER INIT (each pool worker re-loads SPICE kernels)
# =============================================================================

def _init_worker():
    import glob
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
# PER-TRIPLET TASKS
# =============================================================================

def _eval_coarse(args):
    """Stage 1: quick Mars-only screen on one triplet."""
    i, j, k, ids, names, launch_range = args
    from optimization import optimize_times_flyby_quick
    try:
        dv = optimize_times_flyby_quick(*ids, launch_range, 'mars')
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': float(dv)}
    except Exception:
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': 1e6}


def _eval_fine(args):
    """Stage 2: full DE Mars-only on one triplet, multi-seed + multi-Lambert.

    The patched compute_flyby_dv inside score_paths_flyby returns 1e3 for
    geometrically infeasible flybys, so DE will not converge there.
    """
    rank, i, j, k, ids, names, launch_range = args
    from optimization import (optimize_times_flyby, score_paths_flyby,
                              _unpack_flyby_input, compute_path_with_flyby,
                              FLYBY_BODIES)
    from scipy.optimize import differential_evolution
    from core import DAY, WEEK, MONTH, YEAR

    fb = FLYBY_BODIES['mars']
    bounds = list(zip(
        np.array([0, fb['tof_min'], 2*WEEK, 3*MONTH, 2*WEEK, 3*MONTH, 2*WEEK]) / YEAR,
        np.array([launch_range[1]-launch_range[0], fb['tof_max'], 5*YEAR, YEAR,
                  5*YEAR, YEAR, 5*YEAR]) / YEAR))

    t0 = time.time()
    best = None
    best_full = None
    # Multi-seed × small m-revs sweep
    for seed in (42, 137, 314):
        for m_revs in [(0,0,0,0), (1,0,0,0), (0,1,0,0)]:
            try:
                res = differential_evolution(
                    lambda x: score_paths_flyby(x, *ids, launch_range, 'mars', *m_revs),
                    bounds, maxiter=300, tol=1e-7, seed=seed, polish=True,
                    popsize=18, mutation=(0.5, 1.3), recombination=0.8,
                    updating='deferred')
                ets = _unpack_flyby_input(res.x, launch_range, 'mars')
                full = compute_path_with_flyby(*ids, *ets, 'mars', *m_revs)
                full.update(dict(zip(['et_launch','et_flyby','et_arrive_1','et_stay_1',
                                      'et_arrive_2','et_stay_2','et_arrive_3'], ets)))
                full['m_revs'] = m_revs
                full['_seed']  = seed
                if best is None or full['delta_v_total'] < best:
                    best = full['delta_v_total']
                    best_full = full
            except Exception:
                continue
    if best_full is None:
        return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names,
                'error': 'all DE runs failed', 'elapsed_s': time.time() - t0}
    return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names, 'ids': ids,
            'best': best_full, 'arch': 'mars', 'elapsed_s': time.time() - t0}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    os.chdir(project_root)

    from core import load_kernels, audit_flyby_geometry, YEAR
    from optimization import (load_composition_map,
                              _triplet_has_diverse_composition)

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids', flush=True)

    comp_map = load_composition_map('asteroid_tradeoff.csv')
    counts = {}
    for a in asteroid_list:
        cls = comp_map.get(a['NAME'].upper(), 'Unknown')
        counts[cls] = counts.get(cls, 0) + 1
    print(f'Composition: {counts}', flush=True)

    # ---- Build candidate triplet set ----
    et_min = spiceypy.str2et('Jan 1 12:00:00 UTC 2027')
    et_max = spiceypy.str2et('Dec 31 12:00:00 UTC 2035')
    launch_range = [et_min, et_max]

    required = {'C', 'S', 'X/M'}
    n = len(asteroid_list)
    triplets = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)
                if i != j and j != k and i != k
                and _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                      comp_map, required)]
    print(f'\n{len(triplets)} composition-diverse (C+S+X/M) triplets', flush=True)

    # ---- Stage 1: coarse Mars-only screen ----
    coarse_tasks = []
    for (i, j, k) in triplets:
        ids = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        coarse_tasks.append((i, j, k, ids, names, launch_range))

    N_WORKERS = mp.cpu_count()
    print(f'\n=== Stage 1: coarse Mars-only screen on {len(coarse_tasks)} triplets, '
          f'{N_WORKERS} workers ===', flush=True)
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        coarse = []
        done = 0
        for r in pool.imap_unordered(_eval_coarse, coarse_tasks, chunksize=20):
            coarse.append(r); done += 1
            if done % 1000 == 0:
                print(f'  coarse {done}/{len(coarse_tasks)}  '
                      f'({time.time()-t0:.0f}s)', flush=True)
    coarse.sort(key=lambda r: r['coarse_dv'])
    print(f'Stage 1 done in {time.time()-t0:.0f}s', flush=True)

    print(f'\nTop 10 coarse Mars-only screen:', flush=True)
    for r in coarse[:10]:
        n_ = r['names']
        print(f"  {n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}  "
              f"dv≈{r['coarse_dv']:6.2f}", flush=True)

    # ---- Stage 2: full DE on top 50 ----
    TOP_N = 50
    fine_tasks = []
    for rank, r in enumerate(coarse[:TOP_N], 1):
        fine_tasks.append((rank, r['i'], r['j'], r['k'], r['ids'],
                           r['names'], launch_range))

    print(f'\n=== Stage 2: full DE Mars-only on top {TOP_N}, '
          f'{N_WORKERS} workers ===', flush=True)
    print(f'  per triplet: 3 seeds x 3 m-revs combos x 300 iters x 18 pop', flush=True)
    print('-' * 90, flush=True)
    print(f'{"#":>3s}  {"triplet":40s}  {"dv":>6s}  t(s)', flush=True)
    print('-' * 90, flush=True)

    t0 = time.time()
    fine = []
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for out in pool.imap_unordered(_eval_fine, fine_tasks, chunksize=1):
            fine.append(out)
            if 'error' in out:
                print(f"  #{out['rank']:<3d} ERROR: {out['error'][:60]}",
                      flush=True)
                continue
            n_ = out['names']
            tag = f"{n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}"
            print(f"  #{out['rank']:<3d} {tag}  "
                  f"{out['best']['delta_v_total']:6.2f}  "
                  f"{out['elapsed_s']:4.0f}", flush=True)
    print('-' * 90, flush=True)
    print(f'Stage 2 done in {time.time()-t0:.0f}s', flush=True)

    valid = [r for r in fine if 'error' not in r
             and r['best']['delta_v_total'] < 100]
    valid.sort(key=lambda r: r['best']['delta_v_total'])

    # ---- Stage 3: independent flyby physics audit ----
    print(f'\n=== Stage 3: independent geometric audit (top 20) ===', flush=True)
    print(f'{"rank":>4s}  {"triplet":40s}  {"dv":>6s}  '
          f'{"|vinf|":>7s}  {"turn":>5s}  {"max":>5s}  {"feas":>5s}', flush=True)
    print('-' * 100, flush=True)

    audited = []
    for rank, r in enumerate(valid[:20], 1):
        b = r['best']
        try:
            audit = audit_flyby_geometry(b['et_launch'], b['et_flyby'],
                                          b['et_arrive_1'], r['ids'][0], 'mars')
        except Exception as e:
            audit = {'feasible': False, 'reason': str(e)}
        audited.append({'rank': rank, 'i': r['i'], 'j': r['j'], 'k': r['k'],
                         'names': r['names'], 'ids': r['ids'], 'arch': 'mars',
                         'best': b, 'audit': audit})
        n_ = r['names']
        tag = f"{n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}"
        if audit.get('feasible'):
            print(f'  {rank:>4d}  {tag}  {b["delta_v_total"]:6.2f}  '
                  f'{audit.get("v_inf_in_kms", 0):6.2f}  '
                  f'{audit.get("turn_angle_deg", 0):4.1f}  '
                  f'{audit.get("turn_max_deg", 0):4.1f}  OK', flush=True)
        else:
            print(f'  {rank:>4d}  {tag}  {b["delta_v_total"]:6.2f}  '
                  f'{"":17s}  FAIL ({audit.get("reason", "geom")})', flush=True)

    # ---- Final report: top results that survive both DE penalty AND audit ----
    feasible_only = [a for a in audited if a['audit'].get('feasible')]

    print(f'\n{"="*100}', flush=True)
    print(f' MARS-ONLY, COMPOSITION-DIVERSE, PHYSICALLY-FEASIBLE — TOP RESULTS', flush=True)
    print(f'{"="*100}', flush=True)
    print(f'  {len(feasible_only)} of top-20 survived geometric audit', flush=True)
    print(flush=True)

    for rank, a in enumerate(feasible_only[:10], 1):
        n_ = a['names']
        b = a['best']
        au = a['audit']
        comps = [comp_map.get(name.upper(), '?') for name in n_]
        print(f'#{rank}  {n_[0]} [{comps[0]}] -> {n_[1]} [{comps[1]}] -> '
              f'{n_[2]} [{comps[2]}]', flush=True)
        print(f'    Total Δv             : {b["delta_v_total"]:.3f} km/s', flush=True)
        print(f'    Mission duration     : '
              f'{(b["et_arrive_3"]-b["et_launch"])/YEAR:.2f} yr', flush=True)
        print(f'    Mars |v_inf_in/out|  : {au["v_inf_in_kms"]:.3f} / '
              f'{au["v_inf_out_kms"]:.3f} km/s', flush=True)
        print(f'    Mars turn angle      : {au["turn_angle_deg"]:.2f}°  '
              f'(max possible: {au["turn_max_deg"]:.2f}°)', flush=True)
        print(f'    Mars periapsis alt   : {au["periapsis_alt_km"]:.0f} km '
              f'(safe min: 200 km)', flush=True)
        print(f'    Powered flyby Δv     : '
              f'{abs(float(b.get("delta_v_flyby", 0))):.3f} km/s', flush=True)
        print(f'    Launch dv            : '
              f'{np.linalg.norm(b["delta_v_launch"]):.3f} km/s', flush=True)
        print(f'    Launch date          : '
              f'{spiceypy.et2utc(b["et_launch"], "C", 0)}', flush=True)
        print(flush=True)

    # ---- Save ----
    out_path = 'optimal_asteroid_paths/pkl/mars_diverse_feasible.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'audited': audited,
                      'top_feasible': feasible_only,
                      'fine_results': fine,
                      'coarse_results': coarse,
                      'launch_window': ('Jan 1 2027', 'Dec 31 2035'),
                      'flyby': 'mars',
                      'composition_required': sorted(required)}, f)
    print(f'Saved: {out_path}', flush=True)
