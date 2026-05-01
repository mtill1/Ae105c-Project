#!/usr/bin/env python3
"""Mars-only, composition-diverse, physically-feasible — with SCIENCE weighting.

Same as run_mars_diverse.py but the final ranking blends Δv with the
science score from asteroid_tradeoff.csv:

    combined_score = alpha * dv + (1 - alpha) * (SCI_REF - sci_sum)

where sci_sum is the sum of Total_WeightedScore over the three asteroids
in the triplet, and SCI_REF (default 15.0) is a reference value that
makes the units comparable to Δv (km/s).

`alpha = 1.0` ⇒ pure Δv (matches run_mars_diverse.py).
`alpha = 0.7` ⇒ 70% Δv, 30% science (recommended starting point).
`alpha = 0.5` ⇒ equal weighting.

Pipeline:
  Stage 1: coarse Mars-only screen on every C+S+X/M triplet (combined score).
  Stage 2: full DE on top 50 (DE objective is still pure Δv since timing
           doesn't affect science scores; we re-rank afterward by combined).
  Stage 3: independent geometric audit of every flyby.

Configuration via environment variables:
  ALPHA               - weighting factor (default 0.7; lower = more science)
  SCI_REF             - reference for science penalty (default 15.0)
  TOP_FINE_N          - number of triplets to fine-optimize (default 50)
  REQUIRED_ASTEROIDS  - comma-separated names; at least one must appear
                        in every triplet (default: no constraint)
  REQUIRE_ALL_ASTEROIDS - comma-separated names; ALL of them must appear
                        in every triplet (overrides REQUIRED_ASTEROIDS)
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


def _eval_coarse(args):
    """Stage 1: quick Mars-only screen + science score lookup."""
    i, j, k, ids, names, sci_scores, launch_range, alpha, sci_ref = args
    from optimization import optimize_times_flyby_quick
    sci_sum = sum(sci_scores.get(n.upper(), 0.0) for n in names)
    try:
        dv = optimize_times_flyby_quick(*ids, launch_range, 'mars')
        combined = alpha * dv + (1 - alpha) * (sci_ref - sci_sum)
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': float(dv), 'sci_sum': float(sci_sum),
                'combined': float(combined)}
    except Exception:
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': 1e6, 'sci_sum': float(sci_sum),
                'combined': 1e6}


def _eval_fine(args):
    """Stage 2: full DE Mars-only on one triplet (objective is dv only)."""
    rank, i, j, k, ids, names, sci_sum, launch_range = args
    from optimization import (score_paths_flyby, _unpack_flyby_input,
                              compute_path_with_flyby, FLYBY_BODIES)
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
                'sci_sum': float(sci_sum),
                'error': 'all DE runs failed', 'elapsed_s': time.time() - t0}
    return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names, 'ids': ids,
            'sci_sum': float(sci_sum),
            'best': best_full, 'arch': 'mars', 'elapsed_s': time.time() - t0}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    os.chdir(project_root)

    ALPHA       = float(os.environ.get('ALPHA',     '0.7'))
    SCI_REF     = float(os.environ.get('SCI_REF',  '15.0'))
    TOP_FINE_N  = int(  os.environ.get('TOP_FINE_N', '50'))
    required_str = os.environ.get('REQUIRED_ASTEROIDS', '').strip()
    REQUIRED_ASTEROIDS = (
        set(s.strip().upper() for s in required_str.split(',') if s.strip())
        if required_str else set())
    require_all_str = os.environ.get('REQUIRE_ALL_ASTEROIDS', '').strip()
    REQUIRE_ALL_ASTEROIDS = (
        set(s.strip().upper() for s in require_all_str.split(',') if s.strip())
        if require_all_str else set())

    print(f'\nMars-only, composition-diverse, science-weighted optimization', flush=True)
    print(f'  ALPHA       : {ALPHA} ({100*ALPHA:.0f}% Δv + '
          f'{100*(1-ALPHA):.0f}% science)', flush=True)
    print(f'  SCI_REF     : {SCI_REF}', flush=True)
    print(f'  TOP_FINE_N  : {TOP_FINE_N}', flush=True)
    if REQUIRE_ALL_ASTEROIDS:
        print(f'  REQUIRE ALL : every triplet must contain ALL of '
              f'{sorted(REQUIRE_ALL_ASTEROIDS)}', flush=True)
    elif REQUIRED_ASTEROIDS:
        print(f'  REQUIRED    : at least one of {sorted(REQUIRED_ASTEROIDS)} '
              f'must be in each triplet', flush=True)
    print(flush=True)

    from core import load_kernels, audit_flyby_geometry, YEAR
    from optimization import (load_composition_map,
                              _triplet_has_diverse_composition)

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroids', flush=True)

    comp_map = load_composition_map('asteroid_tradeoff.csv')

    # Load science scores from asteroid_tradeoff.csv
    import pandas as pd
    df = pd.read_csv('asteroid_tradeoff.csv')
    sci_scores = {}
    for _, row in df.iterrows():
        raw = str(row['Name_DecRadius']).split('(')[0].strip()
        parts = raw.split()
        name = ' '.join(parts[1:]).upper() if parts and parts[0].replace('.','').isdigit() \
               else raw.upper()
        sci_scores[name] = float(row['Total_WeightedScore'])
    print(f'Loaded {len(sci_scores)} science scores', flush=True)
    print(f'Score range: [{min(sci_scores.values()):.2f}, '
          f'{max(sci_scores.values()):.2f}]', flush=True)

    # Build candidate triplets
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

    if REQUIRE_ALL_ASTEROIDS:
        before = len(triplets)
        triplets = [
            (i, j, k) for (i, j, k) in triplets
            if REQUIRE_ALL_ASTEROIDS.issubset(
                {asteroid_list[x]['NAME'].upper() for x in (i, j, k)})]
        print(f'After REQUIRE_ALL filter ({sorted(REQUIRE_ALL_ASTEROIDS)}): '
              f'{len(triplets)} triplets ({before - len(triplets)} removed)',
              flush=True)
        if not triplets:
            sys.exit('No triplets satisfy REQUIRE_ALL constraint.')
    elif REQUIRED_ASTEROIDS:
        before = len(triplets)
        triplets = [
            (i, j, k) for (i, j, k) in triplets
            if any(asteroid_list[x]['NAME'].upper() in REQUIRED_ASTEROIDS
                    for x in (i, j, k))]
        print(f'After REQUIRED filter ({sorted(REQUIRED_ASTEROIDS)}): '
              f'{len(triplets)} triplets ({before - len(triplets)} removed)',
              flush=True)
        if not triplets:
            sys.exit('No triplets satisfy the required-asteroids constraint.')

    # ---- Stage 1: coarse + science-weighted ranking ----
    coarse_tasks = []
    for (i, j, k) in triplets:
        ids = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        coarse_tasks.append((i, j, k, ids, names, sci_scores,
                              launch_range, ALPHA, SCI_REF))

    N_WORKERS = mp.cpu_count()
    print(f'\n=== Stage 1: coarse pass with science weighting ===', flush=True)
    t0 = time.time()
    coarse = []
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        done = 0
        for r in pool.imap_unordered(_eval_coarse, coarse_tasks, chunksize=20):
            coarse.append(r); done += 1
            if done % 1500 == 0:
                print(f'  coarse {done}/{len(coarse_tasks)}  '
                      f'({time.time()-t0:.0f}s)', flush=True)
    coarse.sort(key=lambda r: r['combined'])
    print(f'Stage 1 done in {time.time()-t0:.0f}s', flush=True)

    print(f'\nTop 10 coarse by combined score (alpha={ALPHA}):', flush=True)
    print(f'  {"triplet":40s}  {"dv":>5s}  {"sci":>5s}  {"score":>5s}', flush=True)
    for r in coarse[:10]:
        n_ = r['names']
        print(f"  {n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}  "
              f"{r['coarse_dv']:5.2f}  {r['sci_sum']:5.2f}  "
              f"{r['combined']:5.2f}", flush=True)

    # ---- Stage 2: full DE on top TOP_FINE_N (by combined score) ----
    fine_tasks = []
    for rank, r in enumerate(coarse[:TOP_FINE_N], 1):
        fine_tasks.append((rank, r['i'], r['j'], r['k'], r['ids'],
                            r['names'], r['sci_sum'], launch_range))

    print(f'\n=== Stage 2: full DE on top {TOP_FINE_N} (objective: pure Δv) ===',
          flush=True)
    print(f'  per triplet: 3 seeds × 3 m-revs × 300 iters × pop 18', flush=True)
    print('-' * 90, flush=True)
    print(f'  {"#":>3s}  {"triplet":40s}  {"dv":>5s}  {"sci":>5s}  '
          f'{"score":>5s}  t(s)', flush=True)
    print('-' * 90, flush=True)

    t0 = time.time()
    fine = []
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for out in pool.imap_unordered(_eval_fine, fine_tasks, chunksize=1):
            fine.append(out)
            if 'error' in out:
                print(f"  #{out['rank']:<3d} ERROR", flush=True); continue
            n_ = out['names']
            dv = out['best']['delta_v_total']
            sci = out['sci_sum']
            score = ALPHA * dv + (1 - ALPHA) * (SCI_REF - sci)
            tag = f"{n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}"
            print(f"  #{out['rank']:<3d} {tag}  {dv:5.2f}  {sci:5.2f}  "
                  f"{score:5.2f}  {out['elapsed_s']:4.0f}", flush=True)
    print('-' * 90, flush=True)
    print(f'Stage 2 done in {time.time()-t0:.0f}s', flush=True)

    valid = [r for r in fine if 'error' not in r
             and r['best']['delta_v_total'] < 100]

    # Re-rank by combined score (now using true DE-optimized dv)
    for r in valid:
        dv = r['best']['delta_v_total']
        r['combined'] = ALPHA * dv + (1 - ALPHA) * (SCI_REF - r['sci_sum'])
    valid.sort(key=lambda r: r['combined'])

    # ---- Stage 3: physical-flyby audit on top 20 ----
    print(f'\n=== Stage 3: independent geometric audit (top 20) ===', flush=True)
    print(f'  {"rank":>4s}  {"triplet":40s}  {"dv":>5s}  {"sci":>5s}  '
          f'{"score":>5s}  feas', flush=True)
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
                         'best': b, 'audit': audit, 'sci_sum': r['sci_sum'],
                         'combined': r['combined']})
        n_ = r['names']
        tag = f"{n_[0][:11]:11s}->{n_[1][:11]:11s}->{n_[2][:11]:11s}"
        feas = 'OK' if audit.get('feasible') else 'FAIL'
        print(f'  {rank:>4d}  {tag}  {b["delta_v_total"]:5.2f}  {r["sci_sum"]:5.2f}  '
              f'{r["combined"]:5.2f}  {feas}', flush=True)

    feasible_only = [a for a in audited if a['audit'].get('feasible')]

    print(f'\n{"="*100}', flush=True)
    print(f' MARS-ONLY, COMPOSITION-DIVERSE, SCIENCE-WEIGHTED, FEASIBLE — TOP RESULTS',
          flush=True)
    print(f' (alpha = {ALPHA}, sci_ref = {SCI_REF})', flush=True)
    print(f'{"="*100}', flush=True)
    print(f'  {len(feasible_only)} of top-20 survived geometric audit', flush=True)
    print(flush=True)

    for rank, a in enumerate(feasible_only[:10], 1):
        n_ = a['names']
        b = a['best']
        au = a['audit']
        comps = [comp_map.get(name.upper(), '?') for name in n_]
        per_ast_sci = [sci_scores.get(name.upper(), 0.0) for name in n_]
        print(f'#{rank}  {n_[0]} [{comps[0]}, sci={per_ast_sci[0]:.2f}] -> '
              f'{n_[1]} [{comps[1]}, sci={per_ast_sci[1]:.2f}] -> '
              f'{n_[2]} [{comps[2]}, sci={per_ast_sci[2]:.2f}]', flush=True)
        print(f'    Combined score        : {a["combined"]:.3f}', flush=True)
        print(f'    Total Δv              : {b["delta_v_total"]:.3f} km/s', flush=True)
        print(f'    Science sum           : {a["sci_sum"]:.3f}', flush=True)
        print(f'    Mission duration      : '
              f'{(b["et_arrive_3"]-b["et_launch"])/YEAR:.2f} yr', flush=True)
        print(f'    Mars |v_inf_in/out|   : {au["v_inf_in_kms"]:.3f} / '
              f'{au["v_inf_out_kms"]:.3f} km/s', flush=True)
        print(f'    Mars turn / max       : {au["turn_angle_deg"]:.2f}° / '
              f'{au["turn_max_deg"]:.2f}°', flush=True)
        print(f'    Periapsis altitude    : {au["periapsis_alt_km"]:.0f} km', flush=True)
        print(f'    Launch                : '
              f'{spiceypy.et2utc(b["et_launch"], "C", 0)}', flush=True)
        print(flush=True)

    suffix = f'a{int(ALPHA*100)}'
    if REQUIRE_ALL_ASTEROIDS:
        suffix += '_all_' + '_'.join(sorted(REQUIRE_ALL_ASTEROIDS))
    elif REQUIRED_ASTEROIDS:
        suffix += '_req' + '_'.join(sorted(REQUIRED_ASTEROIDS))
    out_path = f'optimal_asteroid_paths/pkl/mars_diverse_science_{suffix}.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'alpha': ALPHA, 'sci_ref': SCI_REF,
                      'required_asteroids': sorted(REQUIRED_ASTEROIDS),
                      'audited': audited,
                      'top_feasible': feasible_only,
                      'fine_results': fine,
                      'coarse_results': coarse[:200],  # save top 200 only (size)
                      'launch_window': ('Jan 1 2027', 'Dec 31 2035'),
                      'flyby': 'mars',
                      'composition_required': sorted(required)}, f)
    print(f'Saved: {out_path}', flush=True)
