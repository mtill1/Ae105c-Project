"""
main.py — Single CLI entry point for the Ae105c asteroid mission pipeline.

USAGE
-----
Run from the repo root (Ae105c-Project/), not from Python_Consolidated/.

    # Optimization
    python Python_Consolidated/main.py optimize                       # two-level, all asteroids, dv-only
    python Python_Consolidated/main.py optimize --science 0.7         # 70% dv + 30% science
    python Python_Consolidated/main.py optimize --diverse             # require C+S+X/M
    python Python_Consolidated/main.py optimize --beam 15             # beam search
    python Python_Consolidated/main.py optimize --pareto              # mass-Pareto across 8 architectures

    # Visualization
    python Python_Consolidated/main.py list                           # list pkl results
    python Python_Consolidated/main.py plot RESULT.pkl                # show top-10 from a result
    python Python_Consolidated/main.py plot RESULT.pkl --rank 1       # static 3D plot of #1
    python Python_Consolidated/main.py plot RESULT.pkl --rank 1 --gif # animated GIF of #1
    python Python_Consolidated/main.py plot RESULT.pkl --names HEDDA BEATRIX PROSERPINA --gif

    # Auxiliary
    python Python_Consolidated/main.py rank                           # rebuild asteroid_tradeoff.csv
    python Python_Consolidated/main.py animate-asteroids              # MP4 of all asteroid orbits
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from glob import glob

import numpy as np


# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

DEFAULT_BSP_FOLDER = 'NOTABLE_ASTEROID_BSPs'
DEFAULT_KERNELS = 'generic_kernels'           # symlink in repo root
DEFAULT_TRADEOFF = 'asteroid_tradeoff.csv'
DEFAULT_PKL_DIR = 'optimal_asteroid_paths/pkl'
DEFAULT_RENDER_DIR = 'Renders'
DEFAULT_LAUNCH_MIN = 'Jan 1 12:00:00 UTC 2027'
DEFAULT_LAUNCH_MAX = 'Dec 31 12:00:00 UTC 2035'


# ---------------------------------------------------------------------------
# Lazy imports — keep CLI startup fast and avoid importing pykep/spiceypy
# until the user picks a subcommand that actually needs them.
# ---------------------------------------------------------------------------

def _setup_repo_root():
    """cd to repo root so relative paths (NOTABLE_ASTEROID_BSPs/, etc.) work."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    sys.path.insert(0, here)


def _load_asteroids(bsp_folder, kernels_path):
    from core import load_kernels
    return load_kernels(bsp_folder, kernels_path)


def _load_science_scores(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        name = str(row['Name_DecRadius']).split('(')[0].strip()
        # Strip leading number ("24 Themis" -> "Themis")
        parts = name.split()
        if parts and parts[0].replace('.', '').isdigit():
            name = ' '.join(parts[1:])
        out[name.upper()] = row['Total_WeightedScore']
    return out


# ---------------------------------------------------------------------------
# Subcommand: optimize
# ---------------------------------------------------------------------------

def cmd_optimize(args):
    _setup_repo_root()
    from optimization import (two_level_optimize, beam_search,
                              load_composition_map)

    asteroid_list = _load_asteroids(args.bsp, args.kernels)

    science_scores = None
    alpha = 1.0
    if args.science is not None:
        science_scores = _load_science_scores(args.tradeoff_csv)
        alpha = float(args.science)
        print(f"Science weighting: alpha={alpha} ({100*alpha:.0f}% dv + "
              f"{100*(1-alpha):.0f}% science)")

    comp_map = None
    if args.diverse or args.feasible:
        comp_map = load_composition_map(args.tradeoff_csv)

    if args.feasible:
        return _run_feasible(args, asteroid_list, comp_map)
    if args.pareto:
        return _run_pareto(args, asteroid_list)
    if args.beam:
        return beam_search(asteroid_list, args.launch_min, args.launch_max,
                           beam_width=args.beam, science_scores=science_scores,
                           alpha=alpha)
    return two_level_optimize(
        asteroid_list, 0, 0, 0, args.launch_min, args.launch_max,
        top_n=args.top_n, science_scores=science_scores, alpha=alpha,
        comp_map=comp_map)


def _run_feasible(args, asteroid_list, comp_map):
    """Composition-diverse Δv minimization with physical-flyby audit.

    Three stages:
      1. Coarse architecture screen on every C+S+X/M triplet (parallel).
      2. Full DE on the top-N triplets across direct/Moon-GA/Mars-GA.
      3. Independent geometric audit of every flyby — only solutions whose
         turn angle fits within Mars/Moon natural max at the safe periapsis
         altitude are reported.
    """
    import multiprocessing as mp
    import time
    import spiceypy
    from optimization import (optimize_best_architecture, FLYBY_BODIES,
                              _triplet_has_diverse_composition)
    from core import audit_flyby_geometry, YEAR

    et_min = spiceypy.str2et(args.launch_min)
    et_max = spiceypy.str2et(args.launch_max)
    launch_range = [et_min, et_max]

    required = {'C', 'S', 'X/M'}
    n = len(asteroid_list)
    triplets = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)
                if i != j and j != k and i != k
                and _triplet_has_diverse_composition(i, j, k, asteroid_list,
                                                      comp_map, required)]
    print(f"\n{len(triplets)} composition-diverse (C+S+X/M) triplets to evaluate")

    coarse_tasks = []
    for (i, j, k) in triplets:
        ids = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        coarse_tasks.append((i, j, k, ids, names, launch_range))

    N_WORKERS = mp.cpu_count()
    print(f"\nStage 1: coarse pass ({N_WORKERS} workers)")
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_feasible_init_worker) as pool:
        coarse = list(pool.imap_unordered(_feasible_eval_coarse,
                                            coarse_tasks, chunksize=20))
    coarse.sort(key=lambda r: r['coarse_dv'])
    print(f"  done in {time.time()-t0:.0f}s")

    TOP = args.feasible_top_n
    fine_tasks = []
    for rank_, r in enumerate(coarse[:TOP], 1):
        fine_tasks.append((rank_, r['i'], r['j'], r['k'],
                           r['ids'], r['names'], launch_range))
    print(f"\nStage 2: full DE on top {TOP} ({N_WORKERS} workers)")
    t0 = time.time()
    fine = []
    with mp.Pool(N_WORKERS, initializer=_feasible_init_worker) as pool:
        for out in pool.imap_unordered(_feasible_eval_fine, fine_tasks,
                                        chunksize=1):
            fine.append(out)
    fine = [r for r in fine if 'error' not in r
            and r['best']['delta_v_total'] < 100]
    fine.sort(key=lambda r: r['best']['delta_v_total'])
    print(f"  done in {time.time()-t0:.0f}s")

    print(f"\nStage 3: physical-flyby audit on top {min(10, len(fine))}")
    audited = []
    for rank_, r in enumerate(fine[:10], 1):
        b = r['best']; arch = r['arch']
        if arch == 'direct':
            audit = {'feasible': True, 'note': 'no flyby',
                     'v_inf_in_kms': 0.0, 'v_inf_out_kms': 0.0,
                     'turn_angle_deg': 0.0, 'turn_max_deg': 0.0,
                     'periapsis_alt_km': float('inf')}
        else:
            audit = audit_flyby_geometry(b['et_launch'], b['et_flyby'],
                                          b['et_arrive_1'], r['ids'][0], arch)
        audited.append({'rank': rank_, 'i': r['i'], 'j': r['j'], 'k': r['k'],
                         'names': r['names'], 'ids': r['ids'], 'arch': arch,
                         'best': b, 'audit': audit})

    print()
    print(f"  {'rank':>4s} {'triplet':40s} {'arch':>6s} {'dv':>6s} {'feas':>5s}")
    for a in audited:
        names = a['names']
        tag = f"{names[0][:11]}->{names[1][:11]}->{names[2][:11]}"
        print(f"  {a['rank']:>4d} {tag:40s} {a['arch']:>6s} "
              f"{a['best']['delta_v_total']:6.2f} "
              f"{'OK' if a['audit']['feasible'] else 'FAIL':>5s}")

    feasible_only = [a for a in audited if a['audit']['feasible']]
    print(f"\nTop 3 with feasible flyby:")
    comps = lambda nm: comp_map.get(nm.upper(), '?')
    for rank_, a in enumerate(feasible_only[:3], 1):
        n_ = a['names']
        print(f"  #{rank_}  {n_[0]} [{comps(n_[0])}] -> "
              f"{n_[1]} [{comps(n_[1])}] -> {n_[2]} [{comps(n_[2])}]  "
              f"({a['arch']})  dv={a['best']['delta_v_total']:.3f} km/s")

    out_path = os.path.join(DEFAULT_PKL_DIR, 'diverse_top3_feasible.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'audited': audited, 'top3_feasible': feasible_only[:3],
                      'fine_results': fine, 'coarse_results': coarse}, f)
    print(f"\nSaved: {out_path}")
    return audited


def _feasible_init_worker():
    """Re-load SPICE kernels in each multiprocessing worker."""
    import spiceypy
    from glob import glob as _glob
    repo_root = os.getcwd()
    gk = os.path.join(repo_root, 'generic_kernels')
    spiceypy.furnsh(os.path.join(gk, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'planets',    'de430.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'pck00010.tpc'))
    for bsp in sorted(_glob(os.path.join(repo_root,
                                          'NOTABLE_ASTEROID_BSPs', '*.bsp'))):
        spiceypy.furnsh(bsp)


def _feasible_eval_coarse(args):
    i, j, k, ids, names, launch_range = args
    from optimization import optimize_best_architecture
    try:
        best_dv, arch = optimize_best_architecture(*ids, launch_range, quick=True)
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': float(best_dv), 'arch_quick': arch}
    except Exception:
        return {'i': i, 'j': j, 'k': k, 'ids': ids, 'names': names,
                'coarse_dv': 1e6, 'arch_quick': 'error'}


def _feasible_eval_fine(args):
    import time
    rank, i, j, k, ids, names, launch_range = args
    from optimization import optimize_best_architecture
    t0 = time.time()
    try:
        best, arch = optimize_best_architecture(*ids, launch_range, quick=False)
        return {'rank': rank, 'i': i, 'j': j, 'k': k,
                'names': names, 'ids': ids, 'best': best,
                'arch': arch, 'elapsed_s': time.time() - t0}
    except Exception as e:
        return {'rank': rank, 'i': i, 'j': j, 'k': k, 'names': names,
                'error': str(e), 'elapsed_s': time.time() - t0}


def _run_pareto(args, asteroid_list):
    """Mass-Pareto across all 8 propulsion architectures, top-N triplets."""
    from mass_optimization import (pareto_optimize_triplet, verify_with_full_lt,
                                   ARCH_CODES)
    import spiceypy

    seed_pkl = args.pareto_seed or os.path.join(DEFAULT_PKL_DIR,
                                                'results_69ast_ga.pkl')
    if not os.path.exists(seed_pkl):
        sys.exit(f"Need a seed result pkl (got {seed_pkl}). "
                 f"Run plain optimize first or pass --pareto-seed.")
    with open(seed_pkl, 'rb') as f:
        seed = pickle.load(f)
    if isinstance(seed, dict) and 'all_results' in seed:
        seed = [(r['i'], r['j'], r['k'], r) for r in seed['all_results']
                if 'i' in r]
    triplets = seed[:args.top_n]
    print(f"Pareto sweep: {len(triplets)} triplets x 8 archs from {seed_pkl}")

    et_min = spiceypy.str2et(args.launch_min)
    et_max = spiceypy.str2et(args.launch_max)
    launch_range = [et_min, et_max]

    all_results = []
    for rank, (i, j, k, _) in enumerate(triplets, 1):
        ids = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        print(f"\n[{rank}/{len(triplets)}] {' -> '.join(names)}")
        try:
            res = pareto_optimize_triplet(
                *ids, launch_range, flyby_name=args.flyby,
                archs=ARCH_CODES,
                m_revs_options=[(0, 0, 0, 0), (1, 0, 0, 0),
                                (0, 1, 0, 0), (0, 0, 1, 0)],
                seeds=(42, 137, 314), maxiter=200, popsize=15,
                m_init_kg=args.m_init, thrust_N=args.thrust)
        except Exception as e:
            all_results.append({'rank': rank, 'i': i, 'j': j, 'k': k,
                                'names': names, 'error': str(e)})
            continue
        if not res:
            all_results.append({'rank': rank, 'i': i, 'j': j, 'k': k,
                                'names': names, 'error': 'all infeasible'})
            continue
        sorted_archs = sorted(res.items(), key=lambda kv: -kv[1]['m_final_kg'])
        verified = []
        for arch, surr in sorted_archs[:3]:
            try:
                v = verify_with_full_lt(surr, *ids,
                                        m_init_kg=args.m_init,
                                        thrust_N=args.thrust)
                verified.append(v)
            except Exception as e:
                verified.append({**surr, 'verified': False,
                                 'verify_error': str(e)})
        feasible = [v for v in verified if v.get('verified_feasible')]
        best = (max(feasible, key=lambda v: v['verified_m_final_kg'])
                if feasible else verified[0])
        all_results.append({'rank': rank, 'i': i, 'j': j, 'k': k,
                            'names': names, 'all_archs': res,
                            'verified_top3': verified, 'best_verified': best})
        print(f"  best arch={best.get('arch_code')}  "
              f"m_final={best.get('verified_m_final_kg', 0):.0f} kg")

    out = os.path.join(DEFAULT_PKL_DIR, f'results_mass_pareto_{int(__import__("time").time())}.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump({'source_pkl': seed_pkl, 'm_init_kg': args.m_init,
                     'thrust_N': args.thrust, 'flyby': args.flyby,
                     'launch_range_utc': (args.launch_min, args.launch_max),
                     'all_results': all_results}, f)
    print(f"\nSaved: {out}")
    return all_results


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args):
    _setup_repo_root()
    pkls = sorted(glob(os.path.join(DEFAULT_PKL_DIR, '*.pkl')))
    if not pkls:
        print(f"No results found in {DEFAULT_PKL_DIR}/")
        return
    print(f"{len(pkls)} results in {DEFAULT_PKL_DIR}/")
    for p in pkls:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {os.path.basename(p):55s}  {size_kb:7.1f} KB")


# ---------------------------------------------------------------------------
# Subcommand: plot — generic over the multiple pkl formats
# ---------------------------------------------------------------------------

def _normalize_entries(data):
    """Convert any saved-result format into a list of dicts with consistent keys.

    Output entries each have:
        names: (n1, n2, n3) — uppercase asteroid names if available, else None
        ids: optional int triplet (i, j, k)
        result: the underlying result dict (per-leg dvs, epochs, etc.)
        score: scalar used for sorting (delta_v_total or -m_final_kg)
        label: short string with the score for display
    """
    entries = []

    if isinstance(data, dict) and 'all_results' in data:
        # Mass-pareto format
        for r in data['all_results']:
            if r.get('error'):
                continue
            best = r.get('best_verified') or {}
            entries.append({
                'names': tuple(r['names']),
                'ids': (r.get('i'), r.get('j'), r.get('k')),
                'result': best,
                'score': -float(best.get('verified_m_final_kg', 0.0)),
                'label': (f"m_final={best.get('verified_m_final_kg', 0):.0f} kg  "
                          f"arch={best.get('arch_code', '?')}"),
            })
        entries.sort(key=lambda e: e['score'])
        return entries

    if isinstance(data, dict) and 'best_mars' in data:
        # Single-triplet format
        best = data['best_mars']
        # Names may be in there or have to be reconstructed by caller
        entries.append({
            'names': tuple(best.get('names', (None, None, None))),
            'ids': (None, None, None),
            'result': best,
            'score': float(best.get('delta_v_total', 1e3)),
            'label': f"dv_total={best.get('delta_v_total', 0):.2f} km/s",
        })
        return entries

    if isinstance(data, dict) and 'audited' in data:
        # diverse_top3_feasible format from `optimize --feasible`
        for a in data['audited']:
            best = a.get('best') or {}
            audit = a.get('audit') or {}
            entries.append({
                'names': tuple(a.get('names', (None, None, None))),
                'ids': (a.get('i'), a.get('j'), a.get('k')),
                'result': {**best,
                           'arch': a.get('arch'),
                           'flyby_name': a.get('arch'),
                           'audit': audit},
                'score': float(best.get('delta_v_total', 1e3)),
                'label': (f"dv_total={best.get('delta_v_total', 0):.2f} km/s  "
                          f"arch={a.get('arch')}  "
                          f"feas={'OK' if audit.get('feasible') else 'FAIL'}"),
            })
        entries.sort(key=lambda e: e['score'])
        return entries

    if isinstance(data, dict) and 'best_verified' in data:
        # Single-triplet result from gcp/run_single_triplet.py
        best = data['best_verified']
        names = data.get('triplet') or best.get('names', (None, None, None))
        entries.append({
            'names': tuple(names),
            'ids': (None, None, None),
            'result': best,
            'score': -float(best.get('verified_m_final_kg', 0.0)),
            'label': (f"m_final={best.get('verified_m_final_kg', 0):.0f} kg  "
                      f"arch={best.get('arch_code', '?')}"),
        })
        return entries

    if isinstance(data, list):
        # two_level_optimize: list of (i, j, k, result) tuples
        # beam_search: list of (i, j, k, dv, names, *epochs) tuples
        for item in data:
            if isinstance(item, tuple) and len(item) >= 4 and isinstance(item[3], dict):
                i, j, k, res = item[:4]
                entries.append({'names': (None, None, None),
                                'ids': (i, j, k), 'result': res,
                                'score': float(res.get('delta_v_total', 1e3)),
                                'label': f"dv_total={res.get('delta_v_total', 0):.2f} km/s"})
            elif isinstance(item, tuple) and len(item) >= 5:
                # beam: (i, j, k, dv, names, ...)
                i, j, k, dv, names = item[:5]
                entries.append({'names': tuple(names),
                                'ids': (i, j, k),
                                'result': {'beam_record': item, 'delta_v_total': dv},
                                'score': float(dv),
                                'label': f"dv_total={dv:.2f} km/s"})
        entries.sort(key=lambda e: e['score'])
        return entries

    raise ValueError(f"Unrecognized result format: {type(data)}")


def _resolve_names_ids(entries, asteroid_list):
    """Fill in names from ids (or vice versa) where missing."""
    name_to_idx = {a['NAME'].upper(): n for n, a in enumerate(asteroid_list)}
    for e in entries:
        i, j, k = e['ids']
        n1, n2, n3 = e['names']
        if n1 is None and i is not None:
            e['names'] = (asteroid_list[i]['NAME'].upper(),
                          asteroid_list[j]['NAME'].upper(),
                          asteroid_list[k]['NAME'].upper())
        elif i is None and n1 is not None:
            e['ids'] = tuple(name_to_idx.get(n.upper()) for n in (n1, n2, n3))


def cmd_plot(args):
    _setup_repo_root()
    if not os.path.exists(args.pkl):
        # Allow giving just the basename
        candidate = os.path.join(DEFAULT_PKL_DIR, args.pkl)
        if os.path.exists(candidate):
            args.pkl = candidate
        else:
            sys.exit(f"Result file not found: {args.pkl}")

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    entries = _normalize_entries(data)
    if not entries:
        sys.exit("No usable entries in this pkl.")

    asteroid_list = _load_asteroids(args.bsp, args.kernels)
    _resolve_names_ids(entries, asteroid_list)

    # Pick the entry to render
    chosen = None
    if args.names:
        target = tuple(n.upper() for n in args.names)
        for e in entries:
            if e['names'] == target:
                chosen = e
                break
        if chosen is None:
            sys.exit(f"No entry matches names {target}")
    elif args.rank:
        if args.rank < 1 or args.rank > len(entries):
            sys.exit(f"--rank {args.rank} out of range (1..{len(entries)})")
        chosen = entries[args.rank - 1]
    else:
        # Just list the top entries
        n_show = min(args.top, len(entries))
        print(f"{os.path.basename(args.pkl)}: {len(entries)} entries\n"
              f"Top {n_show}:")
        for r, e in enumerate(entries[:n_show], 1):
            names = ' -> '.join(e['names']) if all(e['names']) else '???'
            print(f"  #{r:>2}  {names:50s}  {e['label']}")
        print("\nRender with --rank N or --names A B C  (add --gif for animation)")
        return

    out_dir = args.out_dir or DEFAULT_RENDER_DIR
    os.makedirs(out_dir, exist_ok=True)
    base = '_'.join(n.lower() for n in chosen['names'])

    if args.gif:
        out = os.path.join(out_dir, f'{base}_trajectory.gif')
        _render_gif(chosen, asteroid_list, out, frames=args.frames, fps=args.fps)
    else:
        out = os.path.join(out_dir, f'{base}_trajectory.png')
        _render_static(chosen, asteroid_list, out)


# ---------------------------------------------------------------------------
# Trajectory reconstruction & rendering
# ---------------------------------------------------------------------------

def _build_legs(result, asteroid_list, names):
    """Reconstruct (label, ts, xs) tuples for each mission leg from a result dict.

    Handles both flyby (mass-pareto) and direct trajectory result formats.
    Returns (legs list, et_launch, et_arr3, has_flyby flag, flyby_name or None).
    """
    from core import (get_state, solve_lambert, propagate_two_body,
                      MU_SUN, DAY)
    import spiceypy  # noqa: F401  (kernels already loaded)

    name_to_idx = {a['NAME'].upper(): n for n, a in enumerate(asteroid_list)}
    a1_id = str(int(asteroid_list[name_to_idx[names[0]]]['ID']))
    a2_id = str(int(asteroid_list[name_to_idx[names[1]]]['ID']))
    a3_id = str(int(asteroid_list[name_to_idx[names[2]]]['ID']))

    et_launch = result['et_launch']
    et_arr_1 = result['et_arrive_1']
    et_stay_1 = result['et_stay_1']
    et_arr_2 = result['et_arrive_2']
    et_stay_2 = result['et_stay_2']
    et_arr_3 = result['et_arrive_3']
    has_flyby = 'et_flyby' in result and result.get('et_flyby') is not None
    fb_name = result.get('flyby_name') or result.get('architecture')
    et_flyby = result.get('et_flyby')
    m_revs = result.get('m_revs') or (0, 0, 0, 0)

    fb_body_id = {'mars': '4', 'moon': '301', 'earth': '399'}.get(fb_name)
    if has_flyby and fb_body_id is None:
        has_flyby = False  # 'direct' counts as no flyby

    if has_flyby:
        leg_specs = [
            (f'Earth -> {fb_name.title()}', '399',  et_launch, fb_body_id, et_flyby,  m_revs[0]),
            (f'{fb_name.title()} -> {names[0].title()}', fb_body_id, et_flyby,  a1_id,  et_arr_1,  m_revs[1]),
            (f'{names[0].title()} stay',    a1_id, et_arr_1,  a1_id,    et_stay_1, None),
            (f'{names[0].title()} -> {names[1].title()}', a1_id, et_stay_1, a2_id, et_arr_2,  m_revs[2]),
            (f'{names[1].title()} stay',    a2_id, et_arr_2,  a2_id,    et_stay_2, None),
            (f'{names[1].title()} -> {names[2].title()}', a2_id, et_stay_2, a3_id, et_arr_3,  m_revs[3]),
        ]
    else:
        leg_specs = [
            (f'Earth -> {names[0].title()}',           '399',  et_launch, a1_id, et_arr_1,  0),
            (f'{names[0].title()} stay',                a1_id, et_arr_1,  a1_id, et_stay_1, None),
            (f'{names[0].title()} -> {names[1].title()}', a1_id, et_stay_1, a2_id, et_arr_2,  0),
            (f'{names[1].title()} stay',                a2_id, et_arr_2,  a2_id, et_stay_2, None),
            (f'{names[1].title()} -> {names[2].title()}', a2_id, et_stay_2, a3_id, et_arr_3,  0),
        ]

    samples = 80
    legs = []
    for label, b0, et0, b1, et1, mrev in leg_specs:
        ts = np.linspace(et0, et1, samples)
        if mrev is None:
            xs = np.array([get_state(b0, t)[0] for t in ts])
        else:
            r0, v0 = get_state(b0, et0)
            r1, _ = get_state(b1, et1)
            tof = et1 - et0
            V1, _, ef = solve_lambert(r0, r1, tof / DAY, mrev, MU_SUN)
            if ef != 1:
                xs = np.array([r0 + (r1 - r0) * ((t - et0) / tof) for t in ts])
            else:
                xs = np.array([propagate_two_body(r0, V1, t - et0, MU_SUN)[0]
                               for t in ts])
        legs.append({'label': label, 'ts': ts, 'xs': xs})
    return legs, et_launch, et_arr_3, has_flyby, fb_name


def _render_static(entry, asteroid_list, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    legs, et_launch, et_arr_3, has_flyby, fb_name = _build_legs(
        entry['result'], asteroid_list, entry['names'])

    fig = plt.figure(figsize=(11, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#16a085']
    for leg, c in zip(legs, colors):
        xs = leg['xs']
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color=c, linewidth=1.5,
                label=leg['label'])
    ax.scatter([0], [0], [0], color='gold', s=120, edgecolor='orange')

    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    title = ' -> '.join(n.title() for n in entry['names'])
    if has_flyby:
        title += f"  via {fb_name.title()} flyby"
    ax.set_title(title)
    ax.legend(fontsize='small', loc='upper left')
    ax.view_init(elev=30, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _render_gif(entry, asteroid_list, out_path, frames=240, fps=24):
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import spiceypy

    from core import get_state

    legs, et_launch, et_arr_3, has_flyby, fb_name = _build_legs(
        entry['result'], asteroid_list, entry['names'])

    name_to_idx = {a['NAME'].upper(): n for n, a in enumerate(asteroid_list)}
    a1_id = str(int(asteroid_list[name_to_idx[entry['names'][0]]]['ID']))
    a2_id = str(int(asteroid_list[name_to_idx[entry['names'][1]]]['ID']))
    a3_id = str(int(asteroid_list[name_to_idx[entry['names'][2]]]['ID']))

    bodies = [('Earth', '399', '#3498db')]
    if has_flyby:
        fb_body_id = {'mars': '4', 'moon': '301', 'earth': '399'}[fb_name]
        bodies.append((fb_name.title(), fb_body_id, '#e74c3c'))
    bodies += [
        (entry['names'][0].title(), a1_id, '#9b59b6'),
        (entry['names'][1].title(), a2_id, '#2ecc71'),
        (entry['names'][2].title(), a3_id, '#f39c12'),
    ]

    all_ts = np.concatenate([L['ts'] for L in legs])
    all_xs = np.concatenate([L['xs'] for L in legs])

    sample_ts = np.linspace(et_launch, et_arr_3, 300)
    body_orbits = {n: np.array([get_state(b, t)[0] for t in sample_ts])
                   for n, b, _ in bodies}

    fig = plt.figure(figsize=(11, 8), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
    ax.scatter([0], [0], [0], color='#fff200', s=180, edgecolor='#ffae00')
    for name, _, color in bodies:
        pts = body_orbits[name]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.18,
                linewidth=0.8)
    ax.plot(all_xs[:, 0], all_xs[:, 1], all_xs[:, 2], color='#b3b3b3',
            alpha=0.20, linewidth=0.8)
    ax.set_xlabel('X (km)', color='#ddd'); ax.set_ylabel('Y (km)', color='#ddd')
    ax.set_zlabel('Z (km)', color='#ddd')
    ax.tick_params(colors='#888', labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color('#444')
        axis.set_pane_color((0.05, 0.06, 0.08, 1.0))
    AU = 1.496e8
    R = 4.5 * AU
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R*0.4, R*0.4)
    ax.set_box_aspect((1, 1, 0.4))

    body_dots = {n: ax.plot([], [], [], 'o', color=c, markersize=8,
                            markeredgecolor='white', markeredgewidth=0.5)[0]
                 for n, _, c in bodies}
    sc_line, = ax.plot([], [], [], '-', color='#00d4ff', linewidth=2.0)
    sc_dot, = ax.plot([], [], [], 'o', color='#00d4ff', markersize=7,
                      markeredgecolor='white', markeredgewidth=0.7)
    title = ' -> '.join(n.title() for n in entry['names'])
    if has_flyby:
        title += f"  via {fb_name.title()} GA"
    ax.text2D(0.02, 0.97, title, transform=ax.transAxes, color='white',
              fontsize=13, fontweight='bold')
    label = entry['label']
    ax.text2D(0.02, 0.93, label, transform=ax.transAxes, color='#9be7ff',
              fontsize=10)
    date_box = ax.text2D(0.02, 0.05, '', transform=ax.transAxes,
                         color='#ffe066', fontsize=12, family='monospace')

    frame_ts = np.linspace(et_launch, et_arr_3, frames)

    def animate(i):
        et = frame_ts[i]
        for name, bid, _ in bodies:
            r = get_state(bid, et)[0]
            body_dots[name].set_data([r[0]], [r[1]])
            body_dots[name].set_3d_properties([r[2]])
        mask = all_ts <= et
        if mask.any():
            sc_line.set_data(all_xs[mask, 0], all_xs[mask, 1])
            sc_line.set_3d_properties(all_xs[mask, 2])
            j = max(1, min(np.searchsorted(all_ts, et), len(all_ts) - 1))
            t0, t1 = all_ts[j-1], all_ts[j]
            f = (et - t0) / (t1 - t0) if t1 > t0 else 0
            r_sc = all_xs[j-1] + f * (all_xs[j] - all_xs[j-1])
            sc_dot.set_data([r_sc[0]], [r_sc[1]])
            sc_dot.set_3d_properties([r_sc[2]])
        date_box.set_text(spiceypy.et2utc(et, 'C', 0))
        ax.view_init(elev=30, azim=-60 + 60 * i / frames)
        return list(body_dots.values()) + [sc_line, sc_dot, date_box]

    print(f"Rendering {frames} frames -> {out_path}")
    ani = FuncAnimation(fig, animate, frames=frames, interval=1000/fps,
                        blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: rank — rebuild asteroid_tradeoff.csv
# ---------------------------------------------------------------------------

def cmd_rank(args):
    _setup_repo_root()
    from tradeoff import run_tradeoff_v3
    run_tradeoff_v3(args.sbdb_csv, args.tradeoff_csv)
    print(f"Wrote: {args.tradeoff_csv}")


# ---------------------------------------------------------------------------
# Subcommand: animate-asteroids
# ---------------------------------------------------------------------------

def cmd_animate_asteroids(args):
    _setup_repo_root()
    from visualization import graph_asteroids
    asteroid_list = _load_asteroids(args.bsp, args.kernels)
    out = args.out or os.path.join(DEFAULT_RENDER_DIR, 'asteroid_orbits.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    graph_asteroids(asteroid_list, t_duration=args.duration, fps=args.fps,
                    start_date=args.launch_min, end_date=args.launch_max,
                    output_video_name=out)


# ---------------------------------------------------------------------------
# Subcommand: verify — audit a saved trajectory's flyby physics
# ---------------------------------------------------------------------------

def cmd_verify(args):
    """Audit Mars/Moon flyby geometry in a saved result pkl.

    Re-derives v_inf vectors from a fresh Lambert solve and checks whether
    the required turn angle is achievable at the safe periapsis altitude.
    """
    _setup_repo_root()
    from core import audit_flyby_geometry, get_id_from_asteroid_name
    import spiceypy

    pkl_path = args.pkl if os.path.isabs(args.pkl) else os.path.join(
        DEFAULT_PKL_DIR, args.pkl)
    if not os.path.exists(pkl_path):
        sys.exit(f"Not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    entries = _normalize_entries(data)
    if args.rank is not None:
        if args.rank < 1 or args.rank > len(entries):
            sys.exit(f"--rank {args.rank} out of range (1..{len(entries)})")
        target = entries[args.rank - 1]
    elif args.names:
        wanted = tuple(n.upper() for n in args.names)
        target = next((e for e in entries if tuple(e['names']) == wanted), None)
        if target is None:
            sys.exit(f"No entry with names {wanted} in {pkl_path}")
    else:
        target = entries[0]

    asteroid_list = _load_asteroids(args.bsp, args.kernels)
    a1_name = target['names'][0]
    a1_id = str(int(get_id_from_asteroid_name(asteroid_list, a1_name)))

    res = target['result']
    arch = res.get('arch') or res.get('flyby_name') or 'mars'
    if arch == 'direct' or 'et_flyby' not in res:
        print(f"Trajectory has no flyby (architecture={arch}). Nothing to audit.")
        return

    audit = audit_flyby_geometry(res['et_launch'], res['et_flyby'],
                                  res['et_arrive_1'], a1_id, arch)

    print(f"\nFlyby audit — {' -> '.join(target['names'])}  via {arch}")
    print('-' * 60)
    if not audit.get('feasible'):
        print(f"  RESULT: INFEASIBLE  ({audit.get('reason', 'turn exceeds max')})")
    else:
        print(f"  RESULT: FEASIBLE")
    if 'v_inf_in_vec' in audit:
        v_in = audit['v_inf_in_vec']; v_out = audit['v_inf_out_vec']
        print(f"  v_inf_in  vector       : "
              f"[{v_in[0]:+.4f}, {v_in[1]:+.4f}, {v_in[2]:+.4f}] km/s")
        print(f"  v_inf_out vector       : "
              f"[{v_out[0]:+.4f}, {v_out[1]:+.4f}, {v_out[2]:+.4f}] km/s")
        print(f"  |v_inf_in|             : {audit['v_inf_in_kms']:.4f} km/s")
        print(f"  |v_inf_out|            : {audit['v_inf_out_kms']:.4f} km/s")
        print(f"  Energy residual        : {audit['energy_residual_kms']:+.2e} km/s")
        print(f"  Turn angle (required)  : {audit['turn_angle_deg']:.3f}°")
        print(f"  Turn angle (max @ safe): {audit['turn_max_deg']:.3f}°")
        print(f"  Periapsis altitude     : {audit['periapsis_alt_km']:.0f} km "
              f"(min allowed: {audit['safe_periapsis_alt_km']} km)")
        print(f"  {arch.title()} surface radius : "
              f"{audit['body_radius_km']:.1f} km")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _add_kernels_args(p):
    p.add_argument('--bsp', default=DEFAULT_BSP_FOLDER,
                   help='Asteroid BSP folder (default: %(default)s)')
    p.add_argument('--kernels', default=DEFAULT_KERNELS,
                   help='Generic SPICE kernel folder (default: %(default)s)')


def build_parser():
    p = argparse.ArgumentParser(
        prog='main.py', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    # optimize
    o = sub.add_parser('optimize', help='Run trajectory optimization')
    _add_kernels_args(o)
    o.add_argument('--launch-min', default=DEFAULT_LAUNCH_MIN)
    o.add_argument('--launch-max', default=DEFAULT_LAUNCH_MAX)
    o.add_argument('--top-n', type=int, default=50,
                   help='Top-N triplets for fine optimization (default: 50)')
    o.add_argument('--science', type=float, default=None, metavar='ALPHA',
                   help='Science weighting; e.g. 0.7 = 70%% dv + 30%% science')
    o.add_argument('--diverse', action='store_true',
                   help='Require C+S+X/M composition diversity')
    o.add_argument('--feasible', action='store_true',
                   help='Composition-diverse + physical-flyby audit '
                        '(Mars/Moon altitude check). Uses optimize_best_architecture.')
    o.add_argument('--feasible-top-n', type=int, default=25,
                   help='Top-N candidates for the fine DE pass in --feasible '
                        '(default: 25)')
    o.add_argument('--beam', type=int, default=None, metavar='K',
                   help='Use beam search with given beam width K')
    o.add_argument('--pareto', action='store_true',
                   help='Mass-Pareto across 8 propulsion architectures')
    o.add_argument('--pareto-seed', default=None,
                   help='Seed pkl for --pareto (default: results_69ast_ga.pkl)')
    o.add_argument('--flyby', default='mars',
                   choices=['mars', 'moon', 'earth'],
                   help='Flyby body for --pareto (default: mars)')
    o.add_argument('--m-init', type=float, default=1500.0,
                   help='Initial spacecraft mass kg (default: 1500)')
    o.add_argument('--thrust', type=float, default=0.30,
                   help='Continuous thrust N (default: 0.30)')
    o.add_argument('--tradeoff-csv', default=DEFAULT_TRADEOFF)
    o.set_defaults(func=cmd_optimize)

    # list
    l = sub.add_parser('list', help='List saved result pkl files')
    l.set_defaults(func=cmd_list)

    # plot
    pl = sub.add_parser('plot', help='Plot or animate a saved result')
    pl.add_argument('pkl', help='Path to result pkl (or basename in pkl/)')
    pl.add_argument('--rank', type=int, default=None,
                    help='Rank to render (1 = best). Without --rank, lists top entries.')
    pl.add_argument('--names', nargs=3, default=None, metavar=('A', 'B', 'C'),
                    help='Render the entry with these three asteroid names')
    pl.add_argument('--gif', action='store_true',
                    help='Render an animated GIF instead of a static PNG')
    pl.add_argument('--top', type=int, default=10,
                    help='Top entries to list when no --rank/--names given')
    pl.add_argument('--frames', type=int, default=240,
                    help='GIF frame count (default: 240)')
    pl.add_argument('--fps', type=int, default=24)
    pl.add_argument('--out-dir', default=None,
                    help='Output dir (default: Renders/)')
    _add_kernels_args(pl)
    pl.set_defaults(func=cmd_plot)

    # rank
    r = sub.add_parser('rank', help='Rebuild asteroid_tradeoff.csv')
    r.add_argument('--sbdb-csv', default='sbdb_query_results.csv')
    r.add_argument('--tradeoff-csv', default=DEFAULT_TRADEOFF)
    r.set_defaults(func=cmd_rank)

    # animate-asteroids
    a = sub.add_parser('animate-asteroids',
                       help='Animate notable asteroid orbits as MP4')
    _add_kernels_args(a)
    a.add_argument('--launch-min', default=DEFAULT_LAUNCH_MIN)
    a.add_argument('--launch-max', default=DEFAULT_LAUNCH_MAX)
    a.add_argument('--duration', type=int, default=8,
                   help='Video length in seconds (default: 8)')
    a.add_argument('--fps', type=int, default=80)
    a.add_argument('--out', default=None)
    a.set_defaults(func=cmd_animate_asteroids)

    # verify
    v = sub.add_parser('verify',
                       help='Audit flyby physics of a saved trajectory')
    v.add_argument('pkl', help='Path to result pkl (or basename in pkl/)')
    v.add_argument('--rank', type=int, default=None,
                   help='Rank to audit (1=best). Default: rank 1.')
    v.add_argument('--names', nargs=3, default=None,
                   metavar=('A', 'B', 'C'),
                   help='Audit the entry with these three asteroid names')
    _add_kernels_args(v)
    v.set_defaults(func=cmd_verify)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
