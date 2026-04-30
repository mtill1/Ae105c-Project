#!/usr/bin/env python3
"""Single-triplet mass+Δv optimization on GCP.

Runs mass-objective DE on all 8 propulsion architectures for one specified
triplet, parallelized across architectures (one DE worker per arch). Then
verifies the top 3 with the real Sims-Flanagan LT solver.

Usage: TRIPLET=PARTHENOPE,PSYCHE,THEMIS python3 run_single_triplet.py
"""
import sys
import os
import time
import pickle
import warnings
import multiprocessing as mp

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


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


def _eval_arch(args):
    """Worker: optimize a single (arch, m_revs, seed) and return the result."""
    arch, m_revs, seed, ids, launch_range, flyby_name, m_init_kg, thrust_N = args
    from mass_optimization import optimize_for_architecture
    try:
        out = optimize_for_architecture(
            *ids, launch_range, flyby_name, arch,
            m_revs=m_revs, seed=seed,
            m_init_kg=m_init_kg, thrust_N=thrust_N,
            maxiter=250, popsize=18)
        return arch, m_revs, seed, out
    except Exception as e:
        return arch, m_revs, seed, {'feasible': False, 'error': str(e)}


if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    triplet_str = os.environ.get('TRIPLET', 'PARTHENOPE,PSYCHE,THEMIS')
    triplet = tuple(s.strip().upper() for s in triplet_str.split(','))
    print(f'Triplet: {" -> ".join(triplet)}', flush=True)

    from core import load_kernels, get_id_from_asteroid_name
    import spiceypy
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))
    ids = [str(int(get_id_from_asteroid_name(asteroid_list, n))) for n in triplet]

    et_min = spiceypy.str2et('Jan 1 12:00:00 UTC 2027')
    et_max = spiceypy.str2et('Dec 31 12:00:00 UTC 2035')
    launch_range = [et_min, et_max]

    M_INIT_KG = 1500.0
    THRUST_N  = 0.30
    FLYBY     = 'mars'
    ARCHS = ['CCC','CCE','CEC','CEE','ECC','ECE','EEC','EEE']
    M_REVS_OPTIONS = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)]
    SEEDS = (42, 137, 314)

    # Build the full task list: every (arch, m_revs, seed) combo as a separate task
    tasks = []
    for arch in ARCHS:
        for m_revs in M_REVS_OPTIONS:
            for seed in SEEDS:
                tasks.append((arch, m_revs, seed, ids, launch_range,
                              FLYBY, M_INIT_KG, THRUST_N))

    N_WORKERS = min(mp.cpu_count(), len(tasks))
    print(f'\nMass + Δv joint optimization (single triplet)', flush=True)
    print(f'  Architectures : {len(ARCHS)}', flush=True)
    print(f'  m-revs combos : {len(M_REVS_OPTIONS)}', flush=True)
    print(f'  Seeds         : {len(SEEDS)}', flush=True)
    print(f'  Total DE runs : {len(tasks)} (each fully parallel)', flush=True)
    print(f'  Workers       : {N_WORKERS}', flush=True)
    print(f'  DE budget     : maxiter=250, popsize=18 (slightly stronger than full sweep)', flush=True)
    print(flush=True)
    print('-' * 90, flush=True)
    print(f'{"arch":4s}  {"m_revs":12s}  {"seed":>4s}  '
          f'{"feasible":>9s}  {"m_final":>8s}  {"dv_eq":>6s}  t(s)', flush=True)
    print('-' * 90, flush=True)

    t_start = time.time()
    raw_results = []
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for arch, m_revs, seed, out in pool.imap_unordered(_eval_arch, tasks, chunksize=1):
            raw_results.append((arch, m_revs, seed, out))
            elapsed = time.time() - t_start
            if 'error' in out:
                print(f'  {arch:3s}  {str(m_revs):12s}  {seed:>4d}  '
                      f'  ERROR: {out["error"][:40]}', flush=True)
            elif not out['feasible']:
                print(f'  {arch:3s}  {str(m_revs):12s}  {seed:>4d}  '
                      f'  infeas      —       —    —', flush=True)
            else:
                print(f'  {arch:3s}  {str(m_revs):12s}  {seed:>4d}  '
                      f'  {"OK":>9s}  {out["m_final_kg"]:8.1f}  '
                      f'{out["dv_equiv_kms"]:6.2f}  {elapsed:4.0f}', flush=True)

    print('-' * 90, flush=True)

    # Pick the best per architecture, then verify top 3 with real LT solver
    by_arch = {}
    for arch, m_revs, seed, out in raw_results:
        if not out.get('feasible'): continue
        if arch not in by_arch or out['m_final_kg'] > by_arch[arch]['m_final_kg']:
            by_arch[arch] = out

    print(f'\nBest per architecture (surrogate):', flush=True)
    print(f'{"arch":4s}  {"m_final":>8s}  {"dv_eq":>6s}', flush=True)
    for arch in sorted(by_arch, key=lambda a: -by_arch[a]['m_final_kg']):
        r = by_arch[arch]
        print(f'  {arch:3s}  {r["m_final_kg"]:8.1f}  {r["dv_equiv_kms"]:6.2f}', flush=True)

    # Verify the top 3 archs with the real LT solver
    from mass_optimization import verify_with_full_lt
    top_archs = sorted(by_arch, key=lambda a: -by_arch[a]['m_final_kg'])[:3]
    print(f'\nVerifying top 3 with real Sims-Flanagan LT solver...', flush=True)
    verified = []
    for arch in top_archs:
        t0 = time.time()
        try:
            v = verify_with_full_lt(by_arch[arch], *ids,
                                     m_init_kg=M_INIT_KG, thrust_N=THRUST_N)
            verified.append(v)
            if v.get('verified_feasible'):
                print(f'  {arch}: surrogate {by_arch[arch]["m_final_kg"]:.0f} kg '
                      f'-> verified {v["verified_m_final_kg"]:.0f} kg '
                      f'(dv_eq {v["verified_dv_equiv_kms"]:.2f}) '
                      f'[{time.time()-t0:.1f}s]', flush=True)
            else:
                print(f'  {arch}: VERIFICATION FAILED [{time.time()-t0:.1f}s]',
                      flush=True)
        except Exception as e:
            print(f'  {arch}: error {e}', flush=True)

    # Pick best verified
    best_verified = None
    for v in verified:
        if v.get('verified_feasible'):
            m = v['verified_m_final_kg']
            if best_verified is None or m > best_verified['verified_m_final_kg']:
                best_verified = v

    if best_verified:
        print(f'\n{"="*90}', flush=True)
        print(f' BEST: {" -> ".join(triplet)} via {FLYBY} flyby, arch={best_verified["arch_code"]}',
              flush=True)
        print(f'{"="*90}', flush=True)
        print(f'  Verified delivered mass: {best_verified["verified_m_final_kg"]:.1f} kg / 1500 kg',
              flush=True)
        print(f'  Verified Δv-equivalent : {best_verified["verified_dv_equiv_kms"]:.2f} km/s',
              flush=True)
        print(f'  Propellant fraction    : '
              f'{(1 - best_verified["verified_m_final_kg"]/1500)*100:.1f}%', flush=True)
        print(f'  Launch                 : {spiceypy.et2utc(best_verified["et_launch"], "C", 0)}',
              flush=True)
        print(f'  Mars flyby             : {spiceypy.et2utc(best_verified["et_flyby"], "C", 0)}',
              flush=True)
        print(f'  Arrive {triplet[0]:11s}: {spiceypy.et2utc(best_verified["et_arrive_1"], "C", 0)}',
              flush=True)
        print(f'  Depart {triplet[0]:11s}: {spiceypy.et2utc(best_verified["et_stay_1"], "C", 0)}',
              flush=True)
        print(f'  Arrive {triplet[1]:11s}: {spiceypy.et2utc(best_verified["et_arrive_2"], "C", 0)}',
              flush=True)
        print(f'  Depart {triplet[1]:11s}: {spiceypy.et2utc(best_verified["et_stay_2"], "C", 0)}',
              flush=True)
        print(f'  Arrive {triplet[2]:11s}: {spiceypy.et2utc(best_verified["et_arrive_3"], "C", 0)}',
              flush=True)
        from core import YEAR
        dur = (best_verified['et_arrive_3'] - best_verified['et_launch']) / YEAR
        print(f'  Mission duration       : {dur:.2f} yr', flush=True)

    # Save
    out_pkl = os.path.join(project_root,
        f'optimal_asteroid_paths/pkl/single_{triplet[0]}_{triplet[1]}_{triplet[2]}.pkl')
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'triplet': triplet,
            'ids': ids,
            'm_init_kg': M_INIT_KG, 'thrust_N': THRUST_N, 'flyby': FLYBY,
            'all_archs_best': by_arch,
            'verified_top3': verified,
            'best_verified': best_verified,
            'raw_results': raw_results,
        }, f)
    print(f'\nSaved: {out_pkl}', flush=True)
