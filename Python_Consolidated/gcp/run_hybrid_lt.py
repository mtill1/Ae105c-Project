#!/usr/bin/env python3
"""Run hybrid impulsive+LT scoring on the top impulsive triplets.

Loads the existing dv-min pickle (top 50 triplets), evaluates all 4
architectures {CC, CE, EC, EE} per triplet via Tsiolkovsky chain + real
low-thrust leg optimization, and saves the hybrid-ranked results.

Prints a progress line every triplet so the user can tail the log.
"""
import sys, os, time, pickle
import multiprocessing as mp
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


def _init_worker():
    import spiceypy, glob
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    gk = os.path.join(project_root, 'generic_kernels')
    spiceypy.furnsh(os.path.join(gk, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'spk', 'planets', 'de430.bsp'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(gk, 'pck', 'pck00010.tpc'))
    for bsp in sorted(glob.glob(os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs', '*.bsp'))):
        spiceypy.furnsh(bsp)


def _eval_one(args):
    rank, i, j, k, res, a_ids, names = args
    from hybrid_mission import evaluate_hybrid
    t0 = time.time()
    try:
        out = evaluate_hybrid(res, a_ids, verbose=False)
        out['_triplet_idx'] = (i, j, k)
        out['_names']       = names
        out['_elapsed_s']   = time.time() - t0
        out['_orig_rank']   = rank
        out['_dv_total_impulsive'] = res['delta_v_total']
        return out
    except Exception as e:
        return {'_triplet_idx': (i, j, k), '_names': names,
                '_orig_rank': rank, '_error': str(e),
                '_elapsed_s': time.time() - t0}


if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')

    # Input: the latest impulsive results
    input_pkl = os.path.join(project_root,
        'optimal_asteroid_paths/pkl/results_69ast_ega_real.pkl')
    print(f"Loading impulsive results: {input_pkl}", flush=True)
    with open(input_pkl, 'rb') as f:
        impulsive_results = pickle.load(f)
    print(f"Loaded {len(impulsive_results)} impulsive candidates.", flush=True)

    # Load asteroid catalog
    from core import load_kernels
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))

    tasks = []
    for rank, (i, j, k, res) in enumerate(impulsive_results, 1):
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        ids   = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        tasks.append((rank, i, j, k, res, ids, names))

    N_WORKERS = min(mp.cpu_count(), len(tasks))
    print(f"\nEvaluating {len(tasks)} triplets × 4 architectures each "
          f"on {N_WORKERS} workers...", flush=True)
    print("-" * 95, flush=True)
    print(f"{'#':>3}  {'triplet':35s}  {'imp_dv':>7s}  {'best':4s}  "
          f"{'m_final':>8s}  {'gain_kg':>7s}  t(s)", flush=True)
    print("-" * 95, flush=True)

    t_start = time.time()
    results = []

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for out in pool.imap_unordered(_eval_one, tasks, chunksize=1):
            results.append(out)
            if '_error' in out:
                print(f"  #{out['_orig_rank']:<3d} ERROR: {out['_error'][:60]}",
                      flush=True)
                continue
            n = out['_names']
            tag = f"{n[0][:10]}->{n[1][:10]}->{n[2][:10]}"
            print(f"  #{out['_orig_rank']:<3d} {tag:35s}  "
                  f"{out['_dv_total_impulsive']:7.2f}  {out['best_arch']:4s}  "
                  f"{out['m_best_kg']:8.1f}  {out['improvement_kg']:+7.1f}  "
                  f"{out['_elapsed_s']:4.1f}", flush=True)

    t_total = time.time() - t_start
    print("-" * 95, flush=True)
    print(f"Done in {t_total:.1f}s  ({t_total/60:.1f} min)", flush=True)

    # Rank by m_final (best) and print top 15
    valid = [r for r in results if '_error' not in r]
    valid.sort(key=lambda r: -r['m_best_kg'])

    print("\n" + "=" * 95, flush=True)
    print(" HYBRID LT TOP 15 by FINAL DELIVERED MASS", flush=True)
    print("=" * 95, flush=True)
    print(f"{'rank':>4s} {'triplet':35s} {'best':4s} {'m_final':>8s} "
          f"{'prop%':>6s} {'imp_dv':>6s} {'CC_mass':>7s}", flush=True)
    for rank, out in enumerate(valid[:15], 1):
        n = out['_names']
        tag = f"{n[0][:10]}->{n[1][:10]}->{n[2][:10]}"
        print(f"  {rank:2d}  {tag:35s} {out['best_arch']:4s} "
              f"{out['m_best_kg']:8.1f} "
              f"{out['prop_fraction_best']*100:5.1f}% "
              f"{out['_dv_total_impulsive']:6.2f} "
              f"{out['m_baseline_CC_kg']:7.1f}", flush=True)

    # Save
    out_path = os.path.join(project_root,
        f'optimal_asteroid_paths/results_hybrid_lt_{int(time.time())}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}", flush=True)
