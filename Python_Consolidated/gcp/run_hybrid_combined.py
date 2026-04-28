#!/usr/bin/env python3
"""Hybrid LT ranking with combined Δv-equivalent + science objective.

Takes the UNION of dv-min top-50 and science-priority top-50 triplets,
runs hybrid LT evaluation on each, and ranks by:

    score = alpha * dv_equiv_hybrid + (1-alpha) * (30 - science_sum)

where dv_equiv_hybrid = -Isp_chem * g0 * ln(m_best / m_init) / 1000 (km/s)
— i.e., the chemical-Isp-equivalent Δv needed to match the hybrid mission's
delivered mass. alpha = 0.3 matches prior science-priority work.
"""
import sys, os, time, pickle
import multiprocessing as mp
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)


ALPHA = 0.3  # 30% cost, 70% science (matches prior science_priority runs)


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
    rank, idx, res, a_ids, names, sci_sum = args
    from hybrid_mission import evaluate_hybrid
    from lowthrust import ISP_CHEM, G0
    t0 = time.time()
    try:
        out = evaluate_hybrid(res, a_ids, verbose=False)
        m_best = out['m_best_kg']; m_init = out['m_init_kg']
        dv_eq = -ISP_CHEM * G0 * np.log(m_best / m_init) / 1000.0  # km/s
        score = ALPHA * dv_eq + (1 - ALPHA) * (30.0 - sci_sum)
        out['_triplet_idx']        = idx
        out['_names']              = names
        out['_elapsed_s']          = time.time() - t0
        out['_orig_rank']          = rank
        out['_dv_total_impulsive'] = res['delta_v_total']
        out['_dv_equiv_hybrid']    = dv_eq
        out['_science_sum']        = sci_sum
        out['_combined_score']     = score
        return out
    except Exception as e:
        return {'_triplet_idx': idx, '_names': names, '_orig_rank': rank,
                '_error': str(e), '_elapsed_s': time.time() - t0}


def build_science_scores(csv_path):
    df = pd.read_csv(csv_path)
    scores = {}
    for _, row in df.iterrows():
        raw = str(row['Name_DecRadius']).split('(')[0].strip()
        parts = raw.split()
        name = ' '.join(parts[1:]).upper() if parts and parts[0].replace('.','').isdigit() else raw.upper()
        sci    = float(row['SciPotential_Score_1to10'])
        inc    = float(row['Sub_Inc_Score'])
        radius = float(row['Sub_Radius_Score'])
        mass   = float(row['Sub_Mass_Score'])
        ecc    = float(row['Sub_Ecc_Score'])
        rot    = float(row['Sub_RotPer_Score'])
        sma    = float(row['Sub_SMA_Score'])
        scores[name] = (0.214*sci + 0.214*inc + 0.186*radius + 0.171*mass
                        + 0.100*ecc + 0.071*rot + 0.043*sma)
    return scores


if __name__ == '__main__':
    project_root = os.path.join(SCRIPT_DIR, '..', '..')
    pkl_dir = os.path.join(project_root, 'optimal_asteroid_paths/pkl')

    # Load both input pickles
    dv_path  = os.path.join(pkl_dir, 'results_69ast_ega_real.pkl')
    sci_path = os.path.join(pkl_dir, 'results_science_priority_ega_real.pkl')
    print(f"Loading  dv-min top: {dv_path}",  flush=True)
    print(f"Loading science top: {sci_path}", flush=True)
    with open(dv_path,  'rb') as f: dv_results  = pickle.load(f)
    with open(sci_path, 'rb') as f: sci_results = pickle.load(f)

    # Asteroid list & science scores
    from core import load_kernels
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))
    sci_scores = build_science_scores(
        os.path.join(project_root, 'asteroid_tradeoff.csv'))

    # Union of triplets, keyed by (i,j,k), with original res dict preserved
    unique = {}
    for src_tag, src_list in [('dv', dv_results), ('sci', sci_results)]:
        for rank, (i, j, k, res) in enumerate(src_list, 1):
            key = (i, j, k)
            if key not in unique:
                unique[key] = {'i':i, 'j':j, 'k':k, 'res':res,
                               'src': src_tag, 'src_rank': rank}

    print(f"\nUnion pool size: {len(unique)} unique triplets "
          f"(dv={len(dv_results)}, sci={len(sci_results)}, "
          f"overlap={len(dv_results)+len(sci_results)-len(unique)})",
          flush=True)

    # Build task list with pre-computed science sums
    tasks = []
    for rank, (key, entry) in enumerate(unique.items(), 1):
        i, j, k = key
        names = [asteroid_list[x]['NAME'] for x in (i, j, k)]
        ids   = [str(int(asteroid_list[x]['ID'])) for x in (i, j, k)]
        sci_sum = sum(sci_scores.get(n.upper(), 5.0) for n in names)
        tasks.append((rank, key, entry['res'], ids, names, sci_sum))

    N_WORKERS = min(mp.cpu_count(), len(tasks))
    print(f"\nRunning hybrid-LT scoring on {len(tasks)} triplets × 4 archs each "
          f"({N_WORKERS} workers)...", flush=True)
    print(f"Objective: score = {ALPHA:.1f} * dv_equiv_hybrid + "
          f"{1-ALPHA:.1f} * (30 - sci_sum)\n", flush=True)
    print("-" * 115, flush=True)
    print(f"{'#':>3}  {'triplet':37s}  {'arch':4s}  {'m_kg':>7s}  "
          f"{'dv_eq':>6s}  {'sci':>5s}  {'score':>5s}  t(s)", flush=True)
    print("-" * 115, flush=True)

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
            tag = f"{n[0][:11]}->{n[1][:11]}->{n[2][:11]}"
            print(f"  #{out['_orig_rank']:<3d} {tag:37s}  {out['best_arch']:4s}  "
                  f"{out['m_best_kg']:7.1f}  {out['_dv_equiv_hybrid']:6.2f}  "
                  f"{out['_science_sum']:5.2f}  {out['_combined_score']:5.2f}  "
                  f"{out['_elapsed_s']:4.1f}", flush=True)

    t_total = time.time() - t_start
    print("-" * 115, flush=True)
    print(f"Done in {t_total:.1f}s  ({t_total/60:.1f} min)", flush=True)

    valid = [r for r in results if '_error' not in r]

    def _print_top(title, sort_key, fmt, hdr):
        print("\n" + "=" * 115, flush=True)
        print(f" {title}", flush=True)
        print("=" * 115, flush=True)
        print(hdr, flush=True)
        ordered = sorted(valid, key=sort_key)
        for rank, o in enumerate(ordered[:15], 1):
            print(fmt(rank, o), flush=True)

    _print_top(
        "TOP 15 BY COMBINED (0.3 dv_equiv + 0.7 × (30 − science))  ← NEW RANKING",
        lambda o: o['_combined_score'],
        lambda r, o: (f"  {r:2d}  {o['_names'][0][:11]:11s}->{o['_names'][1][:11]:11s}->{o['_names'][2][:11]:11s}  "
                      f"{o['best_arch']:4s}  m={o['m_best_kg']:6.1f}  "
                      f"dv_eq={o['_dv_equiv_hybrid']:5.2f}  "
                      f"sci={o['_science_sum']:5.2f}  score={o['_combined_score']:5.2f}"),
        f"{'rank':>4s}  {'triplet':37s}  {'arch':4s}  {'mass':>9s}  {'dv_eq':>8s}  {'sci':>8s}  {'score':>9s}"
    )
    _print_top(
        "TOP 15 BY PURE HYBRID MASS (all weight on final delivered mass)",
        lambda o: -o['m_best_kg'],
        lambda r, o: (f"  {r:2d}  {o['_names'][0][:11]:11s}->{o['_names'][1][:11]:11s}->{o['_names'][2][:11]:11s}  "
                      f"{o['best_arch']:4s}  m={o['m_best_kg']:6.1f}  "
                      f"dv_eq={o['_dv_equiv_hybrid']:5.2f}  "
                      f"sci={o['_science_sum']:5.2f}"),
        f"{'rank':>4s}  {'triplet':37s}  {'arch':4s}  {'mass':>9s}  {'dv_eq':>8s}  {'sci':>8s}"
    )
    _print_top(
        "TOP 15 BY PURE SCIENCE (all weight on science_sum)",
        lambda o: -o['_science_sum'],
        lambda r, o: (f"  {r:2d}  {o['_names'][0][:11]:11s}->{o['_names'][1][:11]:11s}->{o['_names'][2][:11]:11s}  "
                      f"{o['best_arch']:4s}  m={o['m_best_kg']:6.1f}  "
                      f"sci={o['_science_sum']:5.2f}  score={o['_combined_score']:5.2f}"),
        f"{'rank':>4s}  {'triplet':37s}  {'arch':4s}  {'mass':>9s}  {'sci':>8s}  {'score':>9s}"
    )

    out_path = os.path.join(project_root,
        f'optimal_asteroid_paths/results_hybrid_combined_{int(time.time())}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'alpha': ALPHA, 'results': results}, f)
    print(f"\nSaved: {out_path}", flush=True)
