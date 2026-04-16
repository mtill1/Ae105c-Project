#!/usr/bin/env python3
"""Science-priority optimization: 70% science + 30% delta-v, with gravity assists.

Uses enhanced science scoring (35% sci potential + 25% mass + 25% radius + 15% low-inc).
Composition-diverse (C+S+X/M), parallel across all cores.
"""
import sys, os, time, pickle
import multiprocessing as mp
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, CODE_DIR)

def _init_worker():
    """Each worker loads its own SPICE kernels."""
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


def _eval_coarse(args):
    i, j, k, a1_id, a2_id, a3_id, launch_dates = args
    from optimization import optimize_times_quick, score_paths_flyby
    from core import YEAR

    dv_direct = optimize_times_quick(a1_id, a2_id, a3_id, launch_dates, 0, 0, 0)

    dv_mars = 1e3
    for lf in [0.2, 0.5, 0.8]:
        for mt in [0.8, 1.5]:
            for tof in [2.0, 3.5]:
                x = np.array([lf*(launch_dates[1]-launch_dates[0])/YEAR, mt, tof, 0.4, tof, 0.4, tof])
                dv = score_paths_flyby(x, a1_id, a2_id, a3_id, launch_dates, 'mars', 0, 0, 0, 0)
                if dv < dv_mars:
                    dv_mars = dv

    best_dv = min(dv_direct, dv_mars)
    best_arch = 'direct' if dv_direct <= dv_mars else 'mars'
    return (i, j, k, best_dv, best_arch)


def _eval_fine(args):
    i, j, k, a1_id, a2_id, a3_id, launch_dates = args
    from optimization import optimize_best_architecture
    result, best_arch = optimize_best_architecture(a1_id, a2_id, a3_id, launch_dates, 0, 0, 0, quick=False)
    result['architecture'] = best_arch
    return (i, j, k, result)


def build_science_scores(tradeoff_csv):
    """Build enhanced science scores with more spread than Total_WeightedScore.
    
    35% science potential + 25% mass + 25% radius + 15% low-inclination bonus.
    Returns dict: UPPERCASE name -> score (range ~5-8).
    """
    df = pd.read_csv(tradeoff_csv)
    scores = {}
    for _, row in df.iterrows():
        raw = str(row['Name_DecRadius']).split('(')[0].strip()
        parts = raw.split()
        if parts and parts[0].replace('.', '').isdigit():
            name = ' '.join(parts[1:]).upper()
        else:
            name = raw.upper()
        sci = float(row['SciPotential_Score_1to10'])
        mass = float(row['Sub_Mass_Score'])
        radius = float(row['Sub_Radius_Score'])
        inc = float(row['Sub_Inc_Score'])
        scores[name] = 0.35 * sci + 0.25 * mass + 0.25 * radius + 0.15 * inc
    return scores


if __name__ == '__main__':
    import spiceypy
    from tqdm import tqdm
    from core import load_kernels
    from optimization import load_composition_map, _triplet_has_diverse_composition

    project_root = os.path.join(SCRIPT_DIR, '..', '..')

    ALPHA = 0.3  # 30% delta-v, 70% science
    TOP_N = 50

    print(f"{'='*70}")
    print(f"SCIENCE-PRIORITY OPTIMIZATION")
    print(f"  alpha={ALPHA} (70% science, 30% delta-v)")
    print(f"  Gravity assists: Moon + Mars")
    print(f"  Composition: C + S + X/M required")
    print(f"{'='*70}\n")

    print("Loading SPICE kernels...")
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))
    print(f"Loaded {len(asteroid_list)} asteroids.")

    comp_map = load_composition_map(os.path.join(project_root, 'asteroid_tradeoff.csv'))
    science_scores = build_science_scores(os.path.join(project_root, 'asteroid_tradeoff.csv'))
    required = {"C", "S", "X/M"}

    # Show top science asteroids
    print("\nTop 10 science scores in our pool:")
    bsp_scores = [(a['NAME'], science_scores.get(a['NAME'].upper(), 0),
                   comp_map.get(a['NAME'].upper(), '?')) for a in asteroid_list]
    bsp_scores.sort(key=lambda x: -x[1])
    for name, score, comp in bsp_scores[:10]:
        print(f"  {name:20s} [{comp:3s}]  sci={score:.2f}")

    et_min = spiceypy.str2et("Jan 1 12:00:00 UTC 2027")
    et_max = spiceypy.str2et("Dec 31 12:00:00 UTC 2035")
    launch_dates = [et_min, et_max]

    num = len(asteroid_list)
    tasks = []
    for i in range(num):
        for j in range(num):
            for k in range(num):
                if len({asteroid_list[i]['ID'], asteroid_list[j]['ID'], asteroid_list[k]['ID']}) < 3:
                    continue
                if not _triplet_has_diverse_composition(i, j, k, asteroid_list, comp_map, required):
                    continue
                a1_id = str(int(asteroid_list[i]['ID']))
                a2_id = str(int(asteroid_list[j]['ID']))
                a3_id = str(int(asteroid_list[k]['ID']))
                tasks.append((i, j, k, a1_id, a2_id, a3_id, launch_dates))

    N_WORKERS = mp.cpu_count()
    print(f"\nDiverse triplets: {len(tasks):,}")
    print(f"Workers: {N_WORKERS}")
    print()

    t_start = time.time()

    # PASS 1: Parallel coarse
    print(f"Pass 1: Parallel coarse screening ({len(tasks):,} triplets)...")
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        coarse = list(tqdm(pool.imap(_eval_coarse, tasks, chunksize=50),
                           total=len(tasks), desc="Coarse"))

    # Apply science weighting: alpha * dv + (1-alpha) * (30 - sci_sum)
    print(f"\nApplying science weighting (alpha={ALPHA})...")
    scored = []
    for i, j, k, dv, arch in coarse:
        sci_sum = sum(science_scores.get(asteroid_list[x]['NAME'].upper(), 5.0) for x in [i, j, k])
        combined = ALPHA * dv + (1 - ALPHA) * (30 - sci_sum)
        scored.append((i, j, k, dv, arch, sci_sum, combined))

    scored.sort(key=lambda x: x[6])  # sort by combined score
    top = scored[:TOP_N]

    print(f"\nTop 5 by combined score:")
    for i, j, k, dv, arch, sci, comb in top[:5]:
        n1, n2, n3 = asteroid_list[i]['NAME'], asteroid_list[j]['NAME'], asteroid_list[k]['NAME']
        print(f"  {n1}->{n2}->{n3}  dv={dv:.1f}  sci={sci:.1f}  combined={comb:.2f}  [{arch}]")

    t_coarse = time.time() - t_start

    # PASS 2: Parallel fine
    print(f"\nPass 2: Parallel fine optimization (top {TOP_N})...")
    fine_tasks = [(i, j, k,
                   str(int(asteroid_list[i]['ID'])),
                   str(int(asteroid_list[j]['ID'])),
                   str(int(asteroid_list[k]['ID'])),
                   launch_dates) for i, j, k, _, _, _, _ in top]

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = list(tqdm(pool.imap(_eval_fine, fine_tasks, chunksize=2),
                            total=len(fine_tasks), desc="Fine"))

    # Re-score fine results with science weighting
    final = []
    for i, j, k, res in results:
        sci_sum = sum(science_scores.get(asteroid_list[x]['NAME'].upper(), 5.0) for x in [i, j, k])
        combined = ALPHA * res['delta_v_total'] + (1 - ALPHA) * (30 - sci_sum)
        res['science_sum'] = sci_sum
        res['combined_score'] = combined
        final.append((i, j, k, res))

    final.sort(key=lambda x: x[3]['combined_score'])

    t_total = time.time() - t_start

    print(f"\n{'='*95}")
    print(f"TOP 15 SCIENCE-PRIORITY PATHS (alpha={ALPHA}: 70% science, 30% dv)")
    print(f"{'='*95}")
    for rank, (i, j, k, res) in enumerate(final[:15], 1):
        n1, n2, n3 = asteroid_list[i]['NAME'], asteroid_list[j]['NAME'], asteroid_list[k]['NAME']
        dv = res['delta_v_total']
        sci = res['science_sum']
        comb = res['combined_score']
        ldv = np.linalg.norm(res['delta_v_launch']) if len(res['delta_v_launch']) > 0 else 0
        arch = res.get('architecture', 'direct')
        c1 = comp_map.get(n1.upper(), '?')
        c2 = comp_map.get(n2.upper(), '?')
        c3 = comp_map.get(n3.upper(), '?')
        tag = f" [{arch.upper()}]" if arch != 'direct' else ""
        print(f"  #{rank:2d}: {n1:15s}[{c1:3s}] -> {n2:15s}[{c2:3s}] -> {n3:15s}[{c3:3s}]"
              f"  |  dv={dv:5.1f}  sci={sci:5.1f}  score={comb:5.2f}{tag}")

    print(f"\nCoarse: {t_coarse:.1f}s | Fine: {t_total-t_coarse:.1f}s | Total: {t_total:.1f}s")

    output = os.path.join(project_root, f"optimal_asteroid_paths/results_science_priority_{int(time.time())}.pkl")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(final, f)
    print(f"Saved: {output}")
