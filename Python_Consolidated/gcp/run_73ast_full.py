#!/usr/bin/env python3
"""73-asteroid fully parallel gravity-assist optimization on c2-standard-60.

Coarse pass: parallel across 60 cores
Fine pass: ALSO parallel across 60 cores (50 triplets all at once)
"""
import sys, os, time, pickle
import multiprocessing as mp
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

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
    """Coarse: try direct + Mars flyby + Earth flyby, return best."""
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

    dv_earth = 1e3
    for lf in [0.1, 0.4, 0.7]:
        for et_tof in [1.2, 1.8, 2.5]:       # Earth-return loop TOF (years)
            for tof in [2.0, 3.5]:
                x = np.array([lf*(launch_dates[1]-launch_dates[0])/YEAR, et_tof, tof, 0.4, tof, 0.4, tof])
                dv = score_paths_flyby(x, a1_id, a2_id, a3_id, launch_dates, 'earth', 0, 0, 0, 0)
                if dv < dv_earth:
                    dv_earth = dv

    candidates = {'direct': dv_direct, 'mars': dv_mars, 'earth': dv_earth}
    best_arch = min(candidates, key=candidates.get)
    return (i, j, k, candidates[best_arch], best_arch)


def _eval_fine(args):
    """Fine: full optimization with all 3 architectures."""
    i, j, k, a1_id, a2_id, a3_id, launch_dates = args
    from optimization import optimize_best_architecture
    result, best_arch = optimize_best_architecture(a1_id, a2_id, a3_id, launch_dates, 0, 0, 0, quick=False)
    result['architecture'] = best_arch
    return (i, j, k, result)


if __name__ == '__main__':
    import spiceypy
    from tqdm import tqdm
    from core import load_kernels
    from optimization import load_composition_map, _triplet_has_diverse_composition

    project_root = os.path.join(SCRIPT_DIR, '..', '..')

    print("Loading SPICE kernels...")
    asteroid_list = load_kernels(
        os.path.join(project_root, 'NOTABLE_ASTEROID_BSPs'),
        os.path.join(project_root, 'generic_kernels'))
    print(f"Loaded {len(asteroid_list)} asteroids.")

    comp_map = load_composition_map(os.path.join(project_root, 'asteroid_tradeoff.csv'))
    required = {"C", "S", "X/M"}

    counts = {}
    for a in asteroid_list:
        cls = comp_map.get(a['NAME'].upper(), 'Unknown')
        counts[cls] = counts.get(cls, 0) + 1
    print(f"Compositions: {counts}")

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
    TOP_N = 50

    print(f"\nDiverse triplets: {len(tasks):,}")
    print(f"Workers: {N_WORKERS}")
    print(f"Architecture: direct + Mars + Earth flyby (coarse), all 4 (fine)")
    print()

    t_start = time.time()

    # ===== PASS 1: Parallel coarse =====
    print(f"Pass 1: Parallel coarse screening ({len(tasks):,} triplets)...")
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        coarse = list(tqdm(pool.imap(_eval_coarse, tasks, chunksize=50),
                           total=len(tasks), desc="Coarse"))

    coarse.sort(key=lambda x: x[3])
    top = coarse[:TOP_N]
    t_coarse = time.time() - t_start

    arch_counts = {}
    for _, _, _, _, a in top:
        arch_counts[a] = arch_counts.get(a, 0) + 1
    print(f"\nCoarse done in {t_coarse:.1f}s. Top {TOP_N} architectures: {arch_counts}")

    # ===== PASS 2: Parallel fine =====
    print(f"\nPass 2: Parallel fine optimization (top {TOP_N}, all 3 architectures)...")
    fine_tasks = [(i, j, k,
                   str(int(asteroid_list[i]['ID'])),
                   str(int(asteroid_list[j]['ID'])),
                   str(int(asteroid_list[k]['ID'])),
                   launch_dates) for i, j, k, _, _ in top]

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = list(tqdm(pool.imap(_eval_fine, fine_tasks, chunksize=2),
                            total=len(fine_tasks), desc="Fine"))

    results.sort(key=lambda x: x[3]['delta_v_total'])
    t_total = time.time() - t_start

    print(f"\n{'='*90}")
    print(f"TOP 15 PATHS — 73 ASTEROIDS WITH GRAVITY ASSIST")
    print(f"{'='*90}")
    for rank, (i, j, k, res) in enumerate(results[:15], 1):
        n1, n2, n3 = asteroid_list[i]['NAME'], asteroid_list[j]['NAME'], asteroid_list[k]['NAME']
        dv = res['delta_v_total']
        ldv = np.linalg.norm(res['delta_v_launch']) if len(res['delta_v_launch']) > 0 else 0
        arch = res.get('architecture', 'direct')
        c1 = comp_map.get(n1.upper(), '?')
        c2 = comp_map.get(n2.upper(), '?')
        c3 = comp_map.get(n3.upper(), '?')
        tag = f" [{arch.upper()} FLYBY]" if arch != 'direct' else ""
        print(f"  #{rank:2d}: {n1:15s}[{c1:3s}] -> {n2:15s}[{c2:3s}] -> {n3:15s}[{c3:3s}]"
              f"  |  dv={dv:5.2f}  launch_dv={ldv:5.2f}{tag}")

    print(f"\nCoarse: {t_coarse:.1f}s | Fine: {t_total-t_coarse:.1f}s | Total: {t_total:.1f}s")

    output = os.path.join(project_root, f"optimal_asteroid_paths/results_73ast_ga_{int(time.time())}.pkl")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved: {output}")
