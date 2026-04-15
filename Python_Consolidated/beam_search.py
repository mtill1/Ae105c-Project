"""
beam_search.py — Composition-aware beam search for multi-asteroid trajectories.

Complements the beam search in optimization.py with composition filtering
to ensure scientific diversity (C+S+M asteroid selection).

Usage:
    from beam_search import beam_search_optimize
    results = beam_search_optimize(
        asteroid_list, '2027 Jan 1', '2035 Dec 31',
        beam_width=15, composition_filter={'C', 'S', 'M'})
"""

import numpy as np
import spiceypy
from tqdm import tqdm

from core import (solve_lambert, get_state, DAY, YEAR, MONTH, MU_SUN,
                  MAX_MISSION_DURATION)
from optimization import optimize_times


def quick_screen_leg(body_a_id, body_b_id, departure_et, tof_range_days,
                     n_grid=10, mu=MU_SUN):
    """Quickly estimate minimum delta-V for a single transfer leg.

    Evaluates Lambert solutions on a coarse grid of time-of-flight values.

    Returns
    -------
    best_dv : float — estimated minimum delta-V [km/s]
    best_tof : float — best time of flight [seconds]
    """
    tof_min, tof_max = tof_range_days
    tofs = np.linspace(tof_min, tof_max, n_grid)
    r_dep, v_dep = get_state(body_a_id, departure_et)

    best_dv = np.inf
    best_tof = tofs[0] * DAY

    for tof_days in tofs:
        arrival_et = departure_et + tof_days * DAY
        r_arr, v_arr = get_state(body_b_id, arrival_et)
        v1, v2, ef = solve_lambert(r_dep, r_arr, tof_days, 0, mu)
        if ef != 1:
            continue
        dv = np.linalg.norm(v1 - v_dep) + np.linalg.norm(v2 - v_arr)
        if dv < best_dv:
            best_dv = dv
            best_tof = tof_days * DAY

    return best_dv, best_tof


def beam_search(asteroid_list, launch_et, beam_width=10, sequence_length=3,
                stay_time=4 * MONTH, tof_range_days=(14, 8 * 365.25),
                n_grid=10, composition_filter=None):
    """Find the best asteroid visitation sequences using beam search.

    Supports composition filtering to enforce scientific diversity.

    Parameters
    ----------
    asteroid_list : list of dict
        Each dict has 'ID', 'NAME', optionally 'COMPOSITION' ('C', 'S', 'M').
    launch_et : float
        Earth departure epoch (SPICE ET).
    beam_width : int
        Number of partial sequences to keep at each depth.
    sequence_length : int
        Number of asteroids to visit (default 3).
    stay_time : float
        Assumed stay time at each asteroid [seconds].
    tof_range_days : tuple
        (min_tof, max_tof) for each leg [days].
    n_grid : int
        Grid points per leg for screening.
    composition_filter : set or None
        If provided, final sequences must include at least one asteroid
        from each composition class in this set (e.g., {'C', 'S', 'M'}).

    Returns
    -------
    results : list of dict, sorted by estimated delta-V.
    """
    beam = [{'sequence': [], 'ids': [], 'estimated_dv': 0.0,
             'current_et': launch_et, 'current_body': '399'}]

    for depth in range(sequence_length):
        candidates = []
        for partial in tqdm(beam, desc=f"Beam depth {depth+1}/{sequence_length}",
                            leave=False):
            visited = set(partial['ids'])
            for asteroid in asteroid_list:
                a_id = str(int(asteroid['ID']))
                if a_id in visited:
                    continue

                leg_dv, leg_tof = quick_screen_leg(
                    partial['current_body'], a_id,
                    partial['current_et'], tof_range_days, n_grid)
                if leg_dv >= 1e3:
                    continue

                new_et = partial['current_et'] + leg_tof + stay_time
                if (new_et - launch_et) > MAX_MISSION_DURATION:
                    continue

                candidates.append({
                    'sequence': partial['sequence'] + [asteroid],
                    'ids': partial['ids'] + [a_id],
                    'estimated_dv': partial['estimated_dv'] + leg_dv,
                    'current_et': new_et,
                    'current_body': a_id,
                })

        if depth == sequence_length - 1 and composition_filter is not None:
            filtered = [c for c in candidates
                        if composition_filter.issubset(
                            {a.get('COMPOSITION', '?') for a in c['sequence']})]
            if filtered:
                candidates = filtered

        candidates.sort(key=lambda c: c['estimated_dv'])
        beam = candidates[:beam_width]
        if not beam:
            break

    return [{'sequence': e['sequence'], 'ids': e['ids'],
             'estimated_dv': e['estimated_dv']} for e in beam]


def beam_search_optimize(asteroid_list, launch_utc_min, launch_utc_max,
                         m_1=0, m_2=0, m_3=0,
                         beam_width=10, n_refine=None,
                         composition_filter=None):
    """Run beam search + full optimization refinement on top candidates.

    Parameters
    ----------
    asteroid_list : list of dict
        Asteroid catalog with 'ID', 'NAME', optionally 'COMPOSITION'.
    launch_utc_min, launch_utc_max : str
        UTC date strings for launch window.
    beam_width : int
        Beam width for sequence search.
    n_refine : int or None
        Number of top candidates to refine. If None, refines all.
    composition_filter : set or None
        E.g., {'C', 'S', 'M'} for compositional diversity.

    Returns
    -------
    results : list of dict, sorted by optimized delta-V.
    """
    et_min = spiceypy.str2et(launch_utc_min)
    et_max = spiceypy.str2et(launch_utc_max)
    launch_et = (et_min + et_max) / 2.0

    print(f"Phase 1: Beam search (k={beam_width}) over {len(asteroid_list)} asteroids...")
    candidates = beam_search(asteroid_list, launch_et, beam_width=beam_width,
                             composition_filter=composition_filter)
    if not candidates:
        print("No valid sequences found.")
        return []

    print(f"Found {len(candidates)} candidate sequences.")

    if n_refine is None:
        n_refine = len(candidates)
    n_refine = min(n_refine, len(candidates))
    launch_range = [et_min, et_max]

    print(f"Phase 2: Refining top {n_refine} with full optimization...")
    refined = []
    for cand in tqdm(candidates[:n_refine], desc="Refining"):
        ids = cand['ids']
        result = optimize_times(ids[0], ids[1], ids[2],
                                launch_range, m_1, m_2, m_3)
        result['sequence'] = cand['sequence']
        result['ids'] = ids
        result['estimated_dv'] = cand['estimated_dv']
        result['names'] = [a['NAME'] for a in cand['sequence']]
        refined.append(result)

    refined.sort(key=lambda r: r['delta_v_total'])

    print(f"\nTop 5 optimized results:")
    for i, r in enumerate(refined[:5]):
        names = ' -> '.join(r['names'])
        print(f"  {i+1}. {names}: {r['delta_v_total']:.2f} km/s "
              f"(screen: {r['estimated_dv']:.1f})")

    return refined
