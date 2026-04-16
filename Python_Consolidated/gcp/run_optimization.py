#!/usr/bin/env python3
"""
Main entry point for running asteroid trajectory optimization on GCP.

Usage:
    python3 run_optimization.py [mode] [options]

Modes:
    diverse       — Composition-diverse (C+S+X/M) optimization (RECOMMENDED)
    two_level     — Two-level optimization (any composition)
    beam          — Beam search
    brute_force   — Brute force N^3
    mars          — Mars flyby optimization

Options:
    --top_n N         — Number of top candidates for fine pass (default: 50)
    --beam_width K    — Beam width for beam search (default: 10)
    --alpha A         — Science weighting: 1.0=pure dv, 0.7=70% dv + 30% sci (default: 1.0)
    --science_csv F   — Path to asteroid_tradeoff.csv for science scoring
    --subset N        — Only use first N asteroids (for testing)
    --compositions C  — Required compositions, comma-separated (default: C,S,X/M)
"""

import sys
import os
import time
import argparse
import pickle

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def main():
    parser = argparse.ArgumentParser(description='Asteroid trajectory optimization')
    parser.add_argument('mode', nargs='?', default='diverse',
                        choices=['diverse', 'two_level', 'beam', 'brute_force', 'mars'])
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--science_csv', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--compositions', type=str, default='C,S,X/M',
                        help='Required composition classes, comma-separated')
    args = parser.parse_args()

    from core import load_kernels
    from optimization import (two_level_optimize, beam_search,
                              generate_optimized_data,
                              generate_mars_transfer_optimized,
                              load_composition_map)

    # Paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bsp_folder = os.path.join(script_dir, 'NOTABLE_ASTEROID_BSPs')
    generic_kernels = os.path.join(script_dir, 'generic_kernels')

    print(f"Loading SPICE kernels from {bsp_folder}...")
    asteroid_list = load_kernels(bsp_folder, generic_kernels)
    print(f"Loaded {len(asteroid_list)} asteroids.\n")

    if args.subset:
        asteroid_list = asteroid_list[:args.subset]
        print(f"Using subset of {len(asteroid_list)} asteroids.\n")

    # Science scores
    science_scores = None
    if args.science_csv:
        import pandas as pd
        df = pd.read_csv(args.science_csv)
        science_scores = {}
        for _, row in df.iterrows():
            name = str(row['Name_DecRadius']).split('(')[0].strip()
            parts = name.split()
            if parts and parts[0].replace('.', '').isdigit():
                name = ' '.join(parts[1:])
            science_scores[name.upper()] = row['Total_WeightedScore']
        print(f"Loaded science scores for {len(science_scores)} asteroids.\n")

    # Composition map
    comp_map = None
    required_compositions = None
    tradeoff_csv = os.path.join(script_dir, 'asteroid_tradeoff.csv')
    if args.mode == 'diverse' or args.compositions != 'C,S,X/M':
        if os.path.exists(tradeoff_csv):
            comp_map = load_composition_map(tradeoff_csv)
            required_compositions = set(args.compositions.split(','))
            counts = {}
            for a in asteroid_list:
                cls = comp_map.get(a['NAME'].upper(), 'Unknown')
                counts[cls] = counts.get(cls, 0) + 1
            print(f"Composition breakdown: {counts}")
            req_str = ' + '.join(sorted(required_compositions))
            print(f"Required: {req_str}\n")
        else:
            print(f"WARNING: {tradeoff_csv} not found, running without composition filter.\n")

    LAUNCH_MIN = 'Jan 1 12:00:00 UTC 2027'
    LAUNCH_MAX = 'Dec 31 12:00:00 UTC 2035'

    t_start = time.time()

    if args.mode in ('diverse', 'two_level'):
        label = "DIVERSE (C+S+X/M)" if comp_map else "TWO-LEVEL"
        print(f"Running {label} optimization (top_n={args.top_n}, alpha={args.alpha})...\n")
        results = two_level_optimize(
            asteroid_list, 0, 0, 0, LAUNCH_MIN, LAUNCH_MAX,
            top_n=args.top_n, science_scores=science_scores, alpha=args.alpha,
            comp_map=comp_map, required_compositions=required_compositions)

    elif args.mode == 'beam':
        print(f"Running BEAM SEARCH (beam_width={args.beam_width}, alpha={args.alpha})...\n")
        results = beam_search(
            asteroid_list, LAUNCH_MIN, LAUNCH_MAX,
            beam_width=args.beam_width,
            science_scores=science_scores, alpha=args.alpha)

    elif args.mode == 'brute_force':
        print("Running BRUTE FORCE N^3 optimization...\n")
        results = generate_optimized_data(
            asteroid_list, 0, 0, 0, LAUNCH_MIN, LAUNCH_MAX)

    elif args.mode == 'mars':
        print("Running MARS FLYBY optimization...\n")
        results = generate_mars_transfer_optimized(
            asteroid_list, 0, 0, 0, 1, LAUNCH_MIN, LAUNCH_MAX)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"{'='*60}")

    # Save results
    output_path = f'results_{args.mode}_{int(time.time())}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
