#!/usr/bin/env python3
"""Run composition-diverse (C+S+X/M) trajectory optimization."""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core import load_kernels
from optimization import two_level_optimize, load_composition_map

script_dir = os.path.dirname(os.path.abspath(__file__))
bsp_folder = os.path.join(script_dir, "NOTABLE_ASTEROID_BSPs")
generic_kernels = os.path.join(script_dir, "generic_kernels")

print("Loading SPICE kernels...")
asteroid_list = load_kernels(bsp_folder, generic_kernels)
print(f"Loaded {len(asteroid_list)} asteroids.")

print("Loading composition map...")
comp_map = load_composition_map(os.path.join(script_dir, "asteroid_tradeoff.csv"))

counts = {}
for a in asteroid_list:
    cls = comp_map.get(a["NAME"].upper(), "Unknown")
    counts[cls] = counts.get(cls, 0) + 1
print(f"Composition breakdown: {counts}")

required = {"C", "S", "X/M"}
req_str = " + ".join(sorted(required))
print(f"Required: {req_str}")
nc = counts.get("C", 0)
ns = counts.get("S", 0)
nm = counts.get("X/M", 0)
print(f"Expected triplets: {nc}C x {ns}S x {nm}M x 6 = {nc*ns*nm*6}")
print()

t_start = time.time()

results = two_level_optimize(
    asteroid_list, 0, 0, 0,
    "Jan 1 12:00:00 UTC 2027", "Dec 31 12:00:00 UTC 2035",
    top_n=50,
    comp_map=comp_map,
    required_compositions=required,
)

elapsed = time.time() - t_start
print(f"\nCompleted in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")

output = f"results_diverse_CSM_{int(time.time())}.pkl"
with open(output, "wb") as f:
    pickle.dump(results, f)
print(f"Saved: {output}")
