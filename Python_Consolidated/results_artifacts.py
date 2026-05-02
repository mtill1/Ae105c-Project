"""Locations for constrained-run artifacts under optimal_asteroid_paths/ (flat layout)."""

from __future__ import annotations

import os
import re
from typing import List


def optimal_asteroid_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "optimal_asteroid_paths")


def _sort_key(name: str) -> tuple:
    """Prefer results_YYYYMMDD_HHMMSS.*; else lexicographic."""
    m = re.match(r"^results_(\d{8})_(\d{6})\.", name)
    if m:
        return (0, m.group(1), m.group(2), name)
    return (1, name)


def _candidates_in_dir(folder: str, ext: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.startswith("results_") and f.endswith(ext)]


def latest_results_csv(repo_root: str) -> str:
    return _latest_results_path(repo_root, ".csv")


def latest_results_pkl(repo_root: str) -> str:
    return _latest_results_path(repo_root, ".pkl")


def _latest_results_path(repo_root: str, ext: str) -> str:
    oap = optimal_asteroid_dir(repo_root)
    pool = _candidates_in_dir(oap, ext)
    if not pool:
        legacy = os.path.join(oap, "csv" if ext == ".csv" else "pkl")
        pool = _candidates_in_dir(legacy, ext) if os.path.isdir(legacy) else []
        if pool:
            pool.sort(key=_sort_key)
            return os.path.join(legacy, pool[-1])
    if not pool:
        raise FileNotFoundError(
            f"No results_*{ext} found under {oap} (or legacy {ext.strip('.')}/). "
            "Run run_and_export_constrained_results.py first."
        )
    pool.sort(key=_sort_key)
    return os.path.join(oap, pool[-1])
