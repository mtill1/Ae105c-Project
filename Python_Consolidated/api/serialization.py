"""Convert pkl-loaded result entries into JSON-safe dicts.

Numpy arrays → lists. SPICE epochs → ISO datetime strings. NaN/Inf → null.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


def _safe_float(x: Any) -> Any:
    """Convert numpy scalars to floats, NaN/Inf to None."""
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / spice / weird types into JSON-safe form."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [_safe_float(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return _safe_float(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, float):
        return _safe_float(obj)
    return obj


def et_to_iso(et: float) -> str:
    """SPICE ET (seconds past J2000) → ISO 8601 UTC string. Lazy spiceypy import."""
    import spiceypy
    return spiceypy.et2utc(float(et), 'ISOC', 3) + 'Z'


def entry_to_summary(entry: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Compact summary suitable for list views."""
    res = entry.get('result', entry)
    audit = (entry.get('audit_saved')
             or res.get('audit')
             or {})
    out = {
        'rank':         rank,
        'names':        list(entry.get('names', ())),
        'arch':         res.get('arch') or res.get('flyby_name') or 'direct',
        'delta_v_total_kms': _safe_float(res.get('delta_v_total')),
        'science_sum':  _safe_float(entry.get('sci_sum')),
        'combined_score': _safe_float(entry.get('combined')),
        'mission_duration_yr': _safe_float(
            (res.get('et_arrive_3', 0) - res.get('et_launch', 0)) / 31557600.0
            if res.get('et_arrive_3') and res.get('et_launch') else None),
        'feasible':     bool(audit.get('feasible')) if audit else None,
        'flyby_alt_km': _safe_float(audit.get('periapsis_alt_km')) if audit else None,
        'turn_angle_deg':  _safe_float(audit.get('turn_angle_deg')) if audit else None,
        'turn_max_deg':    _safe_float(audit.get('turn_max_deg')) if audit else None,
    }
    return out


def entry_to_detail(entry: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Full per-entry detail (epochs, all delta-v components, audit)."""
    res = entry.get('result', entry)
    audit = (entry.get('audit_saved')
             or res.get('audit')
             or {})

    epochs = {}
    for key in ('et_launch', 'et_flyby', 'et_arrive_1', 'et_stay_1',
                'et_arrive_2', 'et_stay_2', 'et_arrive_3'):
        if key in res:
            try:
                epochs[key] = {
                    'spice_et': _safe_float(res[key]),
                    'utc':      et_to_iso(res[key]),
                }
            except Exception:
                epochs[key] = {'spice_et': _safe_float(res[key]), 'utc': None}

    dv_components = {}
    for key in ('delta_v_launch', 'delta_v_flyby',
                'delta_v_A1_arrive', 'delta_v_A1_leave',
                'delta_v_A2_arrive', 'delta_v_A2_leave',
                'delta_v_A3_arrive'):
        if key in res:
            v = res[key]
            if isinstance(v, np.ndarray):
                dv_components[key] = {
                    'vector_kms':    [_safe_float(x) for x in v.tolist()],
                    'magnitude_kms': _safe_float(np.linalg.norm(v)),
                }
            else:
                dv_components[key] = {
                    'vector_kms':    None,
                    'magnitude_kms': _safe_float(abs(float(v))) if v is not None else None,
                }

    summary = entry_to_summary(entry, rank)
    summary.update({
        'epochs':        epochs,
        'delta_v':       dv_components,
        'audit':         to_jsonable(audit) if audit else None,
        'm_revs':        list(res['m_revs']) if 'm_revs' in res else None,
        'flyby_body_id': res.get('flyby_body'),
    })
    return summary
