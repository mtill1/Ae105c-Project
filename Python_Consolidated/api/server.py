"""FastAPI server. Run with: python -m Python_Consolidated.api"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Path setup so the rest of the codebase imports cleanly
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
CODE_DIR = _HERE.parent
sys.path.insert(0, str(CODE_DIR))
os.chdir(REPO_ROOT)

from . import __version__, jobs as jobs_mod, schemas as S
from .serialization import entry_to_summary, entry_to_detail, to_jsonable

PKL_DIR = REPO_ROOT / 'optimal_asteroid_paths' / 'pkl'

app = FastAPI(
    title='Ae105c Trajectory Optimization API',
    version=__version__,
    description=(
        'HTTP REST API for the Ae105c three-asteroid trajectory pipeline.\n\n'
        'Three groups of endpoints:\n'
        '  • **Browse** (`/results`, `/asteroids`) — list and inspect saved results\n'
        '  • **Audit** (`/verify`, `/inspect`) — physical-feasibility check + per-leg dump\n'
        '  • **Jobs** (`/jobs/...`) — async optimization runs (subprocess or GCP)\n\n'
        'See [`Tutorials/07_using_the_api.md`](Tutorials/07_using_the_api.md) for examples.'
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ============================================================================
# helpers
# ============================================================================

def _resolve_pkl_path(name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = PKL_DIR / name
    if not p.exists():
        raise HTTPException(404, f'pkl not found: {name}')
    return p


def _load_pkl(name: str) -> Any:
    return pickle.load(open(_resolve_pkl_path(name), 'rb'))


def _normalize_entries(data: Any) -> List[Dict[str, Any]]:
    """Reuse main.py's normalizer."""
    from main import _normalize_entries
    return _normalize_entries(data)


def _select_entry(entries: List[Dict[str, Any]],
                  rank: Optional[int],
                  names: Optional[List[str]]) -> Dict[str, Any]:
    if rank is not None:
        if rank < 1 or rank > len(entries):
            raise HTTPException(400, f'rank {rank} out of range (1..{len(entries)})')
        return entries[rank - 1]
    if names:
        wanted = tuple(n.upper() for n in names)
        for e in entries:
            if tuple(n.upper() for n in e['names']) == wanted:
                return e
        raise HTTPException(404, f'no entry with names={names}')
    return entries[0]


# ============================================================================
# Health + asteroids
# ============================================================================

@app.get('/health', response_model=S.HealthResponse)
def health():
    return S.HealthResponse(status='ok', version=__version__,
                             repo_root=str(REPO_ROOT))


@app.get('/api/v1/asteroids', response_model=List[S.AsteroidInfo])
def list_asteroids():
    """List all 73 asteroids with composition class and science score."""
    from core import load_kernels
    from optimization import load_composition_map

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    comp_map = load_composition_map('asteroid_tradeoff.csv')

    sci_scores: Dict[str, float] = {}
    try:
        import pandas as pd
        df = pd.read_csv('asteroid_tradeoff.csv')
        for _, row in df.iterrows():
            raw = str(row['Name_DecRadius']).split('(')[0].strip()
            parts = raw.split()
            name = (' '.join(parts[1:]).upper()
                    if parts and parts[0].replace('.', '').isdigit()
                    else raw.upper())
            sci_scores[name] = float(row['Total_WeightedScore'])
    except Exception:
        pass

    out = []
    for a in asteroid_list:
        n = a['NAME']
        out.append(S.AsteroidInfo(
            name=n,
            spice_id=int(a['ID']),
            composition=comp_map.get(n.upper()),
            science_score=sci_scores.get(n.upper()),
        ))
    out.sort(key=lambda a: a.name)
    return out


# ============================================================================
# Browse results
# ============================================================================

@app.get('/api/v1/results', response_model=S.ResultsList)
def list_results():
    """List all saved result pkl files."""
    files = []
    for f in sorted(PKL_DIR.glob('*.pkl')):
        files.append({
            'filename': f.name,
            'size_kb':  round(f.stat().st_size / 1024, 1),
            'mtime':    f.stat().st_mtime,
        })
    return S.ResultsList(pkl_dir=str(PKL_DIR), count=len(files), files=files)


@app.get('/api/v1/results/{filename}', response_model=S.ResultSummary)
def get_result_summary(filename: str, top: int = 10):
    """Top-N entries plus saved metadata for one pkl."""
    data = _load_pkl(filename)
    entries = _normalize_entries(data)
    metadata: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k in ('alpha', 'sci_ref', 'flyby', 'composition_required',
                   'required_asteroids', 'launch_window', 'm_init_kg',
                   'thrust_N', 'source_pkl'):
            if k in data:
                metadata[k] = to_jsonable(data[k])
    summaries = [entry_to_summary(e, rank=i + 1)
                 for i, e in enumerate(entries[:top])]
    return S.ResultSummary(
        filename=filename,
        n_entries=len(entries),
        metadata=metadata,
        top_entries=[S.ResultEntry(**s) for s in summaries],
    )


@app.get('/api/v1/results/{filename}/entries/{rank}', response_model=S.ResultDetail)
def get_result_entry(filename: str, rank: int):
    """Full per-entry detail (epochs, all delta-v components, audit)."""
    data = _load_pkl(filename)
    entries = _normalize_entries(data)
    if rank < 1 or rank > len(entries):
        raise HTTPException(400, f'rank {rank} out of range (1..{len(entries)})')
    return S.ResultDetail(**entry_to_detail(entries[rank - 1], rank))


# ============================================================================
# Verify + Inspect
# ============================================================================

@app.post('/api/v1/verify', response_model=S.VerifyResponse)
def verify(req: S.VerifyRequest):
    """Audit one saved trajectory. Pass/fail + diagnostic numbers."""
    from core import audit_flyby_geometry, get_id_from_asteroid_name, load_kernels

    data = _load_pkl(req.pkl)
    entries = _normalize_entries(data)
    entry = _select_entry(entries, req.rank, req.names)
    res = entry['result']
    arch = res.get('arch') or res.get('flyby_name') or 'mars'

    if 'et_flyby' not in res or arch == 'direct':
        raise HTTPException(400, f'No flyby in entry — arch={arch}')

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    a1_id = str(int(get_id_from_asteroid_name(asteroid_list, entry['names'][0])))
    audit = audit_flyby_geometry(res['et_launch'], res['et_flyby'],
                                  res['et_arrive_1'], a1_id, arch)

    # Build Δv breakdown
    import numpy as np
    breakdown: Dict[str, Any] = {}
    total = 0.0
    for k, label in [
        ('delta_v_launch',    'launch'),
        ('delta_v_flyby',     f'{arch}_powered'),
        ('delta_v_A1_arrive', 'arrive_a1'),
        ('delta_v_A1_leave',  'depart_a1'),
        ('delta_v_A2_arrive', 'arrive_a2'),
        ('delta_v_A2_leave',  'depart_a2'),
        ('delta_v_A3_arrive', 'arrive_a3'),
    ]:
        if k not in res:
            continue
        v = res[k]
        if hasattr(v, '__len__') and not isinstance(v, str):
            mag = float(np.linalg.norm(v))
        else:
            mag = float(abs(v)) if v is not None else 0.0
        breakdown[label] = mag
        total += mag

    saved = res.get('delta_v_total')
    return S.VerifyResponse(
        pkl=req.pkl,
        names=list(entry['names']),
        arch=arch,
        feasible=bool(audit.get('feasible', False)),
        audit=to_jsonable(audit),
        delta_v_breakdown=breakdown,
        saved_total_kms=float(saved) if saved is not None else None,
        recomputed_total_kms=total,
    )


@app.post('/api/v1/inspect', response_model=S.InspectResponse)
def inspect(req: S.VerifyRequest):
    """Full per-leg dump: Lambert V1/V2 vectors, Δv vectors, flyby diagnostics."""
    from core import (audit_flyby_geometry, get_id_from_asteroid_name,
                       get_state, solve_lambert, load_kernels,
                       MU_SUN, DAY, YEAR)
    from optimization import FLYBY_BODIES
    import numpy as np
    import spiceypy

    data = _load_pkl(req.pkl)
    entries = _normalize_entries(data)
    entry = _select_entry(entries, req.rank, req.names)
    res = entry['result']
    arch = res.get('arch') or res.get('flyby_name') or 'mars'

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    name_to_id = {a['NAME'].upper(): str(int(a['ID'])) for a in asteroid_list}
    names = [n.upper() for n in entry['names']]
    a_ids = [name_to_id[n] for n in names]

    et_launch = res['et_launch']
    et_flyby  = res.get('et_flyby')
    et_arr_1  = res['et_arrive_1']
    et_stay_1 = res['et_stay_1']
    et_arr_2  = res['et_arrive_2']
    et_stay_2 = res['et_stay_2']
    et_arr_3  = res['et_arrive_3']
    m_revs = res.get('m_revs', (0, 0, 0, 0))

    timeline = {'launch': spiceypy.et2utc(et_launch, 'ISOC', 3) + 'Z'}
    if et_flyby:
        timeline[f'{arch}_flyby'] = spiceypy.et2utc(et_flyby, 'ISOC', 3) + 'Z'
    timeline[f'arrive_{names[0]}'] = spiceypy.et2utc(et_arr_1, 'ISOC', 3) + 'Z'
    timeline[f'depart_{names[0]}'] = spiceypy.et2utc(et_stay_1, 'ISOC', 3) + 'Z'
    timeline[f'arrive_{names[1]}'] = spiceypy.et2utc(et_arr_2, 'ISOC', 3) + 'Z'
    timeline[f'depart_{names[1]}'] = spiceypy.et2utc(et_stay_2, 'ISOC', 3) + 'Z'
    timeline[f'arrive_{names[2]}'] = spiceypy.et2utc(et_arr_3, 'ISOC', 3) + 'Z'

    # Per-leg Lambert
    earth_r,  earth_v  = get_state('399', et_launch)
    a1_arr_r, a1_arr_v = get_state(a_ids[0], et_arr_1)
    a1_lv_r,  a1_lv_v  = get_state(a_ids[0], et_stay_1)
    a2_arr_r, a2_arr_v = get_state(a_ids[1], et_arr_2)
    a2_lv_r,  a2_lv_v  = get_state(a_ids[1], et_stay_2)
    a3_arr_r, a3_arr_v = get_state(a_ids[2], et_arr_3)

    legs_in: List[Any] = []
    if et_flyby:
        fb_r, fb_v = get_state(FLYBY_BODIES[arch]['id'], et_flyby)
        legs_in.append(('Earth->' + arch.title(), earth_r, earth_v, fb_r, fb_v,
                         et_launch, et_flyby, m_revs[0]))
        legs_in.append((f'{arch.title()}->{names[0]}', fb_r, fb_v,
                         a1_arr_r, a1_arr_v, et_flyby, et_arr_1, m_revs[1]))
    else:
        legs_in.append((f'Earth->{names[0]}', earth_r, earth_v,
                         a1_arr_r, a1_arr_v, et_launch, et_arr_1, m_revs[0]))
    legs_in.append((f'{names[0]}->{names[1]}', a1_lv_r, a1_lv_v,
                     a2_arr_r, a2_arr_v, et_stay_1, et_arr_2, m_revs[2]))
    legs_in.append((f'{names[1]}->{names[2]}', a2_lv_r, a2_lv_v,
                     a3_arr_r, a3_arr_v, et_stay_2, et_arr_3, m_revs[3]))

    legs_out = []
    for label, r0, v0_b, r1, v1_b, et0, et1, mrev in legs_in:
        tof_d = (et1 - et0) / DAY
        V1, V2, _ = solve_lambert(r0, r1, tof_d, mrev, MU_SUN)
        dv_dep = V1 - v0_b
        dv_arr = v1_b - V2
        legs_out.append({
            'label': label,
            'tof_days': float(tof_d),
            'tof_years': float(tof_d / 365.25),
            'm_revs': int(mrev),
            'body_v_start_kms': [float(x) for x in v0_b],
            'lambert_V1_kms':   [float(x) for x in V1],
            'dv_at_departure_kms':           [float(x) for x in dv_dep],
            'dv_at_departure_magnitude_kms': float(np.linalg.norm(dv_dep)),
            'lambert_V2_kms':   [float(x) for x in V2],
            'body_v_end_kms':   [float(x) for x in v1_b],
            'dv_at_arrival_kms':             [float(x) for x in dv_arr],
            'dv_at_arrival_magnitude_kms':   float(np.linalg.norm(dv_arr)),
        })

    # Flyby audit
    audit: Dict[str, Any] = {}
    feasible = True
    if et_flyby:
        audit = audit_flyby_geometry(et_launch, et_flyby, et_arr_1, a_ids[0], arch)
        feasible = bool(audit.get('feasible', False))

    # Δv breakdown (same as verify)
    breakdown = {}
    total = 0.0
    for k, label in [
        ('delta_v_launch', 'launch'),
        ('delta_v_flyby', f'{arch}_powered'),
        ('delta_v_A1_arrive', 'arrive_a1'),
        ('delta_v_A1_leave', 'depart_a1'),
        ('delta_v_A2_arrive', 'arrive_a2'),
        ('delta_v_A2_leave', 'depart_a2'),
        ('delta_v_A3_arrive', 'arrive_a3'),
    ]:
        if k not in res: continue
        v = res[k]
        mag = (float(np.linalg.norm(v))
                if hasattr(v, '__len__') and not isinstance(v, str)
                else (float(abs(v)) if v is not None else 0.0))
        breakdown[label] = mag
        total += mag

    return S.InspectResponse(
        pkl=req.pkl, names=list(entry['names']), arch=arch,
        feasible=feasible,
        audit=to_jsonable(audit),
        delta_v_breakdown=breakdown,
        saved_total_kms=float(res['delta_v_total']) if 'delta_v_total' in res else None,
        recomputed_total_kms=total,
        legs=legs_out,
        timeline=timeline,
        mission_duration_yr=float((et_arr_3 - et_launch) / YEAR),
    )


# ============================================================================
# Jobs
# ============================================================================

def _job_to_response(j: Dict[str, Any]) -> S.JobInfo:
    return S.JobInfo(**{k: j.get(k) for k in S.JobInfo.model_fields.keys()})


@app.post('/api/v1/jobs/optimize', response_model=S.JobInfo, status_code=202)
def submit_optimize_job(req: S.OptimizeJobRequest):
    """Launch a new optimization. Returns job_id; poll /jobs/{id} for status."""
    body = req.model_dump(exclude_none=False)
    job = jobs_mod.submit_job(body)
    return _job_to_response(job)


@app.get('/api/v1/jobs', response_model=S.JobsList)
def list_jobs(limit: int = 50):
    rows = jobs_mod.list_jobs(limit=limit)
    return S.JobsList(count=len(rows),
                      jobs=[_job_to_response(r) for r in rows])


@app.get('/api/v1/jobs/{job_id}', response_model=S.JobInfo)
def get_job(job_id: str):
    j = jobs_mod.get_job(job_id)
    if j is None:
        raise HTTPException(404, f'job {job_id} not found')
    return _job_to_response(j)


@app.delete('/api/v1/jobs/{job_id}', response_model=S.JobInfo)
def cancel_job(job_id: str):
    j = jobs_mod.cancel_job(job_id)
    if j is None:
        raise HTTPException(404, f'job {job_id} not found')
    return _job_to_response(j)


@app.get('/api/v1/jobs/{job_id}/log')
def get_job_log(job_id: str, tail: int = 200):
    """Last `tail` lines of the job's log file."""
    j = jobs_mod.get_job(job_id)
    if j is None:
        raise HTTPException(404, f'job {job_id} not found')
    log = j.get('log_path')
    if not log or not os.path.exists(log):
        return {'job_id': job_id, 'log': ''}
    try:
        with open(log, 'r', errors='replace') as f:
            lines = f.readlines()[-tail:]
        return {'job_id': job_id, 'log': ''.join(lines)}
    except Exception as e:
        raise HTTPException(500, str(e))
