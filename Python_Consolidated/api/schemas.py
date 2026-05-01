"""Pydantic models for API request/response validation."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Common
# ============================================================================

class HealthResponse(BaseModel):
    status: Literal['ok'] = 'ok'
    version: str
    repo_root: str


class AsteroidInfo(BaseModel):
    name: str
    spice_id: int
    composition: Optional[str] = None
    science_score: Optional[float] = None


# ============================================================================
# Results browsing
# ============================================================================

class ResultsList(BaseModel):
    pkl_dir: str
    count: int
    files: List[Dict[str, Any]]


class ResultEntry(BaseModel):
    rank: int
    names: List[str]
    arch: str
    delta_v_total_kms: Optional[float]
    science_sum: Optional[float] = None
    combined_score: Optional[float] = None
    mission_duration_yr: Optional[float]
    feasible: Optional[bool] = None
    flyby_alt_km: Optional[float] = None
    turn_angle_deg: Optional[float] = None
    turn_max_deg: Optional[float] = None


class ResultSummary(BaseModel):
    filename: str
    n_entries: int
    metadata: Dict[str, Any] = {}
    top_entries: List[ResultEntry]


class ResultDetail(BaseModel):
    rank: int
    names: List[str]
    arch: str
    delta_v_total_kms: Optional[float]
    science_sum: Optional[float] = None
    combined_score: Optional[float] = None
    mission_duration_yr: Optional[float]
    feasible: Optional[bool] = None
    flyby_alt_km: Optional[float] = None
    turn_angle_deg: Optional[float] = None
    turn_max_deg: Optional[float] = None
    epochs: Dict[str, Any]
    delta_v: Dict[str, Any]
    audit: Optional[Dict[str, Any]] = None
    m_revs: Optional[List[int]] = None
    flyby_body_id: Optional[str] = None


# ============================================================================
# Verify / Inspect
# ============================================================================

class VerifyRequest(BaseModel):
    pkl: str = Field(..., description='Filename in pkl/ or absolute path')
    rank: Optional[int] = Field(1, ge=1, description='1-indexed rank to verify')
    names: Optional[List[str]] = Field(
        None, min_length=3, max_length=3,
        description='Three asteroid names — alternative to --rank')


class VerifyResponse(BaseModel):
    pkl: str
    names: List[str]
    arch: str
    feasible: bool
    audit: Dict[str, Any]
    delta_v_breakdown: Dict[str, Any]
    saved_total_kms: Optional[float]
    recomputed_total_kms: Optional[float]


class InspectResponse(VerifyResponse):
    legs: List[Dict[str, Any]]
    timeline: Dict[str, str]
    mission_duration_yr: float


# ============================================================================
# Optimization jobs (async)
# ============================================================================

class OptimizeJobRequest(BaseModel):
    mode: Literal[
        'two_level', 'feasible', 'pareto', 'beam',
        'mars_diverse', 'mars_diverse_science',
    ] = Field(..., description='Which optimization workflow to run')
    backend: Literal['subprocess', 'gcp'] = Field(
        'subprocess', description='Where to run')
    alpha: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description='Δv weight (1.0 pure dv, lower = more science)')
    science_ref: Optional[float] = Field(15.0, ge=1.0)
    required_asteroids: Optional[List[str]] = Field(
        None, description='At least one of these must be in every triplet')
    require_all_asteroids: Optional[List[str]] = Field(
        None, description='ALL of these must be in every triplet (overrides required_asteroids)')
    flyby: Optional[Literal['mars', 'moon', 'earth', 'auto']] = 'mars'
    top_n: Optional[int] = Field(50, ge=1, le=500)
    feasible_top_n: Optional[int] = Field(25, ge=1, le=200)
    beam_width: Optional[int] = Field(15, ge=1, le=50)
    launch_min_utc: Optional[str] = 'Jan 1 12:00:00 UTC 2027'
    launch_max_utc: Optional[str] = 'Dec 31 12:00:00 UTC 2035'
    science: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description='For two_level mode: science weighting')
    diverse: Optional[bool] = False


class JobInfo(BaseModel):
    id: str
    status: Literal['queued', 'running', 'done', 'failed', 'cancelled']
    backend: str
    mode: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    request: Dict[str, Any]
    log_path: Optional[str] = None
    result_pkl: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    vm_name: Optional[str] = None


class JobsList(BaseModel):
    count: int
    jobs: List[JobInfo]
