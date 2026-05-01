"""Python client for the Ae105c API.

Usage:
    from Python_Consolidated.api.client import Client

    cli = Client('http://localhost:8000')

    # Browse
    print(cli.list_results())
    summary = cli.get_result('mars_diverse_feasible_73.pkl', top=5)

    # Verify / inspect
    audit = cli.verify('mars_diverse_feasible_73.pkl', rank=1)
    detail = cli.inspect('mars_diverse_feasible_73.pkl', rank=1)

    # Async optimize
    job = cli.submit_optimize(
        mode='mars_diverse_science',
        backend='subprocess',
        alpha=0.4,
        require_all_asteroids=['THEMIS', 'PSYCHE'])
    job.wait()
    print(job.refresh())   # final status
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError as e:
    raise ImportError(
        'httpx is required: pip install httpx (or install api requirements)'
    ) from e


class Job:
    """Wrapper around a queued job — supports polling and waiting."""

    def __init__(self, client: 'Client', info: Dict[str, Any]):
        self.client = client
        self.info = info

    @property
    def id(self) -> str:
        return self.info['id']

    @property
    def status(self) -> str:
        return self.info['status']

    def refresh(self) -> Dict[str, Any]:
        self.info = self.client._get(f'/api/v1/jobs/{self.id}')
        return self.info

    def wait(self, poll_seconds: float = 10.0,
             timeout_seconds: Optional[float] = None,
             progress: bool = False) -> Dict[str, Any]:
        """Block until the job reaches a terminal status."""
        start = time.time()
        while True:
            self.refresh()
            if self.status in ('done', 'failed', 'cancelled'):
                return self.info
            if progress:
                print(f'[job {self.id[:8]}] status={self.status} '
                      f'elapsed={time.time()-start:.0f}s', flush=True)
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                raise TimeoutError(
                    f'Job {self.id} still {self.status} after '
                    f'{timeout_seconds:.0f}s')
            time.sleep(poll_seconds)

    def log(self, tail: int = 200) -> str:
        return self.client._get(f'/api/v1/jobs/{self.id}/log',
                                 params={'tail': tail})['log']

    def cancel(self) -> Dict[str, Any]:
        self.info = self.client._delete(f'/api/v1/jobs/{self.id}')
        return self.info


class Client:
    """Typed wrapper around the Ae105c REST API."""

    def __init__(self, base_url: str = 'http://localhost:8000',
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- low-level ----

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        r = self._client.get(self.base_url + path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Dict) -> Any:
        r = self._client.post(self.base_url + path, json=body)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> Any:
        r = self._client.delete(self.base_url + path)
        r.raise_for_status()
        return r.json()

    # ---- browse ----

    def health(self) -> Dict[str, Any]:
        return self._get('/health')

    def list_asteroids(self) -> List[Dict[str, Any]]:
        return self._get('/api/v1/asteroids')

    def list_results(self) -> Dict[str, Any]:
        return self._get('/api/v1/results')

    def get_result(self, filename: str, top: int = 10) -> Dict[str, Any]:
        return self._get(f'/api/v1/results/{filename}', params={'top': top})

    def get_entry(self, filename: str, rank: int) -> Dict[str, Any]:
        return self._get(f'/api/v1/results/{filename}/entries/{rank}')

    # ---- audit ----

    def verify(self, pkl: str, rank: Optional[int] = None,
               names: Optional[List[str]] = None) -> Dict[str, Any]:
        body = {'pkl': pkl}
        if rank is not None: body['rank'] = rank
        if names: body['names'] = names
        return self._post('/api/v1/verify', body)

    def inspect(self, pkl: str, rank: Optional[int] = None,
                names: Optional[List[str]] = None) -> Dict[str, Any]:
        body = {'pkl': pkl}
        if rank is not None: body['rank'] = rank
        if names: body['names'] = names
        return self._post('/api/v1/inspect', body)

    # ---- jobs ----

    def submit_optimize(self, **kwargs) -> Job:
        info = self._post('/api/v1/jobs/optimize', kwargs)
        return Job(self, info)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._get('/api/v1/jobs', params={'limit': limit})['jobs']

    def get_job(self, job_id: str) -> Job:
        return Job(self, self._get(f'/api/v1/jobs/{job_id}'))

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._delete(f'/api/v1/jobs/{job_id}')
