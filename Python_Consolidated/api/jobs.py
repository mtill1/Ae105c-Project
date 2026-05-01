"""Async job queue with subprocess and GCP execution backends.

Persistent state in SQLite. Jobs run as detached subprocesses (so the API
server can be restarted without killing them). Status is updated by polling
the subprocess PID and the log file for sentinel strings.

Schema:
    jobs(id TEXT PRIMARY KEY, status, backend, mode, created_at, started_at,
         finished_at, request_json, log_path, result_pkl, error, pid, vm_name)
"""
from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Path setup
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
PKL_DIR = REPO_ROOT / 'optimal_asteroid_paths' / 'pkl'
LOG_DIR = REPO_ROOT / 'optimal_asteroid_paths' / 'api_jobs'
DB_PATH = LOG_DIR / 'jobs.db'
LOG_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _init_db():
    with _conn() as c:
        c.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                status       TEXT NOT NULL,
                backend      TEXT NOT NULL,
                mode         TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                started_at   TEXT,
                finished_at  TEXT,
                request_json TEXT NOT NULL,
                log_path     TEXT,
                result_pkl   TEXT,
                error        TEXT,
                pid          INTEGER,
                vm_name      TEXT
            )
        ''')


_init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d['request'] = json.loads(d.pop('request_json'))
    return d


# ============================================================================
# CLI argument builders for each mode
# ============================================================================

def _build_main_py_args(req: Dict[str, Any]) -> List[str]:
    """For modes that go through main.py optimize."""
    mode = req['mode']
    args = ['optimize']
    if mode == 'feasible':
        args.append('--feasible')
        if req.get('feasible_top_n'):
            args += ['--feasible-top-n', str(req['feasible_top_n'])]
    elif mode == 'two_level':
        if req.get('science') is not None:
            args += ['--science', str(req['science'])]
        if req.get('diverse'):
            args.append('--diverse')
    elif mode == 'beam':
        args += ['--beam', str(req.get('beam_width', 15))]
    elif mode == 'pareto':
        args.append('--pareto')
        if req.get('flyby'):
            args += ['--flyby', req['flyby']]
    if req.get('top_n'):
        args += ['--top-n', str(req['top_n'])]
    if req.get('launch_min_utc'):
        args += ['--launch-min', req['launch_min_utc']]
    if req.get('launch_max_utc'):
        args += ['--launch-max', req['launch_max_utc']]
    return args


def _build_mars_diverse_env(req: Dict[str, Any]) -> Dict[str, str]:
    """For mars_diverse_science mode."""
    env = {}
    if req.get('alpha') is not None:
        env['ALPHA'] = str(req['alpha'])
    if req.get('science_ref') is not None:
        env['SCI_REF'] = str(req['science_ref'])
    if req.get('top_n'):
        env['TOP_FINE_N'] = str(req['top_n'])
    if req.get('required_asteroids'):
        env['REQUIRED_ASTEROIDS'] = ','.join(req['required_asteroids'])
    if req.get('require_all_asteroids'):
        env['REQUIRE_ALL_ASTEROIDS'] = ','.join(req['require_all_asteroids'])
    return env


# ============================================================================
# Local subprocess executor
# ============================================================================

def _launch_subprocess(req: Dict[str, Any], log_path: Path) -> int:
    """Spawn the optimization as a detached subprocess. Returns PID."""
    mode = req['mode']
    py = sys.executable
    code_dir = REPO_ROOT / 'Python_Consolidated'
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    if mode == 'mars_diverse_science':
        cmd = [py, '-u', str(code_dir / 'gcp' / 'run_mars_diverse_science.py')]
        env.update(_build_mars_diverse_env(req))
    elif mode == 'mars_diverse':
        cmd = [py, '-u', str(code_dir / 'gcp' / 'run_mars_diverse.py')]
    else:
        cmd = [py, '-u', str(code_dir / 'main.py')] + _build_main_py_args(req)

    log_f = open(log_path, 'wb', buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )
    return proc.pid


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ============================================================================
# GCP executor (provisions VM, runs job, pulls result, deletes VM)
# ============================================================================

GCLOUD = '/opt/homebrew/share/google-cloud-sdk/bin/gcloud'
GCS_BUCKET = 'gs://ae105c-asteroid-data'
GCP_ZONE = 'us-west1-b'


def _gcp_setup_script(req: Dict[str, Any]) -> str:
    """Bash script that runs on the VM."""
    mode = req['mode']
    env_lines = ''
    cmd = ''
    if mode == 'mars_diverse_science':
        env = _build_mars_diverse_env(req)
        env_lines = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
        cmd = (f'env {env_lines} $HOME/env311/bin/python -u '
               f'Python_Consolidated/gcp/run_mars_diverse_science.py')
    elif mode == 'mars_diverse':
        cmd = ('$HOME/env311/bin/python -u '
               'Python_Consolidated/gcp/run_mars_diverse.py')
    else:
        # main.py optimize ...
        args = ' '.join(shlex.quote(a) for a in _build_main_py_args(req))
        cmd = f'$HOME/env311/bin/python -u Python_Consolidated/main.py {args}'

    return f'''
set -e
mkdir -p ~/project && cd ~/project
tar xzf ~/deploy.tar.gz && rm ~/deploy.tar.gz
mkdir -p generic_kernels/lsk generic_kernels/spk/satellites generic_kernels/spk/planets generic_kernels/pck NOTABLE_ASTEROID_BSPs
gsutil -q cp {GCS_BUCKET}/generic_kernels/naif0012.tls generic_kernels/lsk/
gsutil -q cp {GCS_BUCKET}/generic_kernels/jup310.bsp   generic_kernels/spk/satellites/
gsutil -q cp {GCS_BUCKET}/generic_kernels/de430.bsp    generic_kernels/spk/planets/
gsutil -q cp {GCS_BUCKET}/generic_kernels/gm_de431.tpc generic_kernels/pck/
gsutil -q cp {GCS_BUCKET}/generic_kernels/pck00010.tpc generic_kernels/pck/
gsutil -q -m rsync {GCS_BUCKET}/NOTABLE_ASTEROID_BSPs/ NOTABLE_ASTEROID_BSPs/ >/dev/null 2>&1
if [ ! -d $HOME/mc3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p $HOME/mc3 && rm /tmp/mc.sh
fi
$HOME/mc3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
$HOME/mc3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true
if [ ! -d $HOME/env311 ]; then
  $HOME/mc3/bin/conda create -p $HOME/env311 python=3.11 -y -q
  $HOME/mc3/bin/conda install -p $HOME/env311 -c conda-forge pykep spiceypy scipy numpy pandas tqdm matplotlib -y -q 2>&1 | tail -3
fi
echo "=== Run starting ==="
{cmd} 2>&1
echo "=== Run finished ==="
'''


def _launch_gcp(req: Dict[str, Any], job_id: str, log_path: Path) -> str:
    """Spawn a worker thread that provisions a VM, runs the job, pulls results,
    and deletes the VM. Returns vm_name immediately."""
    vm_name = f'ae105-job-{job_id[:8]}'

    def _worker():
        log_f = open(log_path, 'a', buffering=1)
        log_f.write(f'[{_now()}] GCP backend starting; vm={vm_name}\n')
        try:
            # 1. Create VM
            subprocess.check_call([
                GCLOUD, 'compute', 'instances', 'create', vm_name,
                '--zone', GCP_ZONE,
                '--machine-type', 'e2-custom-12-49152',
                '--boot-disk-size', '50GB',
                '--image-family', 'debian-12',
                '--image-project', 'debian-cloud',
                '--scopes', 'storage-ro,default',
                '--quiet',
            ], stdout=log_f, stderr=subprocess.STDOUT)
            time.sleep(25)

            # 2. Package + upload
            tar_path = LOG_DIR / f'{job_id}_deploy.tar.gz'
            code_dir = REPO_ROOT / 'Python_Consolidated'
            subprocess.check_call([
                'tar', 'czf', str(tar_path),
                '--exclude', '__pycache__',
                '-C', str(REPO_ROOT),
                'Python_Consolidated/core.py',
                'Python_Consolidated/optimization.py',
                'Python_Consolidated/lowthrust.py',
                'Python_Consolidated/mass_optimization.py',
                'Python_Consolidated/visualization.py',
                'Python_Consolidated/tradeoff.py',
                'Python_Consolidated/main.py',
                'Python_Consolidated/check_mission.py',
                'Python_Consolidated/gcp/run_mars_diverse.py',
                'Python_Consolidated/gcp/run_mars_diverse_science.py',
                'Python_Consolidated/gcp/run_single_triplet.py',
                'Python_Consolidated/gcp/run_mass_pareto.py',
                'Python_Consolidated/gcp/gcp_config.py',
                'asteroid_tradeoff.csv',
            ], stdout=log_f, stderr=subprocess.STDOUT)

            subprocess.check_call([
                GCLOUD, 'compute', 'scp', str(tar_path),
                f'{vm_name}:~/deploy.tar.gz',
                '--zone', GCP_ZONE,
            ], stdout=log_f, stderr=subprocess.STDOUT)

            # 3. Run setup + job script
            script = _gcp_setup_script(req)
            subprocess.check_call([
                GCLOUD, 'compute', 'ssh', vm_name,
                '--zone', GCP_ZONE,
                '--command', script,
            ], stdout=log_f, stderr=subprocess.STDOUT)

            # 4. Pull all new pkls
            subprocess.run([
                GCLOUD, 'compute', 'scp', '--recurse',
                f'{vm_name}:~/project/optimal_asteroid_paths/pkl/',
                str(PKL_DIR.parent / 'pkl_gcp_tmp'),
                '--zone', GCP_ZONE,
            ], stdout=log_f, stderr=subprocess.STDOUT, check=False)

            tmp = PKL_DIR.parent / 'pkl_gcp_tmp'
            if tmp.exists():
                for f in tmp.glob('*.pkl'):
                    target = PKL_DIR / f.name
                    if not target.exists():
                        f.rename(target)

            # 5. Delete VM
            subprocess.check_call([
                GCLOUD, 'compute', 'instances', 'delete', vm_name,
                '--zone', GCP_ZONE,
                '--delete-disks', 'all',
                '--quiet',
            ], stdout=log_f, stderr=subprocess.STDOUT)

            update_status(job_id, 'done')
        except subprocess.CalledProcessError as e:
            log_f.write(f'[{_now()}] GCP backend failed: {e}\n')
            update_status(job_id, 'failed', error=str(e))
            try:
                subprocess.run([
                    GCLOUD, 'compute', 'instances', 'delete', vm_name,
                    '--zone', GCP_ZONE, '--delete-disks', 'all', '--quiet',
                ], stdout=log_f, stderr=subprocess.STDOUT, check=False)
            except Exception:
                pass
        except Exception as e:
            log_f.write(f'[{_now()}] worker exception: {e}\n')
            update_status(job_id, 'failed', error=str(e))
        finally:
            log_f.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return vm_name


# ============================================================================
# Public API
# ============================================================================

def submit_job(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new job and launch its execution. Returns the row dict."""
    job_id = str(uuid.uuid4())
    log_path = LOG_DIR / f'{job_id}.log'
    backend = request.get('backend', 'subprocess')
    mode    = request['mode']
    created = _now()

    with _conn() as c:
        c.execute(
            '''INSERT INTO jobs (id, status, backend, mode, created_at,
                                  request_json, log_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (job_id, 'queued', backend, mode, created,
              json.dumps(request), str(log_path)))

    if backend == 'subprocess':
        try:
            pid = _launch_subprocess(request, log_path)
            with _conn() as c:
                c.execute(
                    '''UPDATE jobs SET status=?, started_at=?, pid=?
                       WHERE id=?''',
                    ('running', _now(), pid, job_id))
        except Exception as e:
            with _conn() as c:
                c.execute(
                    '''UPDATE jobs SET status=?, error=?, finished_at=?
                       WHERE id=?''',
                    ('failed', str(e), _now(), job_id))
    elif backend == 'gcp':
        try:
            vm_name = _launch_gcp(request, job_id, log_path)
            with _conn() as c:
                c.execute(
                    '''UPDATE jobs SET status=?, started_at=?, vm_name=?
                       WHERE id=?''',
                    ('running', _now(), vm_name, job_id))
        except Exception as e:
            with _conn() as c:
                c.execute(
                    '''UPDATE jobs SET status=?, error=?, finished_at=?
                       WHERE id=?''',
                    ('failed', str(e), _now(), job_id))
    else:
        with _conn() as c:
            c.execute(
                '''UPDATE jobs SET status=?, error=?, finished_at=?
                   WHERE id=?''',
                ('failed', f'unknown backend: {backend}', _now(), job_id))

    return get_job(job_id)


def update_status(job_id: str, status: str, **fields):
    """Update job row. Mostly used by GCP worker thread."""
    sets = ['status=?']
    vals: List[Any] = [status]
    if status in ('done', 'failed', 'cancelled'):
        sets.append('finished_at=?'); vals.append(_now())
    for k, v in fields.items():
        sets.append(f'{k}=?'); vals.append(v)
    vals.append(job_id)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {','.join(sets)} WHERE id=?", vals)


def _refresh_subprocess_status(row: Dict[str, Any]) -> Dict[str, Any]:
    """If a subprocess job's PID has exited, infer status from the log."""
    if row['backend'] != 'subprocess' or row['status'] != 'running':
        return row
    pid = row.get('pid')
    if not pid or _is_pid_alive(pid):
        return row
    log_text = ''
    try:
        with open(row['log_path'], 'r', errors='replace') as f:
            log_text = f.read()
    except Exception:
        pass
    new_status = 'failed'
    error = None
    result_pkl = None
    if 'Saved:' in log_text:
        new_status = 'done'
        for line in log_text.splitlines():
            if line.startswith('Saved:'):
                path = line.split('Saved:', 1)[1].strip()
                result_pkl = os.path.basename(path)
                break
    elif 'Traceback' in log_text:
        last_lines = '\n'.join(log_text.splitlines()[-10:])
        error = last_lines
    update_status(row['id'], new_status, error=error, result_pkl=result_pkl)
    return get_job(row['id'])


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if row is None:
        return None
    d = _row_to_dict(row)
    return _refresh_subprocess_status(d)


def list_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',
            (limit,)).fetchall()
    return [_refresh_subprocess_status(_row_to_dict(r)) for r in rows]


def cancel_job(job_id: str) -> Optional[Dict[str, Any]]:
    job = get_job(job_id)
    if job is None:
        return None
    if job['status'] not in ('queued', 'running'):
        return job
    if job['backend'] == 'subprocess' and job.get('pid'):
        try:
            os.killpg(os.getpgid(job['pid']), 15)  # SIGTERM
        except (ProcessLookupError, PermissionError):
            pass
    elif job['backend'] == 'gcp' and job.get('vm_name'):
        try:
            subprocess.run([
                GCLOUD, 'compute', 'instances', 'delete', job['vm_name'],
                '--zone', GCP_ZONE, '--delete-disks', 'all', '--quiet',
            ], check=False, timeout=60)
        except Exception:
            pass
    update_status(job_id, 'cancelled')
    return get_job(job_id)
