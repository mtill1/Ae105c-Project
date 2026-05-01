# Tutorial 07 — Using the HTTP API

The optimization pipeline is exposed as a FastAPI HTTP service so you can call
it from notebooks, scripts, MATLAB, curl, or anything that speaks JSON.

**Two execution backends:**
- `subprocess` (default) — runs the optimization on the same machine that
  hosts the API server. Works without any GCP setup.
- `gcp` — provisions a GCP VM, runs the optimization there, copies results
  back, deletes the VM. Same API surface; just one flag.

**Time:** ~30 s setup. **No GCP required for subprocess backend.**

## Pre-requisites

- Tutorial 00 finished (Python env + kernels + 73 BSPs)
- Install API extras (already in `requirements.txt`):

  ```bash
  pip install -r Python_Consolidated/requirements.txt
  ```

## Start the server

```bash
python -m Python_Consolidated.api
```

You'll see:

```
Ae105c API → http://127.0.0.1:8000
Swagger UI → http://127.0.0.1:8000/docs
Health     → http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` in a browser for the interactive Swagger UI
— you can hit every endpoint there with no code.

For LAN access (other machines, your phone): `--host 0.0.0.0 --port 8000`.

## Endpoint reference

### Browse

| Method | Path | Returns |
|---|---|---|
| `GET`  | `/health`                                     | `{status, version, repo_root}` |
| `GET`  | `/api/v1/asteroids`                           | List of 73 asteroids with composition + science score |
| `GET`  | `/api/v1/results`                             | List of saved `.pkl` files |
| `GET`  | `/api/v1/results/{filename}?top=N`            | Top-N entries + saved metadata |
| `GET`  | `/api/v1/results/{filename}/entries/{rank}`   | Single entry detail (epochs, all Δv components) |

### Audit

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/v1/verify`  | `{pkl, rank?, names?}` | Pass/fail audit + flyby diagnostics + Δv breakdown |
| `POST` | `/api/v1/inspect` | `{pkl, rank?, names?}` | Full per-leg dump: Lambert V1/V2 vectors, Δv vectors, timeline |

### Async optimization jobs

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/v1/jobs/optimize`            | `{mode, backend, alpha?, ...}` | `JobInfo` with new `id` |
| `GET`  | `/api/v1/jobs?limit=N`             | (none)                          | All jobs |
| `GET`  | `/api/v1/jobs/{id}`                | (none)                          | One job's status |
| `DELETE` | `/api/v1/jobs/{id}`              | (none)                          | Cancel (kills subprocess or deletes VM) |
| `GET`  | `/api/v1/jobs/{id}/log?tail=N`     | (none)                          | Tail of the job's log file |

`mode` is one of: `two_level`, `feasible`, `pareto`, `beam`, `mars_diverse`,
`mars_diverse_science`. `backend` is `subprocess` or `gcp`.

## Calling from Python

```python
from Python_Consolidated.api.client import Client

cli = Client('http://localhost:8000')

# Browse
print(cli.list_results()['count'])                  # 38
detail = cli.get_entry('mars_diverse_feasible_73.pkl', rank=1)
print(detail['delta_v_total_kms'])                  # 12.97

# Audit (fast, sync)
audit = cli.verify('mars_diverse_feasible_73.pkl', rank=1)
print(audit['feasible'], audit['audit']['turn_angle_deg'])

# Full per-leg dump
det = cli.inspect('mars_diverse_feasible_73.pkl', rank=1)
for leg in det['legs']:
    print(leg['label'],
          leg['lambert_V1_kms'],
          leg['dv_at_departure_magnitude_kms'])

# Async optimization (subprocess on local machine)
job = cli.submit_optimize(
    mode='mars_diverse_science',
    backend='subprocess',
    alpha=0.4,
    require_all_asteroids=['THEMIS', 'PSYCHE'],
)
print(f'job id={job.id} status={job.status}')

job.wait(progress=True)         # blocks, polls every 10 s
print(job.refresh()['result_pkl'])

# Same job, but on GCP instead
job = cli.submit_optimize(
    mode='mars_diverse_science',
    backend='gcp',
    alpha=0.4,
    require_all_asteroids=['THEMIS', 'PSYCHE'],
)
job.wait(timeout_seconds=3600)
```

## Calling from curl / shell

```bash
# List results
curl -s http://localhost:8000/api/v1/results | jq '.count'

# Top 3 entries of a specific pkl
curl -s 'http://localhost:8000/api/v1/results/mars_diverse_feasible_73.pkl?top=3' \
  | jq '.top_entries[] | {names, dv: .delta_v_total_kms, feasible}'

# Verify rank 1
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"pkl":"mars_diverse_feasible_73.pkl","rank":1}' \
  http://localhost:8000/api/v1/verify | jq '.feasible, .audit.turn_angle_deg'

# Submit a science-weighted Mars-only job locally
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{
        "mode":"mars_diverse_science",
        "backend":"subprocess",
        "alpha":0.4,
        "require_all_asteroids":["THEMIS","PSYCHE"]
      }' \
  http://localhost:8000/api/v1/jobs/optimize | jq '.id, .status'

# Poll
curl -s http://localhost:8000/api/v1/jobs/<JOB_ID> | jq '.status, .result_pkl'

# Stream log
curl -s 'http://localhost:8000/api/v1/jobs/<JOB_ID>/log?tail=50' | jq -r '.log'
```

## Subprocess vs GCP backend

| | subprocess | gcp |
|---|---|---|
| Setup needed beyond Tutorial 00 | none | `gcloud auth login` |
| Where it runs | API host machine | freshly-provisioned GCP VM |
| Cores | whatever the host has | 12 vCPU per VM |
| Wall-time for a typical run | similar | similar |
| Cost | $0 (uses your laptop) | $0.05–$1.50 per job |
| Survives API server restart | yes (PID-tracked, OS-level) | yes (worker thread, but if API restarts mid-provision the VM may leak — check `gcloud compute instances list`) |
| Max concurrency | bounded by host CPU | bounded by your `CPUS_ALL_REGIONS` quota (12) |

**Default:** `subprocess`. Switch with `backend: "gcp"` per request — no
restart of the API server needed.

## Job state persistence

Jobs are stored in
`optimal_asteroid_paths/api_jobs/jobs.db` (SQLite). Logs are
`optimal_asteroid_paths/api_jobs/<uuid>.log`. Both survive API restarts.
On startup the server's status-poller will reconcile any "running" jobs
against their PID — terminated processes get marked `done` or `failed`.

## Common gotchas

- **API hangs starting**: another process holding port 8000. Use `--port 8001`.
- **GCP backend hangs in "running"**: VM took longer than expected to spin up.
  Check `gcloud compute instances list --zones=us-west1-b`. The job will
  eventually update.
- **`Saved:` not in log → job stays "running" forever**: the optimizer crashed
  silently. Check `cli.get_job(id).get('log_path')` for the traceback.
- **CORS**: the server allows all origins by default. If you change that,
  update `Python_Consolidated/api/server.py:CORSMiddleware`.

## What's next

- **`02_diverse_csm.md`** — composition-diverse search you can now trigger via API
- **`06_verify_physics.md`** — the same `verify`/`inspect` are accessible via HTTP

## Stop the server

`Ctrl-C` in the terminal running it. Running jobs continue (they're detached
subprocesses or GCP VMs); their state stays in `jobs.db`.
