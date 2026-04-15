## Asteroid Trajectory Selection (MATLAB)

### Quick start (Notable asteroids, Method B / Greedy)

1. `cd` to the repo root.
2. Add this module to your path:

```matlab
addpath(genpath('code/Asteroid_Trajectory_Selection'));
```

3. Run the greedy optimizer (one full pass by default):

```matlab
run('code/Asteroid_Trajectory_Selection/scripts/greedy/greedy_selector.m');
```

4. Print the best 3-asteroid tour found in the current workspace:

```matlab
run('code/Asteroid_Trajectory_Selection/scripts/greedy/find_best_path_greedy.m');
```

5. Generate the animation for the best tour:

```matlab
run('code/Asteroid_Trajectory_Selection/visualization/PlotBestGreedyPath.m');
```

### Convenience wrappers (`run*.m`)

From repo root, you can run one file per workflow:

```matlab
run('code/Asteroid_Trajectory_Selection/runGreedy.m')
run('code/Asteroid_Trajectory_Selection/runDirect.m')
run('code/Asteroid_Trajectory_Selection/runMars.m')
```

- `runGreedy.m`: optimize greedy + print best + render best-path MP4
- `runDirect.m`: optimize direct + print best + render MP4s for listed best paths
- `runMars.m`: optimize Mars-transfer + print best + render best-path MP4

Outputs:
- Greedy `.mat` files are saved to `outputs/greedy/` with a timestamp suffix
- Direct `.mat` files are saved to `outputs/direct/` with a timestamp suffix
- Mars `.mat` files are saved to `outputs/mars/` with a timestamp suffix
- Videos (e.g. `GREEDY_BEST_PATH_YYYYMMDD_HHMMSS.mp4`) are saved under `outputs/XXX/videos/`

### Other entry points

- **Direct method (non-greedy)**:
  - `code/Asteroid_Trajectory_Selection/scripts/direct/asteroid_selector.m`
  - `code/Asteroid_Trajectory_Selection/scripts/direct/find_best_path.m`
- **Mars-transfer variant**:
  - `code/Asteroid_Trajectory_Selection/scripts/mars/mars_transfer_selector.m`
  - `code/Asteroid_Trajectory_Selection/scripts/mars/find_best_path_mars.m`
  - `code/Asteroid_Trajectory_Selection/visualization/PlotBestMarsPath.m`

### Optional scripts (not used by wrappers)

- `code/Asteroid_Trajectory_Selection/visualization/optional/`
  - extra standalone plotting scripts for exploratory visuals
- `code/Asteroid_Trajectory_Selection/scripts/greedy/analysis/`
  - analysis utilities (multi-file compare, DV surface scan)

### Folder guide

- `scripts/`: top-level runnable scripts (entry points)
- `visualization/`: animation/video and plotting helpers
- `visualization/optional/`: extra one-off plotting scripts
- `core/`: objective/scoring/optimization + Δv computation routines
  - `core/greedy/`: greedy core (DV, scoring, leg optimization)
  - `core/direct/`: non-greedy direct method core
  - `core/mars/`: Mars-transfer variant core
- `utils/`: SPICE loading + small shared utilities

