# Rank 3: AMPHITRITE -> AURORA -> EUNOMIA

## Mission Summary

- Architecture: mars (mars_flyby)
- Composition sequence: S -> C -> S
- Earth launch DV [km/s]: 7.000499
- Post-launch transfer DV [km/s]: 17.175774
- Mission duration [years]: 14.5351
- Mars/Moon flyby epoch (ET): 983926536.861 — UTC: 2031 MAR 07 12:54:27.676

## Transfer DV Breakdown

| segment                        | from_body   | to_body    | depart_utc               | arrive_utc               |   dv_km_s |
|:-------------------------------|:------------|:-----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference         | Earth       | Mars       | 2028 JUL 14 00:43:46.888 | 2028 JUL 14 00:43:46.888 |   7.0005  |
| ballistic_earth_to_mars_flyby  | Earth       | Mars       | 2028 JUL 14 00:43:46.888 | 2031 MAR 07 12:54:27.676 |   0       |
| mars_gravity_assist_unpowered  | Mars        | Mars       | 2031 MAR 07 12:54:27.676 | 2031 MAR 07 12:54:27.676 |   0       |
| transfer_post_mars_flyby_to_a1 | Mars        | AMPHITRITE | 2031 MAR 07 12:54:27.676 | 2033 MAY 07 13:55:31.365 |   8.9094  |
| transfer_a1_to_a2              | AMPHITRITE  | AURORA     | 2033 OCT 02 16:24:12.872 | 2037 FEB 05 15:37:00.149 |   3.30584 |
| transfer_a2_to_a3              | AURORA      | EUNOMIA    | 2040 JAN 29 15:38:47.145 | 2043 JAN 25 23:42:16.253 |   4.96053 |

## Flyby Geometry

| feasible   |   turn_angle_deg |   turn_max_deg |   periapsis_alt_km |   safe_periapsis_alt_km |   energy_residual_kms |   v_inf_in_kms |   v_inf_out_kms |   v_inf_in_x |   v_inf_in_y |   v_inf_in_z |   v_inf_out_x |   v_inf_out_y |   v_inf_out_z |
|:-----------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| True       |          7.36037 |        10.8075 |             2054.7 |                     200 |           9.78094e-05 |         10.703 |         10.7029 |      10.3273 |      2.77346 |     0.456211 |         10.08 |       3.53709 |     -0.658926 |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
