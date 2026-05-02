# Rank 4: AMPHITRITE -> AURORA -> HYGIEA

## Mission Summary

- Architecture: mars (mars_flyby)
- Composition sequence: S -> C -> C
- Earth launch DV [km/s]: 7.000499
- Post-launch transfer DV [km/s]: 17.931688
- Mission duration [years]: 11.2507
- Mars/Moon flyby epoch (ET): 983929050.659 — UTC: 2031 MAR 07 13:36:21.473

## Transfer DV Breakdown

| segment                        | from_body   | to_body    | depart_utc               | arrive_utc               |   dv_km_s |
|:-------------------------------|:------------|:-----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference         | Earth       | Mars       | 2028 JUL 14 01:00:04.421 | 2028 JUL 14 01:00:04.421 |   7.0005  |
| ballistic_earth_to_mars_flyby  | Earth       | Mars       | 2028 JUL 14 01:00:04.421 | 2031 MAR 07 13:36:21.473 |   0       |
| mars_gravity_assist_unpowered  | Mars        | Mars       | 2031 MAR 07 13:36:21.473 | 2031 MAR 07 13:36:21.473 |   0       |
| transfer_post_mars_flyby_to_a1 | Mars        | AMPHITRITE | 2031 MAR 07 13:36:21.473 | 2033 MAY 07 14:37:25.163 |   8.90876 |
| transfer_a1_to_a2              | AMPHITRITE  | AURORA     | 2034 JAN 18 23:29:39.467 | 2037 JAN 25 19:16:47.333 |   2.63519 |
| transfer_a2_to_a3              | AURORA      | HYGIEA     | 2038 MAY 01 10:23:20.375 | 2039 OCT 14 08:50:35.508 |   6.38774 |

## Flyby Geometry

| feasible   |   turn_angle_deg |   turn_max_deg |   periapsis_alt_km |   safe_periapsis_alt_km |   energy_residual_kms |   v_inf_in_kms |   v_inf_out_kms |   v_inf_in_x |   v_inf_in_y |   v_inf_in_z |   v_inf_out_x |   v_inf_out_y |   v_inf_out_z |
|:-----------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| True       |          7.35801 |        10.8088 |            2057.27 |                     200 |           0.000949652 |        10.7027 |         10.7018 |      10.3264 |      2.77575 |     0.456353 |       10.0783 |        3.5386 |     -0.658491 |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
