# Rank 5: AMPHITRITE -> AURORA -> CYBELE

## Mission Summary

- Architecture: mars (mars_flyby)
- Composition sequence: S -> C -> Unknown
- Earth launch DV [km/s]: 7.000499
- Post-launch transfer DV [km/s]: 18.089185
- Mission duration [years]: 16.8335
- Mars/Moon flyby epoch (ET): 983924582.427 — UTC: 2031 MAR 07 12:21:53.241

## Transfer DV Breakdown

| segment                        | from_body   | to_body    | depart_utc               | arrive_utc               |   dv_km_s |
|:-------------------------------|:------------|:-----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference         | Earth       | Mars       | 2028 JUL 14 00:31:04.183 | 2028 JUL 14 00:31:04.183 |   7.0005  |
| ballistic_earth_to_mars_flyby  | Earth       | Mars       | 2028 JUL 14 00:31:04.183 | 2031 MAR 07 12:21:53.241 |   0       |
| mars_gravity_assist_unpowered  | Mars        | Mars       | 2031 MAR 07 12:21:53.241 | 2031 MAR 07 12:21:53.241 |   0       |
| transfer_post_mars_flyby_to_a1 | Mars        | AMPHITRITE | 2031 MAR 07 12:21:53.241 | 2033 MAY 07 13:22:56.931 |   8.90989 |
| transfer_a1_to_a2              | AMPHITRITE  | AURORA     | 2034 SEP 05 13:19:04.368 | 2037 OCT 16 03:42:25.861 |   3.41895 |
| transfer_a2_to_a3              | AURORA      | CYBELE     | 2040 OCT 08 03:44:12.856 | 2045 MAY 14 11:18:35.819 |   5.76034 |

## Flyby Geometry

| feasible   |   turn_angle_deg |   turn_max_deg |   periapsis_alt_km |   safe_periapsis_alt_km |   energy_residual_kms |   v_inf_in_kms |   v_inf_out_kms |   v_inf_in_x |   v_inf_in_y |   v_inf_in_z |   v_inf_out_x |   v_inf_out_y |   v_inf_out_z |
|:-----------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| True       |          7.36221 |        10.8065 |            2052.71 |                     200 |          -0.000564521 |        10.7032 |         10.7037 |       10.328 |      2.77168 |     0.456101 |       10.0813 |       3.53591 |     -0.659264 |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
