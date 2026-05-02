# Rank 1: AMPHITRITE -> AURORA -> HERMIONE

## Mission Summary

- Architecture: mars (mars_flyby)
- Composition sequence: S -> C -> Unknown
- Earth launch DV [km/s]: 7.000499
- Post-launch transfer DV [km/s]: 15.903968
- Mission duration [years]: 17.5201
- Mars/Moon flyby epoch (ET): 983928330.260 — UTC: 2031 MAR 07 13:24:21.075

## Transfer DV Breakdown

| segment                        | from_body   | to_body    | depart_utc               | arrive_utc               |   dv_km_s |
|:-------------------------------|:------------|:-----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference         | Earth       | Mars       | 2028 JUL 14 00:55:21.086 | 2028 JUL 14 00:55:21.086 |   7.0005  |
| ballistic_earth_to_mars_flyby  | Earth       | Mars       | 2028 JUL 14 00:55:21.086 | 2031 MAR 07 13:24:21.075 |   0       |
| mars_gravity_assist_unpowered  | Mars        | Mars       | 2031 MAR 07 13:24:21.075 | 2031 MAR 07 13:24:21.075 |   0       |
| transfer_post_mars_flyby_to_a1 | Mars        | AMPHITRITE | 2031 MAR 07 13:24:21.075 | 2033 MAY 07 14:25:24.764 |   8.90894 |
| transfer_a1_to_a2              | AMPHITRITE  | AURORA     | 2035 JAN 21 06:31:56.515 | 2038 MAR 02 20:55:18.007 |   3.67767 |
| transfer_a2_to_a3              | AURORA      | HERMIONE   | 2041 FEB 22 20:57:05.002 | 2046 JAN 20 06:02:59.542 |   3.31735 |

## Flyby Geometry

| feasible   |   turn_angle_deg |   turn_max_deg |   periapsis_alt_km |   safe_periapsis_alt_km |   energy_residual_kms |   v_inf_in_kms |   v_inf_out_kms |   v_inf_in_x |   v_inf_in_y |   v_inf_in_z |   v_inf_out_x |   v_inf_out_y |   v_inf_out_z |
|:-----------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| True       |          7.35869 |        10.8084 |            2056.54 |                     200 |           0.000705598 |        10.7028 |         10.7021 |      10.3267 |      2.77509 |     0.456312 |       10.0788 |       3.53817 |     -0.658616 |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
