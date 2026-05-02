# Rank 2: AMPHITRITE -> AURORA -> EUROPA_52

## Mission Summary

- Architecture: mars (mars_flyby)
- Composition sequence: S -> C -> Unknown
- Earth launch DV [km/s]: 7.000383
- Post-launch transfer DV [km/s]: 17.156429
- Mission duration [years]: 14.6994
- Mars/Moon flyby epoch (ET): 983925811.054 — UTC: 2031 MAR 07 12:42:21.869

## Transfer DV Breakdown

| segment                        | from_body   | to_body    | depart_utc               | arrive_utc               |   dv_km_s |
|:-------------------------------|:------------|:-----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference         | Earth       | Mars       | 2028 JUL 14 01:08:43.814 | 2028 JUL 14 01:08:43.814 |   7.00038 |
| ballistic_earth_to_mars_flyby  | Earth       | Mars       | 2028 JUL 14 01:08:43.814 | 2031 MAR 07 12:42:21.869 |   0       |
| mars_gravity_assist_unpowered  | Mars        | Mars       | 2031 MAR 07 12:42:21.869 | 2031 MAR 07 12:42:21.869 |   0       |
| transfer_post_mars_flyby_to_a1 | Mars        | AMPHITRITE | 2031 MAR 07 12:42:21.869 | 2033 MAY 07 13:12:05.822 |   8.90864 |
| transfer_a1_to_a2              | AMPHITRITE  | AURORA     | 2034 APR 24 23:25:42.223 | 2036 APR 18 19:11:56.391 |   2.41453 |
| transfer_a2_to_a3              | AURORA      | EUROPA_52  | 2039 MAR 15 02:13:53.615 | 2043 MAR 27 00:10:07.799 |   5.83326 |

## Flyby Geometry

| feasible   |   turn_angle_deg |   turn_max_deg |   periapsis_alt_km |   safe_periapsis_alt_km |   energy_residual_kms |   v_inf_in_kms |   v_inf_out_kms |   v_inf_in_x |   v_inf_in_y |   v_inf_in_z |   v_inf_out_x |   v_inf_out_y |   v_inf_out_z |
|:-----------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| True       |          7.36133 |        10.8083 |            2054.36 |                     200 |           0.000999841 |         10.703 |          10.702 |      10.3275 |      2.77279 |      0.45631 |       10.0795 |       3.53578 |     -0.659264 |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
