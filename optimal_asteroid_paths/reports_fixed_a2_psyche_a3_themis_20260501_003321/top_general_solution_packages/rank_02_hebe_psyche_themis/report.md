# Rank 2: HEBE -> PSYCHE -> THEMIS

## Mission Summary

- Architecture: earth (earth_gravity_assist)
- Composition sequence: S -> X/M -> C
- Earth launch DV [km/s]: 6.970335
- Post-launch transfer DV [km/s]: 36.263080
- Mission duration [years]: 16.9542

## Transfer DV Breakdown

| segment                | from_body   | to_body   | depart_utc               | arrive_utc               |   dv_km_s |
|:-----------------------|:------------|:----------|:-------------------------|:-------------------------|----------:|
| earth_launch_reference | Earth       | HEBE      | 2029 OCT 11 18:34:17.686 | 2029 OCT 11 18:34:17.686 |   6.97034 |
| transfer_earth_to_a1   | Earth       | HEBE      | 2029 OCT 11 18:34:17.686 | 2034 JUN 17 22:29:45.596 |  11.8418  |
| transfer_a1_to_a2      | HEBE        | PSYCHE    | 2036 SEP 03 18:52:41.550 | 2041 OCT 26 11:07:30.699 |  19.7915  |
| transfer_a2_to_a3      | PSYCHE      | THEMIS    | 2043 APR 12 21:25:16.117 | 2046 SEP 25 07:00:35.767 |   4.62978 |

## Flyby Geometry

| note                     | architecture   |
|:-------------------------|:---------------|
| no_flyby_in_architecture | earth          |

## Generated Files

- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.
- `time_position_velocity.csv` includes dense spacecraft/body time-state data.
- `visualization_flightpath.mp4` is generated with `visualization.py`.
