# Evaluation report: zero_policy

- policy: zero base command (no policy); arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `81a69b9+dirty`, 2026-10-16T10:50:45
- extra mass at the gripper: each scenario's own payload, drawn like the training domain randomization
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All scenarios

| metric                                                          |    ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |       |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |    291 |    38 |    42 |    44 |       82 |    40 |      45 |
| failure rate [%]                                                |   48.5 |  38.0 |  42.0 |  44.0 |     82.0 |  40.0 |    45.0 |
| mean time of failure [s]                                        |   3.09 |  2.08 |  2.67 |  2.59 |     4.16 |  2.35 |    3.55 |
| would have failed: tipped or COM left footprint at any time (n) |    291 |    38 |    42 |    44 |       82 |    40 |      45 |
| tipped at any time (n)                                          |    266 |    36 |    37 |    37 |       75 |    37 |      44 |
| COM left footprint at any time (n)                              |    291 |    38 |    42 |    44 |       82 |    40 |      45 |
| **Position error (TCP vs reference)**                           |        |       |       |       |          |       |         |
| RMS [cm]                                                        |  11.93 |  1.66 |  1.99 |  3.67 |    27.11 |  1.82 |    3.84 |
| mean [cm]                                                       |   3.08 |  0.92 |  1.16 |  1.45 |    10.80 |  1.10 |    1.73 |
| std [cm]                                                        |  11.53 |  1.38 |  1.61 |  3.38 |    24.87 |  1.46 |    3.43 |
| 95th percentile [cm]                                            |   8.07 |  4.01 |  4.51 |  5.37 |    77.01 |  4.07 |    7.14 |
| max [cm]                                                        | 114.97 | 10.70 | 14.93 | 47.76 |   114.97 | 10.42 |   32.93 |
| RMS, completed scenarios only [cm]                              |  13.62 |  1.37 |  1.72 |  3.77 |    44.66 |  1.65 |    4.04 |
| per-scenario RMS: mean [cm]                                     |   3.15 |  1.28 |  1.61 |  1.95 |    10.16 |  1.48 |    2.39 |
| per-scenario RMS: std [cm]                                      |   7.67 |  1.11 |  1.23 |  2.79 |    16.60 |  1.10 |    2.38 |
| per-scenario RMS: median [cm]                                   |   1.57 |  1.07 |  1.38 |  1.36 |     2.58 |  1.33 |    1.74 |
| per-scenario RMS: worst [cm]                                    |  65.16 |  5.29 |  4.82 | 24.50 |    65.16 |  4.81 |   12.76 |
| final error, last 0.5 s: mean [cm]                              |   6.11 |  0.93 |  1.61 |  2.78 |    68.86 |  1.42 |    4.69 |
| final error, last 0.5 s: worst [cm]                             | 114.95 |  7.11 |  8.63 | 47.16 |   114.95 |  9.87 |   23.35 |
| **Orientation error**                                           |        |       |       |       |          |       |         |
| RMS [deg]                                                       |   3.36 |  0.83 |  1.25 |  2.00 |     7.10 |  0.93 |    2.01 |
| mean [deg]                                                      |   1.17 |  0.50 |  0.78 |  0.85 |     3.08 |  0.60 |    0.87 |
| std [deg]                                                       |   3.15 |  0.66 |  0.98 |  1.81 |     6.40 |  0.71 |    1.81 |
| 95th percentile [deg]                                           |   3.66 |  1.90 |  2.69 |  3.00 |    21.22 |  2.00 |    3.08 |
| max [deg]                                                       |  35.13 |  5.48 |  7.37 | 31.76 |    35.13 |  5.93 |   32.50 |
| per-scenario RMS: mean [deg]                                    |   1.35 |  0.73 |  1.04 |  1.17 |     3.05 |  0.81 |    1.30 |
| per-scenario RMS: std [deg]                                     |   2.10 |  0.54 |  0.75 |  1.52 |     4.22 |  0.51 |    1.25 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |       |          |       |         |
| rho mean                                                        |  0.412 | 0.428 | 0.376 | 0.380 |    0.449 | 0.391 |   0.425 |
| rho std                                                         |  0.216 | 0.209 | 0.214 | 0.228 |    0.239 | 0.221 |   0.185 |
| rho max                                                         |  1.144 | 1.039 | 1.062 | 1.144 |    1.046 | 1.033 |   1.038 |
| margin to edge, mean [cm]                                       |   8.30 |  8.10 |  8.71 |  8.39 |     7.80 |  8.69 |    8.29 |
| margin to edge, std [cm]                                        |   4.26 |  4.33 |  4.32 |  4.27 |     4.22 |  4.56 |    3.99 |
| margin to edge, worst step [cm]                                 |  -1.50 | -0.42 | -0.60 | -1.50 |    -0.71 | -0.36 |   -0.92 |
| per-scenario closest margin: mean [cm]                          |   3.25 |  4.38 |  4.02 |  3.09 |     0.72 |  4.08 |    3.19 |
| per-scenario closest margin: 5th pct [cm]                       |  -0.39 | -0.33 | -0.32 | -0.46 |    -0.44 | -0.29 |   -0.27 |
| **Base motion**                                                 |        |       |       |       |          |       |         |
| speed mean [m/s]                                                |  0.014 | 0.014 | 0.015 | 0.014 |    0.015 | 0.016 |   0.014 |
| speed max [m/s]                                                 |  0.249 | 0.170 | 0.164 | 0.249 |    0.187 | 0.202 |   0.199 |
| mean absolute yaw rate [rad/s]                                  |  0.427 | 0.413 | 0.422 | 0.391 |    0.453 | 0.469 |   0.419 |
| max roll/pitch [deg]                                            |   14.3 |  13.1 |  14.3 |  13.3 |     10.0 |  13.9 |    13.2 |
| max distance from spawn [m]                                     |   0.19 |  0.07 |  0.09 |  0.16 |     0.10 |  0.08 |    0.19 |
| **Solver**                                                      |        |       |       |       |          |       |         |
| arm QP failures (n)                                             |      0 |     0 |     0 |     0 |        0 |     0 |       0 |

## Definitions

- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, up to and including the step where it failed
- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and `COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure (the simulation keeps running), so `tipped` can exceed what ended a scenario
- **position error**: distance between the TCP and the reference at the same time (env-local world frame); RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the last 0.5 s of each scenario that did not fail
- **orientation error**: angle of the rotation between the TCP and reference orientation
- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)
- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, use rho to compare platforms
- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point
- **arm QP failures**: solver failures of the arm controller during the runs
