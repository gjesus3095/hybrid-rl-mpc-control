# Evaluation report: zero_policy_fixed_base

- policy: zero base command (no policy), base welded to the world (cannot move or tip, nothing terminates); arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `81a69b9+dirty`, 2026-10-16T10:48:43
- extra mass at the gripper: each scenario's own payload, drawn like the training domain randomization
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion. With the welded base nothing fails; `would have failed` counts the scenarios where it would have
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All scenarios

| metric                                                          |    ALL | fixed |  sine |   poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|-------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |        |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |    100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |      0 |     0 |     0 |      0 |        0 |     0 |       0 |
| failure rate [%]                                                |    0.0 |   0.0 |   0.0 |    0.0 |      0.0 |   0.0 |     0.0 |
| mean time of failure [s]                                        |      - |     - |     - |      - |        - |     - |       - |
| would have failed: tipped or COM left footprint at any time (n) |    194 |    20 |    21 |     30 |       67 |    27 |      29 |
| tipped at any time (n)                                          |      0 |     0 |     0 |      0 |        0 |     0 |       0 |
| COM left footprint at any time (n)                              |    194 |    20 |    21 |     30 |       67 |    27 |      29 |
| **Position error (TCP vs reference)**                           |        |       |       |        |          |       |         |
| RMS [cm]                                                        |  21.85 |  0.13 |  0.26 |   3.04 |    44.18 |  0.44 |    5.03 |
| mean [cm]                                                       |   6.72 |  0.02 |  0.04 |   0.63 |    26.22 |  0.06 |    1.22 |
| std [cm]                                                        |  20.79 |  0.13 |  0.26 |   2.97 |    35.56 |  0.44 |    4.88 |
| 95th percentile [cm]                                            |  62.84 |  0.03 |  0.04 |   3.91 |   100.41 |  0.05 |    8.14 |
| max [cm]                                                        | 115.15 |  2.53 |  5.45 |  46.85 |   115.15 |  8.26 |   39.90 |
| RMS, completed scenarios only [cm]                              |  21.85 |  0.13 |  0.26 |   3.04 |    44.18 |  0.44 |    5.03 |
| per-scenario RMS: mean [cm]                                     |   7.22 |  0.04 |  0.09 |   1.14 |    39.58 |  0.10 |    2.35 |
| per-scenario RMS: std [cm]                                      |  16.07 |  0.12 |  0.25 |   2.82 |    16.12 |  0.39 |    4.47 |
| per-scenario RMS: median [cm]                                   |   0.02 |  0.01 |  0.02 |   0.02 |    41.14 |  0.01 |    0.01 |
| per-scenario RMS: worst [cm]                                    |  65.23 |  0.81 |  1.37 |  22.69 |    65.23 |  2.72 |   21.55 |
| final error, last 0.5 s: mean [cm]                              |  13.06 |  0.01 |  0.05 |   1.44 |    72.16 |  0.15 |    4.57 |
| final error, last 0.5 s: worst [cm]                             | 115.15 |  1.02 |  3.50 |  46.41 |   115.15 |  7.46 |   39.60 |
| **Orientation error**                                           |        |       |       |        |          |       |         |
| RMS [deg]                                                       |   5.81 |  0.05 |  0.09 |   1.62 |    11.43 |  0.18 |    2.74 |
| mean [deg]                                                      |   1.83 |  0.01 |  0.04 |   0.32 |     6.75 |  0.03 |    0.61 |
| std [deg]                                                       |   5.51 |  0.05 |  0.08 |   1.59 |     9.23 |  0.18 |    2.68 |
| 95th percentile [deg]                                           |  16.25 |  0.04 |  0.07 |   1.71 |    24.49 |  0.03 |    3.64 |
| max [deg]                                                       |  41.69 |  1.53 |  1.85 |  31.23 |    39.43 |  4.42 |   41.69 |
| per-scenario RMS: mean [deg]                                    |   2.03 |  0.02 |  0.05 |   0.54 |    10.26 |  0.04 |    1.23 |
| per-scenario RMS: std [deg]                                     |   4.30 |  0.05 |  0.07 |   1.53 |     4.51 |  0.15 |    2.45 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |        |          |       |         |
| rho mean                                                        |  0.599 | 0.572 | 0.471 |  0.488 |    0.828 | 0.510 |   0.558 |
| rho std                                                         |  0.402 | 0.339 | 0.301 |  0.342 |    0.517 | 0.349 |   0.306 |
| rho max                                                         |  2.516 | 1.621 | 1.894 |  2.215 |    2.516 | 2.003 |   1.703 |
| margin to edge, mean [cm]                                       |   5.75 |  6.13 |  7.33 |   7.01 |     2.79 |  7.00 |    6.35 |
| margin to edge, std [cm]                                        |   5.89 |  5.40 |  5.00 |   5.18 |     6.93 |  5.54 |    4.92 |
| margin to edge, worst step [cm]                                 | -15.24 | -6.01 | -9.65 | -11.63 |   -15.24 | -9.53 |   -6.73 |
| per-scenario closest margin: mean [cm]                          |   2.55 |  4.42 |  4.14 |   2.47 |    -2.42 |  3.81 |    2.85 |
| per-scenario closest margin: 5th pct [cm]                       |  -7.25 | -3.37 | -2.61 |  -5.52 |   -10.93 | -5.01 |   -4.73 |
| **Base motion**                                                 |        |       |       |        |          |       |         |
| speed mean [m/s]                                                |  0.000 | 0.000 | 0.000 |  0.000 |    0.000 | 0.000 |   0.000 |
| speed max [m/s]                                                 |  0.006 | 0.001 | 0.001 |  0.001 |    0.002 | 0.001 |   0.006 |
| mean absolute yaw rate [rad/s]                                  |  0.000 | 0.000 | 0.000 |  0.000 |    0.000 | 0.000 |   0.000 |
| max roll/pitch [deg]                                            |    0.0 |   0.0 |   0.0 |    0.0 |      0.0 |   0.0 |     0.0 |
| max distance from spawn [m]                                     |   0.00 |  0.00 |  0.00 |   0.00 |     0.00 |  0.00 |    0.00 |
| **Solver**                                                      |        |       |       |        |          |       |         |
| arm QP failures (n)                                             |      0 |     0 |     0 |      0 |        0 |     0 |       0 |

## Definitions

- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, up to and including the step where it failed
- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and `COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure (the simulation keeps running), so `tipped` can exceed what ended a scenario
- **position error**: distance between the TCP and the reference at the same time (env-local world frame); RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the last 0.5 s of each scenario that did not fail
- **orientation error**: angle of the rotation between the TCP and reference orientation
- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)
- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, use rho to compare platforms
- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point
- **arm QP failures**: solver failures of the arm controller during the runs
