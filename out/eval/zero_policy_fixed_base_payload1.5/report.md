# Evaluation report: zero_policy_fixed_base_payload1.5

- policy: zero base command (no policy), base welded to the world (cannot move or tip, nothing terminates); arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `4734db5+dirty`, 2026-10-16T11:16:20
- extra mass at the gripper: 1.5 kg (the same 100 scenarios per trajectory type each time; only the payload differs)
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion. With the welded base nothing fails; `would have failed` counts the scenarios where it would have
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All payloads together (1.5 kg)

| metric                                                          |    ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |       |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |      0 |     0 |     0 |     0 |        0 |     0 |       0 |
| failure rate [%]                                                |    0.0 |   0.0 |   0.0 |   0.0 |      0.0 |   0.0 |     0.0 |
| mean time of failure [s]                                        |      - |     - |     - |     - |        - |     - |       - |
| would have failed: tipped or COM left footprint at any time (n) |    166 |    16 |    19 |    25 |       63 |    20 |      23 |
| tipped at any time (n)                                          |      0 |     0 |     0 |     0 |        0 |     0 |       0 |
| COM left footprint at any time (n)                              |    166 |    16 |    19 |    25 |       63 |    20 |      23 |
| **Position error (TCP vs reference)**                           |        |       |       |       |          |       |         |
| RMS [cm]                                                        |  21.85 |  0.13 |  0.26 |  3.04 |    44.18 |  0.44 |    5.03 |
| mean [cm]                                                       |   6.72 |  0.02 |  0.04 |  0.63 |    26.22 |  0.06 |    1.22 |
| std [cm]                                                        |  20.80 |  0.13 |  0.26 |  2.97 |    35.56 |  0.44 |    4.88 |
| 95th percentile [cm]                                            |  62.85 |  0.03 |  0.03 |  3.91 |   100.41 |  0.04 |    8.15 |
| max [cm]                                                        | 115.15 |  2.53 |  5.47 | 46.84 |   115.15 |  8.29 |   39.90 |
| RMS, completed scenarios only [cm]                              |  21.85 |  0.13 |  0.26 |  3.04 |    44.18 |  0.44 |    5.03 |
| per-scenario RMS: mean [cm]                                     |   7.22 |  0.04 |  0.08 |  1.14 |    39.58 |  0.09 |    2.35 |
| per-scenario RMS: std [cm]                                      |  16.07 |  0.12 |  0.25 |  2.82 |    16.12 |  0.39 |    4.47 |
| per-scenario RMS: median [cm]                                   |   0.01 |  0.01 |  0.01 |  0.02 |    41.14 |  0.01 |    0.01 |
| per-scenario RMS: worst [cm]                                    |  65.23 |  0.81 |  1.38 | 22.69 |    65.23 |  2.72 |   21.55 |
| final error, last 0.5 s: mean [cm]                              |  13.07 |  0.01 |  0.05 |  1.44 |    72.16 |  0.15 |    4.58 |
| final error, last 0.5 s: worst [cm]                             | 115.14 |  1.02 |  3.50 | 46.40 |   115.14 |  7.48 |   39.59 |
| **Orientation error**                                           |        |       |       |       |          |       |         |
| RMS [deg]                                                       |   5.81 |  0.05 |  0.09 |  1.62 |    11.43 |  0.18 |    2.75 |
| mean [deg]                                                      |   1.83 |  0.01 |  0.04 |  0.31 |     6.75 |  0.03 |    0.61 |
| std [deg]                                                       |   5.51 |  0.05 |  0.08 |  1.59 |     9.23 |  0.18 |    2.68 |
| 95th percentile [deg]                                           |  16.26 |  0.04 |  0.07 |  1.70 |    24.49 |  0.03 |    3.64 |
| max [deg]                                                       |  41.39 |  1.46 |  1.78 | 31.23 |    39.43 |  4.42 |   41.39 |
| per-scenario RMS: mean [deg]                                    |   2.03 |  0.02 |  0.05 |  0.54 |    10.26 |  0.04 |    1.23 |
| per-scenario RMS: std [deg]                                     |   4.31 |  0.05 |  0.07 |  1.53 |     4.51 |  0.15 |    2.46 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |       |          |       |         |
| rho mean                                                        |  0.560 | 0.531 | 0.441 | 0.450 |    0.777 | 0.471 |   0.526 |
| rho std                                                         |  0.376 | 0.322 | 0.283 | 0.315 |    0.485 | 0.323 |   0.281 |
| rho max                                                         |  2.174 | 1.433 | 1.724 | 1.954 |    2.174 | 1.750 |   1.592 |
| margin to edge, mean [cm]                                       |   6.22 |  6.63 |  7.65 |  7.47 |     3.45 |  7.48 |    6.70 |
| margin to edge, std [cm]                                        |   5.63 |  5.20 |  4.84 |  4.97 |     6.61 |  5.35 |    4.67 |
| margin to edge, worst step [cm]                                 | -11.89 | -4.18 | -7.81 | -9.13 |   -11.89 | -7.13 |   -5.70 |
| per-scenario closest margin: mean [cm]                          |   3.16 |  5.00 |  4.57 |  3.18 |    -1.51 |  4.42 |    3.31 |
| per-scenario closest margin: 5th pct [cm]                       |  -5.99 | -2.42 | -2.13 | -4.97 |    -8.80 | -4.70 |   -3.95 |
| **Base motion**                                                 |        |       |       |       |          |       |         |
| speed mean [m/s]                                                |  0.000 | 0.000 | 0.000 | 0.000 |    0.000 | 0.000 |   0.000 |
| speed max [m/s]                                                 |  0.004 | 0.001 | 0.001 | 0.001 |    0.002 | 0.001 |   0.004 |
| mean absolute yaw rate [rad/s]                                  |  0.000 | 0.000 | 0.000 | 0.000 |    0.000 | 0.000 |   0.000 |
| max roll/pitch [deg]                                            |    0.0 |   0.0 |   0.0 |   0.0 |      0.0 |   0.0 |     0.0 |
| max distance from spawn [m]                                     |   0.00 |  0.00 |  0.00 |  0.00 |     0.00 |  0.00 |    0.00 |
| **Solver**                                                      |        |       |       |       |          |       |         |
| arm QP failures (n)                                             |      0 |     0 |     0 |     0 |        0 |     0 |       0 |

### 1b. By payload (all trajectory types pooled)

| metric                                                          | 1.5 kg |    all |
|-----------------------------------------------------------------|-------:|-------:|
| **Outcome**                                                     |        |        |
| scenarios                                                       |    600 |    600 |
| failed: COM left footprint or tipped (n)                        |      0 |      0 |
| failure rate [%]                                                |    0.0 |    0.0 |
| mean time of failure [s]                                        |      - |      - |
| would have failed: tipped or COM left footprint at any time (n) |    166 |    166 |
| tipped at any time (n)                                          |      0 |      0 |
| COM left footprint at any time (n)                              |    166 |    166 |
| **Position error (TCP vs reference)**                           |        |        |
| RMS [cm]                                                        |  21.85 |  21.85 |
| mean [cm]                                                       |   6.72 |   6.72 |
| std [cm]                                                        |  20.80 |  20.80 |
| 95th percentile [cm]                                            |  62.85 |  62.85 |
| max [cm]                                                        | 115.15 | 115.15 |
| RMS, completed scenarios only [cm]                              |  21.85 |  21.85 |
| per-scenario RMS: mean [cm]                                     |   7.22 |   7.22 |
| per-scenario RMS: std [cm]                                      |  16.07 |  16.07 |
| per-scenario RMS: median [cm]                                   |   0.01 |   0.01 |
| per-scenario RMS: worst [cm]                                    |  65.23 |  65.23 |
| final error, last 0.5 s: mean [cm]                              |  13.07 |  13.07 |
| final error, last 0.5 s: worst [cm]                             | 115.14 | 115.14 |
| **Orientation error**                                           |        |        |
| RMS [deg]                                                       |   5.81 |   5.81 |
| mean [deg]                                                      |   1.83 |   1.83 |
| std [deg]                                                       |   5.51 |   5.51 |
| 95th percentile [deg]                                           |  16.26 |  16.26 |
| max [deg]                                                       |  41.39 |  41.39 |
| per-scenario RMS: mean [deg]                                    |   2.03 |   2.03 |
| per-scenario RMS: std [deg]                                     |   4.31 |   4.31 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |        |
| rho mean                                                        |  0.560 |  0.560 |
| rho std                                                         |  0.376 |  0.376 |
| rho max                                                         |  2.174 |  2.174 |
| margin to edge, mean [cm]                                       |   6.22 |   6.22 |
| margin to edge, std [cm]                                        |   5.63 |   5.63 |
| margin to edge, worst step [cm]                                 | -11.89 | -11.89 |
| per-scenario closest margin: mean [cm]                          |   3.16 |   3.16 |
| per-scenario closest margin: 5th pct [cm]                       |  -5.99 |  -5.99 |
| **Base motion**                                                 |        |        |
| speed mean [m/s]                                                |  0.000 |  0.000 |
| speed max [m/s]                                                 |  0.004 |  0.004 |
| mean absolute yaw rate [rad/s]                                  |  0.000 |  0.000 |
| max roll/pitch [deg]                                            |    0.0 |    0.0 |
| max distance from spawn [m]                                     |   0.00 |   0.00 |
| **Solver**                                                      |        |        |
| arm QP failures (n)                                             |      0 |      0 |

## 2. Extra mass 1.5 kg

| metric                                                          |    ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |       |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |      0 |     0 |     0 |     0 |        0 |     0 |       0 |
| failure rate [%]                                                |    0.0 |   0.0 |   0.0 |   0.0 |      0.0 |   0.0 |     0.0 |
| mean time of failure [s]                                        |      - |     - |     - |     - |        - |     - |       - |
| would have failed: tipped or COM left footprint at any time (n) |    166 |    16 |    19 |    25 |       63 |    20 |      23 |
| tipped at any time (n)                                          |      0 |     0 |     0 |     0 |        0 |     0 |       0 |
| COM left footprint at any time (n)                              |    166 |    16 |    19 |    25 |       63 |    20 |      23 |
| **Position error (TCP vs reference)**                           |        |       |       |       |          |       |         |
| RMS [cm]                                                        |  21.85 |  0.13 |  0.26 |  3.04 |    44.18 |  0.44 |    5.03 |
| mean [cm]                                                       |   6.72 |  0.02 |  0.04 |  0.63 |    26.22 |  0.06 |    1.22 |
| std [cm]                                                        |  20.80 |  0.13 |  0.26 |  2.97 |    35.56 |  0.44 |    4.88 |
| 95th percentile [cm]                                            |  62.85 |  0.03 |  0.03 |  3.91 |   100.41 |  0.04 |    8.15 |
| max [cm]                                                        | 115.15 |  2.53 |  5.47 | 46.84 |   115.15 |  8.29 |   39.90 |
| RMS, completed scenarios only [cm]                              |  21.85 |  0.13 |  0.26 |  3.04 |    44.18 |  0.44 |    5.03 |
| per-scenario RMS: mean [cm]                                     |   7.22 |  0.04 |  0.08 |  1.14 |    39.58 |  0.09 |    2.35 |
| per-scenario RMS: std [cm]                                      |  16.07 |  0.12 |  0.25 |  2.82 |    16.12 |  0.39 |    4.47 |
| per-scenario RMS: median [cm]                                   |   0.01 |  0.01 |  0.01 |  0.02 |    41.14 |  0.01 |    0.01 |
| per-scenario RMS: worst [cm]                                    |  65.23 |  0.81 |  1.38 | 22.69 |    65.23 |  2.72 |   21.55 |
| final error, last 0.5 s: mean [cm]                              |  13.07 |  0.01 |  0.05 |  1.44 |    72.16 |  0.15 |    4.58 |
| final error, last 0.5 s: worst [cm]                             | 115.14 |  1.02 |  3.50 | 46.40 |   115.14 |  7.48 |   39.59 |
| **Orientation error**                                           |        |       |       |       |          |       |         |
| RMS [deg]                                                       |   5.81 |  0.05 |  0.09 |  1.62 |    11.43 |  0.18 |    2.75 |
| mean [deg]                                                      |   1.83 |  0.01 |  0.04 |  0.31 |     6.75 |  0.03 |    0.61 |
| std [deg]                                                       |   5.51 |  0.05 |  0.08 |  1.59 |     9.23 |  0.18 |    2.68 |
| 95th percentile [deg]                                           |  16.26 |  0.04 |  0.07 |  1.70 |    24.49 |  0.03 |    3.64 |
| max [deg]                                                       |  41.39 |  1.46 |  1.78 | 31.23 |    39.43 |  4.42 |   41.39 |
| per-scenario RMS: mean [deg]                                    |   2.03 |  0.02 |  0.05 |  0.54 |    10.26 |  0.04 |    1.23 |
| per-scenario RMS: std [deg]                                     |   4.31 |  0.05 |  0.07 |  1.53 |     4.51 |  0.15 |    2.46 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |       |          |       |         |
| rho mean                                                        |  0.560 | 0.531 | 0.441 | 0.450 |    0.777 | 0.471 |   0.526 |
| rho std                                                         |  0.376 | 0.322 | 0.283 | 0.315 |    0.485 | 0.323 |   0.281 |
| rho max                                                         |  2.174 | 1.433 | 1.724 | 1.954 |    2.174 | 1.750 |   1.592 |
| margin to edge, mean [cm]                                       |   6.22 |  6.63 |  7.65 |  7.47 |     3.45 |  7.48 |    6.70 |
| margin to edge, std [cm]                                        |   5.63 |  5.20 |  4.84 |  4.97 |     6.61 |  5.35 |    4.67 |
| margin to edge, worst step [cm]                                 | -11.89 | -4.18 | -7.81 | -9.13 |   -11.89 | -7.13 |   -5.70 |
| per-scenario closest margin: mean [cm]                          |   3.16 |  5.00 |  4.57 |  3.18 |    -1.51 |  4.42 |    3.31 |
| per-scenario closest margin: 5th pct [cm]                       |  -5.99 | -2.42 | -2.13 | -4.97 |    -8.80 | -4.70 |   -3.95 |
| **Base motion**                                                 |        |       |       |       |          |       |         |
| speed mean [m/s]                                                |  0.000 | 0.000 | 0.000 | 0.000 |    0.000 | 0.000 |   0.000 |
| speed max [m/s]                                                 |  0.004 | 0.001 | 0.001 | 0.001 |    0.002 | 0.001 |   0.004 |
| mean absolute yaw rate [rad/s]                                  |  0.000 | 0.000 | 0.000 | 0.000 |    0.000 | 0.000 |   0.000 |
| max roll/pitch [deg]                                            |    0.0 |   0.0 |   0.0 |   0.0 |      0.0 |   0.0 |     0.0 |
| max distance from spawn [m]                                     |   0.00 |  0.00 |  0.00 |  0.00 |     0.00 |  0.00 |    0.00 |
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
