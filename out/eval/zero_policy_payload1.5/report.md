# Evaluation report: zero_policy_payload1.5

- policy: zero base command (no policy); arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `4734db5+dirty`, 2026-10-16T11:18:24
- extra mass at the gripper: 1.5 kg (the same 100 scenarios per trajectory type each time; only the payload differs)
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All payloads together (1.5 kg)

| metric                                                          |    ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |       |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| failure rate [%]                                                |   45.0 |  33.0 |  38.0 |  42.0 |     78.0 |  38.0 |    41.0 |
| mean time of failure [s]                                        |   3.24 |  2.00 |  2.74 |  2.72 |     4.20 |  2.51 |    4.11 |
| would have failed: tipped or COM left footprint at any time (n) |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| tipped at any time (n)                                          |    243 |    32 |    33 |    35 |       73 |    33 |      37 |
| COM left footprint at any time (n)                              |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| **Position error (TCP vs reference)**                           |        |       |       |       |          |       |         |
| RMS [cm]                                                        |  13.57 |  1.65 |  1.91 |  3.69 |    30.85 |  1.85 |    3.62 |
| mean [cm]                                                       |   3.42 |  0.84 |  1.10 |  1.33 |    12.83 |  1.08 |    1.62 |
| std [cm]                                                        |  13.13 |  1.42 |  1.56 |  3.45 |    28.06 |  1.51 |    3.23 |
| 95th percentile [cm]                                            |   8.78 |  3.89 |  4.50 |  5.37 |    87.35 |  4.32 |    7.09 |
| max [cm]                                                        | 115.10 |  9.84 | 13.55 | 47.77 |   115.10 | 10.27 |   32.94 |
| RMS, completed scenarios only [cm]                              |  16.16 |  1.55 |  1.66 |  3.68 |    49.30 |  1.71 |    3.85 |
| per-scenario RMS: mean [cm]                                     |   3.41 |  1.23 |  1.54 |  1.92 |    11.90 |  1.47 |    2.37 |
| per-scenario RMS: std [cm]                                      |   8.90 |  1.17 |  1.19 |  2.90 |    19.25 |  1.11 |    2.28 |
| per-scenario RMS: median [cm]                                   |   1.52 |  1.02 |  1.42 |  1.19 |     2.54 |  1.33 |    1.79 |
| per-scenario RMS: worst [cm]                                    |  65.15 |  5.55 |  4.46 | 24.12 |    65.15 |  4.65 |   12.76 |
| final error, last 0.5 s: mean [cm]                              |   7.38 |  0.93 |  1.23 |  2.58 |    82.09 |  1.33 |    4.40 |
| final error, last 0.5 s: worst [cm]                             | 114.94 |  7.91 |  5.47 | 47.50 |   114.94 |  7.60 |   23.36 |
| **Orientation error**                                           |        |       |       |       |          |       |         |
| RMS [deg]                                                       |   3.75 |  0.80 |  1.23 |  2.04 |     8.09 |  0.92 |    1.85 |
| mean [deg]                                                      |   1.23 |  0.45 |  0.74 |  0.79 |     3.59 |  0.57 |    0.79 |
| std [deg]                                                       |   3.55 |  0.66 |  0.98 |  1.88 |     7.25 |  0.71 |    1.68 |
| 95th percentile [deg]                                           |   3.83 |  1.88 |  2.58 |  2.87 |    23.08 |  1.94 |    2.71 |
| max [deg]                                                       |  39.84 |  7.13 |  8.35 | 32.06 |    39.84 |  7.37 |   32.25 |
| per-scenario RMS: mean [deg]                                    |   1.38 |  0.67 |  1.00 |  1.12 |     3.52 |  0.78 |    1.22 |
| per-scenario RMS: std [deg]                                     |   2.43 |  0.56 |  0.77 |  1.61 |     4.95 |  0.54 |    1.19 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |       |          |       |         |
| rho mean                                                        |  0.400 | 0.407 | 0.359 | 0.360 |    0.437 | 0.375 |   0.427 |
| rho std                                                         |  0.215 | 0.209 | 0.207 | 0.221 |    0.234 | 0.221 |   0.192 |
| rho max                                                         |  1.106 | 1.047 | 1.068 | 1.059 |    1.055 | 1.106 |   1.026 |
| margin to edge, mean [cm]                                       |   8.41 |  8.32 |  8.85 |  8.66 |     7.98 |  8.86 |    8.17 |
| margin to edge, std [cm]                                        |   4.24 |  4.27 |  4.25 |  4.24 |     4.20 |  4.60 |    4.01 |
| margin to edge, worst step [cm]                                 |  -1.04 | -0.45 | -0.65 | -0.67 |    -0.81 | -1.04 |   -0.26 |
| per-scenario closest margin: mean [cm]                          |   3.59 |  4.82 |  4.38 |  3.47 |     1.07 |  4.35 |    3.47 |
| per-scenario closest margin: 5th pct [cm]                       |  -0.33 | -0.26 | -0.30 | -0.38 |    -0.47 | -0.31 |   -0.19 |
| **Base motion**                                                 |        |       |       |       |          |       |         |
| speed mean [m/s]                                                |  0.013 | 0.011 | 0.014 | 0.012 |    0.014 | 0.015 |   0.012 |
| speed max [m/s]                                                 |  0.243 | 0.193 | 0.143 | 0.243 |    0.171 | 0.154 |   0.184 |
| mean absolute yaw rate [rad/s]                                  |  0.393 | 0.348 | 0.397 | 0.336 |    0.431 | 0.435 |   0.399 |
| max roll/pitch [deg]                                            |   29.5 |   6.2 |  15.3 |  28.7 |      6.8 |  14.2 |    29.5 |
| max distance from spawn [m]                                     |   0.13 |  0.09 |  0.11 |  0.13 |     0.11 |  0.08 |    0.13 |
| **Solver**                                                      |        |       |       |       |          |       |         |
| arm QP failures (n)                                             |      1 |     0 |     0 |     0 |        1 |     0 |       0 |

### 1b. By payload (all trajectory types pooled)

| metric                                                          | 1.5 kg |    all |
|-----------------------------------------------------------------|-------:|-------:|
| **Outcome**                                                     |        |        |
| scenarios                                                       |    600 |    600 |
| failed: COM left footprint or tipped (n)                        |    270 |    270 |
| failure rate [%]                                                |   45.0 |   45.0 |
| mean time of failure [s]                                        |   3.24 |   3.24 |
| would have failed: tipped or COM left footprint at any time (n) |    270 |    270 |
| tipped at any time (n)                                          |    243 |    243 |
| COM left footprint at any time (n)                              |    270 |    270 |
| **Position error (TCP vs reference)**                           |        |        |
| RMS [cm]                                                        |  13.57 |  13.57 |
| mean [cm]                                                       |   3.42 |   3.42 |
| std [cm]                                                        |  13.13 |  13.13 |
| 95th percentile [cm]                                            |   8.78 |   8.78 |
| max [cm]                                                        | 115.10 | 115.10 |
| RMS, completed scenarios only [cm]                              |  16.16 |  16.16 |
| per-scenario RMS: mean [cm]                                     |   3.41 |   3.41 |
| per-scenario RMS: std [cm]                                      |   8.90 |   8.90 |
| per-scenario RMS: median [cm]                                   |   1.52 |   1.52 |
| per-scenario RMS: worst [cm]                                    |  65.15 |  65.15 |
| final error, last 0.5 s: mean [cm]                              |   7.38 |   7.38 |
| final error, last 0.5 s: worst [cm]                             | 114.94 | 114.94 |
| **Orientation error**                                           |        |        |
| RMS [deg]                                                       |   3.75 |   3.75 |
| mean [deg]                                                      |   1.23 |   1.23 |
| std [deg]                                                       |   3.55 |   3.55 |
| 95th percentile [deg]                                           |   3.83 |   3.83 |
| max [deg]                                                       |  39.84 |  39.84 |
| per-scenario RMS: mean [deg]                                    |   1.38 |   1.38 |
| per-scenario RMS: std [deg]                                     |   2.43 |   2.43 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |        |
| rho mean                                                        |  0.400 |  0.400 |
| rho std                                                         |  0.215 |  0.215 |
| rho max                                                         |  1.106 |  1.106 |
| margin to edge, mean [cm]                                       |   8.41 |   8.41 |
| margin to edge, std [cm]                                        |   4.24 |   4.24 |
| margin to edge, worst step [cm]                                 |  -1.04 |  -1.04 |
| per-scenario closest margin: mean [cm]                          |   3.59 |   3.59 |
| per-scenario closest margin: 5th pct [cm]                       |  -0.33 |  -0.33 |
| **Base motion**                                                 |        |        |
| speed mean [m/s]                                                |  0.013 |  0.013 |
| speed max [m/s]                                                 |  0.243 |  0.243 |
| mean absolute yaw rate [rad/s]                                  |  0.393 |  0.393 |
| max roll/pitch [deg]                                            |   29.5 |   29.5 |
| max distance from spawn [m]                                     |   0.13 |   0.13 |
| **Solver**                                                      |        |        |
| arm QP failures (n)                                             |      1 |      1 |

## 2. Extra mass 1.5 kg

| metric                                                          |    ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|-------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |        |       |       |       |          |       |         |
| scenarios                                                       |    600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| failure rate [%]                                                |   45.0 |  33.0 |  38.0 |  42.0 |     78.0 |  38.0 |    41.0 |
| mean time of failure [s]                                        |   3.24 |  2.00 |  2.74 |  2.72 |     4.20 |  2.51 |    4.11 |
| would have failed: tipped or COM left footprint at any time (n) |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| tipped at any time (n)                                          |    243 |    32 |    33 |    35 |       73 |    33 |      37 |
| COM left footprint at any time (n)                              |    270 |    33 |    38 |    42 |       78 |    38 |      41 |
| **Position error (TCP vs reference)**                           |        |       |       |       |          |       |         |
| RMS [cm]                                                        |  13.57 |  1.65 |  1.91 |  3.69 |    30.85 |  1.85 |    3.62 |
| mean [cm]                                                       |   3.42 |  0.84 |  1.10 |  1.33 |    12.83 |  1.08 |    1.62 |
| std [cm]                                                        |  13.13 |  1.42 |  1.56 |  3.45 |    28.06 |  1.51 |    3.23 |
| 95th percentile [cm]                                            |   8.78 |  3.89 |  4.50 |  5.37 |    87.35 |  4.32 |    7.09 |
| max [cm]                                                        | 115.10 |  9.84 | 13.55 | 47.77 |   115.10 | 10.27 |   32.94 |
| RMS, completed scenarios only [cm]                              |  16.16 |  1.55 |  1.66 |  3.68 |    49.30 |  1.71 |    3.85 |
| per-scenario RMS: mean [cm]                                     |   3.41 |  1.23 |  1.54 |  1.92 |    11.90 |  1.47 |    2.37 |
| per-scenario RMS: std [cm]                                      |   8.90 |  1.17 |  1.19 |  2.90 |    19.25 |  1.11 |    2.28 |
| per-scenario RMS: median [cm]                                   |   1.52 |  1.02 |  1.42 |  1.19 |     2.54 |  1.33 |    1.79 |
| per-scenario RMS: worst [cm]                                    |  65.15 |  5.55 |  4.46 | 24.12 |    65.15 |  4.65 |   12.76 |
| final error, last 0.5 s: mean [cm]                              |   7.38 |  0.93 |  1.23 |  2.58 |    82.09 |  1.33 |    4.40 |
| final error, last 0.5 s: worst [cm]                             | 114.94 |  7.91 |  5.47 | 47.50 |   114.94 |  7.60 |   23.36 |
| **Orientation error**                                           |        |       |       |       |          |       |         |
| RMS [deg]                                                       |   3.75 |  0.80 |  1.23 |  2.04 |     8.09 |  0.92 |    1.85 |
| mean [deg]                                                      |   1.23 |  0.45 |  0.74 |  0.79 |     3.59 |  0.57 |    0.79 |
| std [deg]                                                       |   3.55 |  0.66 |  0.98 |  1.88 |     7.25 |  0.71 |    1.68 |
| 95th percentile [deg]                                           |   3.83 |  1.88 |  2.58 |  2.87 |    23.08 |  1.94 |    2.71 |
| max [deg]                                                       |  39.84 |  7.13 |  8.35 | 32.06 |    39.84 |  7.37 |   32.25 |
| per-scenario RMS: mean [deg]                                    |   1.38 |  0.67 |  1.00 |  1.12 |     3.52 |  0.78 |    1.22 |
| per-scenario RMS: std [deg]                                     |   2.43 |  0.56 |  0.77 |  1.61 |     4.95 |  0.54 |    1.19 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |       |       |          |       |         |
| rho mean                                                        |  0.400 | 0.407 | 0.359 | 0.360 |    0.437 | 0.375 |   0.427 |
| rho std                                                         |  0.215 | 0.209 | 0.207 | 0.221 |    0.234 | 0.221 |   0.192 |
| rho max                                                         |  1.106 | 1.047 | 1.068 | 1.059 |    1.055 | 1.106 |   1.026 |
| margin to edge, mean [cm]                                       |   8.41 |  8.32 |  8.85 |  8.66 |     7.98 |  8.86 |    8.17 |
| margin to edge, std [cm]                                        |   4.24 |  4.27 |  4.25 |  4.24 |     4.20 |  4.60 |    4.01 |
| margin to edge, worst step [cm]                                 |  -1.04 | -0.45 | -0.65 | -0.67 |    -0.81 | -1.04 |   -0.26 |
| per-scenario closest margin: mean [cm]                          |   3.59 |  4.82 |  4.38 |  3.47 |     1.07 |  4.35 |    3.47 |
| per-scenario closest margin: 5th pct [cm]                       |  -0.33 | -0.26 | -0.30 | -0.38 |    -0.47 | -0.31 |   -0.19 |
| **Base motion**                                                 |        |       |       |       |          |       |         |
| speed mean [m/s]                                                |  0.013 | 0.011 | 0.014 | 0.012 |    0.014 | 0.015 |   0.012 |
| speed max [m/s]                                                 |  0.243 | 0.193 | 0.143 | 0.243 |    0.171 | 0.154 |   0.184 |
| mean absolute yaw rate [rad/s]                                  |  0.393 | 0.348 | 0.397 | 0.336 |    0.431 | 0.435 |   0.399 |
| max roll/pitch [deg]                                            |   29.5 |   6.2 |  15.3 |  28.7 |      6.8 |  14.2 |    29.5 |
| max distance from spawn [m]                                     |   0.13 |  0.09 |  0.11 |  0.13 |     0.11 |  0.08 |    0.13 |
| **Solver**                                                      |        |       |       |       |          |       |         |
| arm QP failures (n)                                             |      1 |     0 |     0 |     0 |        1 |     0 |       0 |

## Definitions

- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, up to and including the step where it failed
- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and `COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure (the simulation keeps running), so `tipped` can exceed what ended a scenario
- **position error**: distance between the TCP and the reference at the same time (env-local world frame); RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the last 0.5 s of each scenario that did not fail
- **orientation error**: angle of the rotation between the TCP and reference orientation
- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)
- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, use rho to compare platforms
- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point
- **arm QP failures**: solver failures of the arm controller during the runs
