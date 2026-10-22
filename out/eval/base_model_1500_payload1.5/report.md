# Evaluation report: base_model_1500_payload1.5

- checkpoint: `/home/u0110021/hybrid-rl-mpc-control/logs/base_/model_1500.pt`; arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `4734db5+dirty`, 2026-10-16T11:27:38
- extra mass at the gripper: 1.5 kg (the same 100 scenarios per trajectory type each time; only the payload differs). Training used 0-3 kg.
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All payloads together (1.5 kg)

| metric                                                          |   ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |       |       |       |       |          |       |         |
| scenarios                                                       |   600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| failure rate [%]                                                |   0.5 |   0.0 |   1.0 |   0.0 |      1.0 |   0.0 |     1.0 |
| mean time of failure [s]                                        |  4.03 |     - |  2.37 |     - |     3.27 |     - |    6.47 |
| would have failed: tipped or COM left footprint at any time (n) |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| tipped at any time (n)                                          |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| COM left footprint at any time (n)                              |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| **Position error (TCP vs reference)**                           |       |       |       |       |          |       |         |
| RMS [cm]                                                        |  3.05 |  2.85 |  2.99 |  3.88 |     3.81 |  2.34 |    2.02 |
| mean [cm]                                                       |  2.03 |  1.98 |  2.14 |  2.69 |     2.38 |  1.73 |    1.47 |
| std [cm]                                                        |  2.27 |  2.05 |  2.08 |  2.80 |     2.97 |  1.57 |    1.38 |
| 95th percentile [cm]                                            |  6.62 |  6.35 |  5.89 |  8.87 |     8.74 |  4.58 |    3.87 |
| max [cm]                                                        | 26.58 | 15.61 | 18.34 | 18.54 |    26.58 | 11.69 |   13.29 |
| RMS, completed scenarios only [cm]                              |  3.02 |  2.85 |  2.98 |  3.88 |     3.74 |  2.34 |    1.95 |
| per-scenario RMS: mean [cm]                                     |  2.46 |  2.26 |  2.52 |  3.17 |     3.08 |  1.99 |    1.77 |
| per-scenario RMS: std [cm]                                      |  1.87 |  1.74 |  1.60 |  2.24 |     2.48 |  1.18 |    1.08 |
| per-scenario RMS: median [cm]                                   |  1.87 |  1.66 |  2.01 |  2.32 |     2.04 |  1.75 |    1.55 |
| per-scenario RMS: worst [cm]                                    | 12.44 |  9.83 |  9.58 | 10.96 |    12.44 |  6.14 |    6.60 |
| final error, last 0.5 s: mean [cm]                              |  2.10 |  2.27 |  2.21 |  3.03 |     2.15 |  1.70 |    1.23 |
| final error, last 0.5 s: worst [cm]                             | 18.23 | 15.06 | 15.17 | 14.04 |    18.23 | 10.97 |   10.72 |
| **Orientation error**                                           |       |       |       |       |          |       |         |
| RMS [deg]                                                       |  1.82 |  1.62 |  2.25 |  2.09 |     2.02 |  1.63 |    1.38 |
| mean [deg]                                                      |  1.30 |  1.30 |  1.69 |  1.62 |     1.34 |  1.24 |    0.93 |
| std [deg]                                                       |  1.28 |  0.96 |  1.49 |  1.32 |     1.51 |  1.05 |    1.03 |
| 95th percentile [deg]                                           |  3.62 |  3.19 |  4.33 |  4.15 |     4.25 |  3.21 |    2.92 |
| max [deg]                                                       | 14.88 |  4.59 | 14.45 | 11.91 |    14.88 | 11.23 |   11.85 |
| per-scenario RMS: mean [deg]                                    |  1.70 |  1.54 |  2.06 |  1.94 |     1.83 |  1.55 |    1.27 |
| per-scenario RMS: std [deg]                                     |  0.79 |  0.50 |  0.91 |  0.77 |     0.98 |  0.55 |    0.59 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |       |       |       |       |          |       |         |
| rho mean                                                        | 0.131 | 0.106 | 0.142 | 0.152 |    0.160 | 0.106 |   0.111 |
| rho std                                                         | 0.095 | 0.072 | 0.098 | 0.099 |    0.111 | 0.071 |   0.083 |
| rho max                                                         | 1.029 | 0.379 | 1.029 | 0.688 |    1.021 | 0.503 |   1.025 |
| margin to edge, mean [cm]                                       | 11.08 | 11.30 | 10.99 | 10.94 |    10.74 | 11.41 |   11.25 |
| margin to edge, std [cm]                                        |  3.78 |  3.65 |  3.85 |  3.87 |     3.82 |  3.85 |    3.68 |
| margin to edge, worst step [cm]                                 | -0.29 |  5.94 | -0.29 |  3.38 |    -0.20 |  4.72 |   -0.24 |
| per-scenario closest margin: mean [cm]                          |  9.45 |  9.88 |  9.19 |  9.19 |     8.98 |  9.90 |    9.56 |
| per-scenario closest margin: 5th pct [cm]                       |  5.57 |  6.25 |  5.34 |  5.47 |     4.90 |  6.25 |    5.59 |
| **Base motion**                                                 |       |       |       |       |          |       |         |
| speed mean [m/s]                                                | 0.164 | 0.102 | 0.141 | 0.182 |    0.200 | 0.134 |   0.179 |
| speed max [m/s]                                                 | 0.875 | 0.532 | 0.608 | 0.875 |    0.721 | 0.425 |   0.612 |
| mean absolute yaw rate [rad/s]                                  | 0.115 | 0.108 | 0.139 | 0.127 |    0.117 | 0.113 |   0.101 |
| max roll/pitch [deg]                                            |   8.0 |   0.8 |   8.0 |   2.4 |      5.0 |   1.1 |     4.3 |
| max distance from spawn [m]                                     |  2.50 |  0.99 |  1.06 |  1.20 |     2.50 |  1.06 |    1.27 |
| **Solver**                                                      |       |       |       |       |          |       |         |
| arm QP failures (n)                                             |     1 |     0 |     0 |     0 |        1 |     0 |       0 |

### 1b. By payload (all trajectory types pooled)

| metric                                                          | 1.5 kg |   all |
|-----------------------------------------------------------------|-------:|------:|
| **Outcome**                                                     |        |       |
| scenarios                                                       |    600 |   600 |
| failed: COM left footprint or tipped (n)                        |      3 |     3 |
| failure rate [%]                                                |    0.5 |   0.5 |
| mean time of failure [s]                                        |   4.03 |  4.03 |
| would have failed: tipped or COM left footprint at any time (n) |      3 |     3 |
| tipped at any time (n)                                          |      3 |     3 |
| COM left footprint at any time (n)                              |      3 |     3 |
| **Position error (TCP vs reference)**                           |        |       |
| RMS [cm]                                                        |   3.05 |  3.05 |
| mean [cm]                                                       |   2.03 |  2.03 |
| std [cm]                                                        |   2.27 |  2.27 |
| 95th percentile [cm]                                            |   6.62 |  6.62 |
| max [cm]                                                        |  26.58 | 26.58 |
| RMS, completed scenarios only [cm]                              |   3.02 |  3.02 |
| per-scenario RMS: mean [cm]                                     |   2.46 |  2.46 |
| per-scenario RMS: std [cm]                                      |   1.87 |  1.87 |
| per-scenario RMS: median [cm]                                   |   1.87 |  1.87 |
| per-scenario RMS: worst [cm]                                    |  12.44 | 12.44 |
| final error, last 0.5 s: mean [cm]                              |   2.10 |  2.10 |
| final error, last 0.5 s: worst [cm]                             |  18.23 | 18.23 |
| **Orientation error**                                           |        |       |
| RMS [deg]                                                       |   1.82 |  1.82 |
| mean [deg]                                                      |   1.30 |  1.30 |
| std [deg]                                                       |   1.28 |  1.28 |
| 95th percentile [deg]                                           |   3.62 |  3.62 |
| max [deg]                                                       |  14.88 | 14.88 |
| per-scenario RMS: mean [deg]                                    |   1.70 |  1.70 |
| per-scenario RMS: std [deg]                                     |   0.79 |  0.79 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |        |       |
| rho mean                                                        |  0.131 | 0.131 |
| rho std                                                         |  0.095 | 0.095 |
| rho max                                                         |  1.029 | 1.029 |
| margin to edge, mean [cm]                                       |  11.08 | 11.08 |
| margin to edge, std [cm]                                        |   3.78 |  3.78 |
| margin to edge, worst step [cm]                                 |  -0.29 | -0.29 |
| per-scenario closest margin: mean [cm]                          |   9.45 |  9.45 |
| per-scenario closest margin: 5th pct [cm]                       |   5.57 |  5.57 |
| **Base motion**                                                 |        |       |
| speed mean [m/s]                                                |  0.164 | 0.164 |
| speed max [m/s]                                                 |  0.875 | 0.875 |
| mean absolute yaw rate [rad/s]                                  |  0.115 | 0.115 |
| max roll/pitch [deg]                                            |    8.0 |   8.0 |
| max distance from spawn [m]                                     |   2.50 |  2.50 |
| **Solver**                                                      |        |       |
| arm QP failures (n)                                             |      1 |     1 |

## 2. Extra mass 1.5 kg

| metric                                                          |   ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |       |       |       |       |          |       |         |
| scenarios                                                       |   600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| failure rate [%]                                                |   0.5 |   0.0 |   1.0 |   0.0 |      1.0 |   0.0 |     1.0 |
| mean time of failure [s]                                        |  4.03 |     - |  2.37 |     - |     3.27 |     - |    6.47 |
| would have failed: tipped or COM left footprint at any time (n) |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| tipped at any time (n)                                          |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| COM left footprint at any time (n)                              |     3 |     0 |     1 |     0 |        1 |     0 |       1 |
| **Position error (TCP vs reference)**                           |       |       |       |       |          |       |         |
| RMS [cm]                                                        |  3.05 |  2.85 |  2.99 |  3.88 |     3.81 |  2.34 |    2.02 |
| mean [cm]                                                       |  2.03 |  1.98 |  2.14 |  2.69 |     2.38 |  1.73 |    1.47 |
| std [cm]                                                        |  2.27 |  2.05 |  2.08 |  2.80 |     2.97 |  1.57 |    1.38 |
| 95th percentile [cm]                                            |  6.62 |  6.35 |  5.89 |  8.87 |     8.74 |  4.58 |    3.87 |
| max [cm]                                                        | 26.58 | 15.61 | 18.34 | 18.54 |    26.58 | 11.69 |   13.29 |
| RMS, completed scenarios only [cm]                              |  3.02 |  2.85 |  2.98 |  3.88 |     3.74 |  2.34 |    1.95 |
| per-scenario RMS: mean [cm]                                     |  2.46 |  2.26 |  2.52 |  3.17 |     3.08 |  1.99 |    1.77 |
| per-scenario RMS: std [cm]                                      |  1.87 |  1.74 |  1.60 |  2.24 |     2.48 |  1.18 |    1.08 |
| per-scenario RMS: median [cm]                                   |  1.87 |  1.66 |  2.01 |  2.32 |     2.04 |  1.75 |    1.55 |
| per-scenario RMS: worst [cm]                                    | 12.44 |  9.83 |  9.58 | 10.96 |    12.44 |  6.14 |    6.60 |
| final error, last 0.5 s: mean [cm]                              |  2.10 |  2.27 |  2.21 |  3.03 |     2.15 |  1.70 |    1.23 |
| final error, last 0.5 s: worst [cm]                             | 18.23 | 15.06 | 15.17 | 14.04 |    18.23 | 10.97 |   10.72 |
| **Orientation error**                                           |       |       |       |       |          |       |         |
| RMS [deg]                                                       |  1.82 |  1.62 |  2.25 |  2.09 |     2.02 |  1.63 |    1.38 |
| mean [deg]                                                      |  1.30 |  1.30 |  1.69 |  1.62 |     1.34 |  1.24 |    0.93 |
| std [deg]                                                       |  1.28 |  0.96 |  1.49 |  1.32 |     1.51 |  1.05 |    1.03 |
| 95th percentile [deg]                                           |  3.62 |  3.19 |  4.33 |  4.15 |     4.25 |  3.21 |    2.92 |
| max [deg]                                                       | 14.88 |  4.59 | 14.45 | 11.91 |    14.88 | 11.23 |   11.85 |
| per-scenario RMS: mean [deg]                                    |  1.70 |  1.54 |  2.06 |  1.94 |     1.83 |  1.55 |    1.27 |
| per-scenario RMS: std [deg]                                     |  0.79 |  0.50 |  0.91 |  0.77 |     0.98 |  0.55 |    0.59 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |       |       |       |       |          |       |         |
| rho mean                                                        | 0.131 | 0.106 | 0.142 | 0.152 |    0.160 | 0.106 |   0.111 |
| rho std                                                         | 0.095 | 0.072 | 0.098 | 0.099 |    0.111 | 0.071 |   0.083 |
| rho max                                                         | 1.029 | 0.379 | 1.029 | 0.688 |    1.021 | 0.503 |   1.025 |
| margin to edge, mean [cm]                                       | 11.08 | 11.30 | 10.99 | 10.94 |    10.74 | 11.41 |   11.25 |
| margin to edge, std [cm]                                        |  3.78 |  3.65 |  3.85 |  3.87 |     3.82 |  3.85 |    3.68 |
| margin to edge, worst step [cm]                                 | -0.29 |  5.94 | -0.29 |  3.38 |    -0.20 |  4.72 |   -0.24 |
| per-scenario closest margin: mean [cm]                          |  9.45 |  9.88 |  9.19 |  9.19 |     8.98 |  9.90 |    9.56 |
| per-scenario closest margin: 5th pct [cm]                       |  5.57 |  6.25 |  5.34 |  5.47 |     4.90 |  6.25 |    5.59 |
| **Base motion**                                                 |       |       |       |       |          |       |         |
| speed mean [m/s]                                                | 0.164 | 0.102 | 0.141 | 0.182 |    0.200 | 0.134 |   0.179 |
| speed max [m/s]                                                 | 0.875 | 0.532 | 0.608 | 0.875 |    0.721 | 0.425 |   0.612 |
| mean absolute yaw rate [rad/s]                                  | 0.115 | 0.108 | 0.139 | 0.127 |    0.117 | 0.113 |   0.101 |
| max roll/pitch [deg]                                            |   8.0 |   0.8 |   8.0 |   2.4 |      5.0 |   1.1 |     4.3 |
| max distance from spawn [m]                                     |  2.50 |  0.99 |  1.06 |  1.20 |     2.50 |  1.06 |    1.27 |
| **Solver**                                                      |       |       |       |       |          |       |         |
| arm QP failures (n)                                             |     1 |     0 |     0 |     0 |        1 |     0 |       0 |

## Definitions

- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, up to and including the step where it failed
- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and `COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure (the simulation keeps running), so `tipped` can exceed what ended a scenario
- **position error**: distance between the TCP and the reference at the same time (env-local world frame); RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the last 0.5 s of each scenario that did not fail
- **orientation error**: angle of the rotation between the TCP and reference orientation
- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)
- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, use rho to compare platforms
- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point
- **arm QP failures**: solver failures of the arm controller during the runs
