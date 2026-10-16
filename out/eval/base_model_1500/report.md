# Evaluation report: base_model_1500

- checkpoint: `/home/u0110021/hybrid-rl-mpc-control/logs/base_/model_1500.pt`; arm controller: `impedance`, benchmark `a7355009e5dd` (seed 2026, poly paths of Bezier order 1), commit `81a69b9+dirty`, 2026-10-16T10:46:48
- extra mass at the gripper: each scenario's own payload, drawn like the training domain randomization (checkpoint trained with 0-3 kg)
- trajectory types: fixed, sine, poly, poly_far, rigid, ellipse; 600 scenarios
- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); this is also the training termination criterion
- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end

## 1. All scenarios

| metric                                                          |   ALL | fixed |  sine |  poly | poly_far | rigid | ellipse |
|-----------------------------------------------------------------|------:|------:|------:|------:|---------:|------:|--------:|
| **Outcome**                                                     |       |       |       |       |          |       |         |
| scenarios                                                       |   600 |   100 |   100 |   100 |      100 |   100 |     100 |
| failed: COM left footprint or tipped (n)                        |     2 |     0 |     0 |     1 |        0 |     0 |       1 |
| failure rate [%]                                                |   0.3 |   0.0 |   0.0 |   1.0 |      0.0 |   0.0 |     1.0 |
| mean time of failure [s]                                        |  6.58 |     - |     - |  4.53 |        - |     - |    8.63 |
| would have failed: tipped or COM left footprint at any time (n) |     2 |     0 |     0 |     1 |        0 |     0 |       1 |
| tipped at any time (n)                                          |     2 |     0 |     0 |     1 |        0 |     0 |       1 |
| COM left footprint at any time (n)                              |     2 |     0 |     0 |     1 |        0 |     0 |       1 |
| **Position error (TCP vs reference)**                           |       |       |       |       |          |       |         |
| RMS [cm]                                                        |  3.31 |  3.08 |  3.09 |  4.22 |     3.98 |  2.51 |    2.57 |
| mean [cm]                                                       |  2.18 |  2.09 |  2.21 |  2.90 |     2.49 |  1.81 |    1.73 |
| std [cm]                                                        |  2.49 |  2.26 |  2.16 |  3.06 |     3.10 |  1.74 |    1.89 |
| 95th percentile [cm]                                            |  7.19 |  6.90 |  6.33 |  9.24 |     9.39 |  5.06 |    4.84 |
| max [cm]                                                        | 24.47 | 16.91 | 15.81 | 21.83 |    23.04 | 16.20 |   24.47 |
| RMS, completed scenarios only [cm]                              |  3.26 |  3.08 |  3.09 |  4.08 |     3.98 |  2.51 |    2.40 |
| per-scenario RMS: mean [cm]                                     |  2.63 |  2.38 |  2.62 |  3.41 |     3.18 |  2.06 |    2.11 |
| per-scenario RMS: std [cm]                                      |  2.02 |  1.95 |  1.64 |  2.50 |     2.47 |  1.28 |    1.57 |
| per-scenario RMS: median [cm]                                   |  1.93 |  1.63 |  2.08 |  2.47 |     2.29 |  1.82 |    1.70 |
| per-scenario RMS: worst [cm]                                    | 11.95 | 10.66 |  8.63 | 11.95 |    10.18 |  8.49 |   10.13 |
| final error, last 0.5 s: mean [cm]                              |  2.37 |  2.47 |  2.34 |  3.51 |     2.33 |  1.82 |    1.73 |
| final error, last 0.5 s: worst [cm]                             | 20.26 | 16.29 | 13.55 | 16.10 |    20.26 | 15.69 |   12.41 |
| **Orientation error**                                           |       |       |       |       |          |       |         |
| RMS [deg]                                                       |  1.90 |  1.66 |  2.36 |  2.18 |     2.04 |  1.61 |    1.58 |
| mean [deg]                                                      |  1.35 |  1.33 |  1.75 |  1.69 |     1.37 |  1.24 |    1.02 |
| std [deg]                                                       |  1.34 |  0.99 |  1.59 |  1.38 |     1.50 |  1.03 |    1.20 |
| 95th percentile [deg]                                           |  3.76 |  3.22 |  4.72 |  4.29 |     4.30 |  3.14 |    3.09 |
| max [deg]                                                       | 16.17 |  5.91 | 14.75 | 13.37 |    14.00 | 11.20 |   16.17 |
| per-scenario RMS: mean [deg]                                    |  1.75 |  1.58 |  2.15 |  2.02 |     1.84 |  1.54 |    1.40 |
| per-scenario RMS: std [deg]                                     |  0.83 |  0.53 |  0.97 |  0.83 |     0.97 |  0.53 |    0.78 |
| **COM stability (rho: 0 = centred, 1 = at the footprint edge)** |       |       |       |       |          |       |         |
| rho mean                                                        | 0.144 | 0.119 | 0.151 | 0.165 |    0.171 | 0.114 |   0.130 |
| rho std                                                         | 0.107 | 0.081 | 0.105 | 0.107 |    0.120 | 0.081 |   0.111 |
| rho max                                                         | 1.037 | 0.504 | 0.676 | 1.037 |    0.926 | 0.560 |   1.025 |
| margin to edge, mean [cm]                                       | 10.94 | 11.18 | 10.89 | 10.78 |    10.60 | 11.30 |   11.06 |
| margin to edge, std [cm]                                        |  3.83 |  3.66 |  3.91 |  3.81 |     3.85 |  3.85 |    3.83 |
| margin to edge, worst step [cm]                                 | -0.36 |  4.75 |  3.36 | -0.36 |     0.70 |  4.18 |   -0.26 |
| per-scenario closest margin: mean [cm]                          |  9.15 |  9.61 |  8.92 |  8.89 |     8.73 |  9.59 |    9.19 |
| per-scenario closest margin: 5th pct [cm]                       |  4.89 |  5.59 |  4.72 |  4.67 |     4.89 |  5.24 |    3.50 |
| **Base motion**                                                 |       |       |       |       |          |       |         |
| speed mean [m/s]                                                | 0.163 | 0.103 | 0.146 | 0.181 |    0.199 | 0.134 |   0.173 |
| speed max [m/s]                                                 | 0.815 | 0.531 | 0.606 | 0.815 |    0.697 | 0.466 |   0.649 |
| mean absolute yaw rate [rad/s]                                  | 0.120 | 0.114 | 0.147 | 0.137 |    0.121 | 0.116 |   0.104 |
| max roll/pitch [deg]                                            |  17.4 |   1.1 |   1.4 |  16.4 |      3.0 |   1.1 |    17.4 |
| max distance from spawn [m]                                     |  2.32 |  1.05 |  1.06 |  1.18 |     2.32 |  1.00 |    1.27 |
| **Solver**                                                      |       |       |       |       |          |       |         |
| arm QP failures (n)                                             |     0 |     0 |     0 |     0 |        0 |     0 |       0 |

## Definitions

- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, up to and including the step where it failed
- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and `COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure (the simulation keeps running), so `tipped` can exceed what ended a scenario
- **position error**: distance between the TCP and the reference at the same time (env-local world frame); RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the last 0.5 s of each scenario that did not fail
- **orientation error**: angle of the rotation between the TCP and reference orientation
- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)
- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, use rho to compare platforms
- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point
- **arm QP failures**: solver failures of the arm controller during the runs
