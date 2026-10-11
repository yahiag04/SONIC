# SONIC results: split `eval`, run `main`

Samples: 60 (dog_env: 20, intermittent: 20, short_event: 20).
Values: mean [95% bootstrap CI]. SI-SDR/suppression in dB; latency in s; memory in MB.

## Reconstruction quality (overall)
| method | sdr_bg | sisdr_bg | sisdri | sdr_bg_out | sdr_bg_in | mse_out | target_suppression_db |
|---|---|---|---|---|---|---|---|
| baseline | 15.48 [13.53, 17.53] | 15.18 [13.11, 17.33] | 7.72 [5.79, 9.70] | 20.78 [17.39, 24.21] | 13.12 [11.68, 14.57] | 0.00 [0.00, 0.00] | 13.74 [12.43, 15.32] |
| sonic | 17.59 [15.57, 19.71] | 17.49 [15.48, 19.66] | 10.03 [8.35, 11.90] | 36.74 [33.30, 40.00] | 10.56 [9.01, 12.28] | 0.00 [0.00, 0.00] | 11.18 [9.52, 12.93] |
| oracle | 20.93 [19.18, 22.65] | 20.86 [19.07, 22.61] | 13.41 [12.08, 14.95] | 39.33 [36.50, 42.15] | 13.12 [11.68, 14.57] | 0.00 [0.00, 0.00] | 13.74 [12.43, 15.32] |

## Reconstruction quality by scenario
| scenario | method | sdr_bg | sdr_bg_out | sdr_bg_in | target_suppression_db |
|---|---|---|---|---|---|
| dog_env | baseline | 10.59 [8.62, 13.01] | 11.10 [8.27, 14.49] | 13.78 [11.85, 15.82] | 13.08 [11.39, 14.76] |
| dog_env | oracle | 19.63 [17.29, 22.04] | 33.89 [29.24, 38.46] | 13.78 [11.85, 15.82] | 13.08 [11.39, 14.76] |
| dog_env | sonic | 15.68 [13.16, 18.51] | 30.67 [25.01, 35.91] | 10.86 [8.60, 13.30] | 10.17 [7.79, 12.62] |
| intermittent | baseline | 18.01 [14.74, 21.77] | 29.41 [23.72, 35.40] | 13.82 [11.01, 16.81] | 14.18 [11.48, 17.12] |
| intermittent | oracle | 19.53 [16.37, 23.00] | 40.80 [35.68, 45.72] | 13.82 [11.01, 16.81] | 14.18 [11.48, 17.12] |
| intermittent | sonic | 14.76 [11.59, 18.72] | 42.84 [38.80, 47.01] | 9.01 [6.24, 12.76] | 9.36 [6.49, 13.12] |
| short_event | baseline | 17.84 [14.73, 21.29] | 21.82 [16.64, 26.92] | 11.77 [9.38, 14.04] | 13.96 [10.45, 17.18] |
| short_event | oracle | 23.64 [20.48, 26.55] | 43.30 [38.96, 47.43] | 11.77 [9.38, 14.04] | 13.96 [10.45, 17.18] |
| short_event | sonic | 22.35 [18.76, 25.79] | 36.70 [30.27, 43.34] | 11.81 [9.45, 14.07] | 14.01 [10.49, 17.27] |

## Localization (SONIC vs oracle)
| scenario | method | loc_precision | loc_recall | loc_f1 | loc_iou | onset_err_s | offset_err_s | event_recall |
|---|---|---|---|---|---|---|---|---|
| dog_env | oracle | 0.92 [0.90, 0.93] | 1.00 [1.00, 1.00] | 0.96 [0.95, 0.96] | 0.92 [0.90, 0.93] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 1.00 [1.00, 1.00] |
| dog_env | sonic | 0.73 [0.63, 0.82] | 0.85 [0.76, 0.92] | 0.77 [0.68, 0.84] | 0.65 [0.55, 0.73] | 0.38 [0.28, 0.47] | 0.59 [0.42, 0.80] | 1.00 [1.00, 1.00] |
| intermittent | oracle | 0.86 [0.85, 0.87] | 1.00 [1.00, 1.00] | 0.93 [0.92, 0.93] | 0.86 [0.85, 0.87] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 1.00 [1.00, 1.00] |
| intermittent | sonic | 0.70 [0.65, 0.76] | 0.67 [0.59, 0.77] | 0.67 [0.61, 0.73] | 0.52 [0.45, 0.59] | 0.39 [0.31, 0.49] | 0.32 [0.26, 0.38] | 0.85 [0.75, 0.95] |
| short_event | oracle | 0.82 [0.77, 0.86] | 1.00 [1.00, 1.00] | 0.90 [0.87, 0.92] | 0.82 [0.77, 0.86] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 1.00 [1.00, 1.00] |
| short_event | sonic | 0.48 [0.38, 0.57] | 0.96 [0.92, 0.99] | 0.61 [0.52, 0.68] | 0.46 [0.37, 0.54] | 0.50 [0.42, 0.58] | 0.51 [0.36, 0.68] | 1.00 [1.00, 1.00] |

## Cost
| method | latency_s | latency_localization_s | peak_gpu_mb | masked_fraction | edge_click_db |
|---|---|---|---|---|---|
| baseline | 0.37 [0.37, 0.38] | 0.00 [0.00, 0.00] | 2270.28 [2270.28, 2270.28] | 1.00 [1.00, 1.00] | n/a |
| sonic | 1.21 [1.20, 1.22] | 0.84 [0.83, 0.85] | 2346.25 [2346.25, 2346.25] | 0.34 [0.32, 0.37] | -1.94 [-3.83, -0.35] |
| oracle | 0.37 [0.37, 0.38] | 0.00 [0.00, 0.00] | 2270.28 [2270.28, 2270.28] | 0.33 [0.29, 0.37] | -2.55 [-3.98, -1.19] |

## Paired Wilcoxon tests (difference = b - a)
| metric | a | b | n | mean_diff | median_diff | p_value | scenario |
|---|---|---|---|---|---|---|---|
| sdr_bg | baseline | sonic | 60 | 2.11 | 1.77 | 0.00721 | all |
| sdr_bg | baseline | oracle | 60 | 5.45 | 3.64 | 1.63e-11 | all |
| sdr_bg | sonic | oracle | 60 | 3.34 | 1.35 | 1.25e-09 | all |
| sisdr_bg | baseline | sonic | 60 | 2.31 | 2.14 | 0.00481 | all |
| sisdr_bg | baseline | oracle | 60 | 5.68 | 3.92 | 1.71e-11 | all |
| sisdr_bg | sonic | oracle | 60 | 3.37 | 1.18 | 1.31e-09 | all |
| sisdri | baseline | sonic | 60 | 2.31 | 2.14 | 0.00481 | all |
| sisdri | baseline | oracle | 60 | 5.68 | 3.92 | 1.71e-11 | all |
| sisdri | sonic | oracle | 60 | 3.37 | 1.18 | 1.31e-09 | all |
| sdr_bg_out | baseline | sonic | 60 | 16 | 11.9 | 1.63e-11 | all |
| sdr_bg_out | baseline | oracle | 60 | 18.6 | 16.8 | 1.63e-11 | all |
| sdr_bg_out | sonic | oracle | 60 | 2.59 | 2.04 | 0.145 | all |
| sdr_bg_in | baseline | sonic | 60 | -2.56 | 0 | 1.45e-05 | all |
| sdr_bg_in | baseline | oracle | 60 | 0 | 0 | 1 | all |
| sdr_bg_in | sonic | oracle | 60 | 2.56 | 0 | 1.45e-05 | all |
| mse_out | baseline | sonic | 60 | -0.00041 | -4.57e-05 | 1.63e-11 | all |
| mse_out | baseline | oracle | 60 | -0.000449 | -5.7e-05 | 1.63e-11 | all |
| mse_out | sonic | oracle | 60 | -3.9e-05 | -2.66e-08 | 0.0414 | all |
| target_suppression_db | baseline | sonic | 60 | -2.56 | 0 | 1.45e-05 | all |
| target_suppression_db | baseline | oracle | 60 | 0 | 0 | 1 | all |
| target_suppression_db | sonic | oracle | 60 | 2.56 | 0 | 1.45e-05 | all |
| edge_click_db | baseline | sonic | 60 | nan | nan | nan | all |
| edge_click_db | baseline | oracle | 60 | nan | nan | nan | all |
| edge_click_db | sonic | oracle | 60 | -0.61 | -0.328 | 0.2 | all |
| sdr_bg | baseline | sonic | 20 | 5.09 | 4.3 | 0.00422 | dog_env |
| sdr_bg_out | baseline | sonic | 20 | 19.6 | 18.1 | 1.91e-06 | dog_env |
| target_suppression_db | baseline | sonic | 20 | -2.92 | -0.511 | 0.00474 | dog_env |
| sdr_bg | baseline | sonic | 20 | -3.26 | -1.69 | 0.0328 | intermittent |
| sdr_bg_out | baseline | sonic | 20 | 13.4 | 8.48 | 1.91e-06 | intermittent |
| target_suppression_db | baseline | sonic | 20 | -4.82 | -2.09 | 0.000536 | intermittent |
| sdr_bg | baseline | sonic | 20 | 4.51 | 3.22 | 1.91e-06 | short_event |
| sdr_bg_out | baseline | sonic | 20 | 14.9 | 9.57 | 1.91e-06 | short_event |
| target_suppression_db | baseline | sonic | 20 | 0.0448 | 0 | 0.753 | short_event |

## Auxiliary CLAP similarity (not a fidelity measure)
| method | clap_sim_output | clap_target_sim_drop |
|---|---|---|
| baseline | -0.05 [-0.07, -0.03] | 0.21 [0.19, 0.24] |
| sonic | -0.01 [-0.03, 0.01] | 0.17 [0.15, 0.19] |
| oracle | -0.05 [-0.07, -0.03] | 0.21 [0.19, 0.24] |
