# Experiment C: generative removal (split `eval`)

Clips: 60. Model: `cvssp/audioldm2` (pretrained, unchanged), 100 DDIM steps, guidance 3.5, negative prompt = target, prompt = "natural background ambience, high quality".
`gen_*@s`: regenerate a fraction s of the diffusion schedule; masks are the same as in the subtractive experiment.
LSD = log-spectral distance to the reference background (dB, lower is better, phase-insensitive).
`clap_event_target_drop`: decrease of CLAP similarity to the target text inside true event regions (higher = more removed).

## Front-end check (vocoder / VAE round trip, LSD dB)

| lsd_vocoder_db | lsd_vae_vocoder_db | sdr_vocoder_db |
|---|---|---|
| 7.06 | 7.11 | -2.63 |

## Overall

| method | sdr_bg | sdr_bg_out | sdr_bg_in | target_suppression_db | lsd_out | lsd_in | clap_event_target_sim | clap_event_target_drop | latency_generation_s | masked_fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 15.48 [13.53, 17.53] | 20.78 [17.39, 24.21] | 13.12 [11.68, 14.57] | 13.74 [12.43, 15.32] | 2.58 [2.10, 3.04] | 5.02 [4.50, 5.60] | -0.05 [-0.07, -0.03] | 0.26 [0.23, 0.29] | n/a | 1.00 [1.00, 1.00] |
| sonic | 17.59 [15.57, 19.71] | 36.74 [33.30, 40.00] | 10.56 [9.01, 12.28] | 11.18 [9.52, 12.93] | 0.35 [0.27, 0.43] | 5.07 [4.51, 5.65] | 0.01 [-0.02, 0.03] | 0.21 [0.18, 0.24] | n/a | 0.34 [0.32, 0.37] |
| oracle | 20.93 [19.18, 22.65] | 39.33 [36.50, 42.15] | 13.12 [11.68, 14.57] | 13.74 [12.43, 15.32] | 0.18 [0.14, 0.23] | 5.02 [4.50, 5.60] | -0.05 [-0.07, -0.03] | 0.26 [0.23, 0.29] | n/a | 0.33 [0.29, 0.37] |
| gen_global@0.6 | -2.49 [-2.77, -2.22] | -2.33 [-2.62, -2.07] | -3.63 [-4.84, -2.86] | -3.01 [-3.88, -2.11] | 12.82 [11.68, 14.02] | 13.26 [12.25, 14.30] | -0.07 [-0.08, -0.05] | 0.28 [0.24, 0.31] | 6.47 [6.38, 6.57] | 1.00 [1.00, 1.00] |
| gen_sonic@0.6 | 2.97 [2.36, 3.66] | 7.80 [6.52, 9.24] | -2.85 [-4.03, -1.99] | -2.23 [-3.07, -1.37] | 1.88 [1.60, 2.20] | 11.77 [10.72, 12.85] | -0.01 [-0.04, 0.01] | 0.23 [0.19, 0.26] | 6.41 [6.33, 6.49] | 0.34 [0.32, 0.37] |
| gen_oracle@0.6 | 3.44 [2.59, 4.33] | 10.36 [9.17, 11.88] | -3.63 [-4.73, -2.93] | -3.01 [-3.87, -2.07] | 1.07 [0.86, 1.29] | 13.17 [12.13, 14.22] | -0.05 [-0.07, -0.03] | 0.27 [0.23, 0.30] | 6.38 [6.29, 6.47] | 0.33 [0.29, 0.37] |
| gen_global@1 | -7.96 [-8.47, -7.45] | -7.92 [-8.47, -7.39] | -9.17 [-10.57, -8.08] | -8.55 [-9.25, -7.79] | 16.24 [15.20, 17.39] | 15.64 [14.63, 16.75] | -0.11 [-0.13, -0.09] | 0.32 [0.29, 0.36] | 10.50 [10.42, 10.58] | 1.00 [1.00, 1.00] |
| gen_sonic@1 | -1.76 [-2.74, -0.59] | 2.39 [0.98, 3.90] | -7.41 [-8.66, -6.38] | -6.79 [-7.80, -5.66] | 2.28 [1.95, 2.64] | 12.99 [11.95, 14.10] | -0.06 [-0.09, -0.02] | 0.27 [0.23, 0.31] | 10.52 [10.46, 10.59] | 0.34 [0.32, 0.37] |
| gen_oracle@1 | -0.95 [-2.26, 0.49] | 5.47 [4.26, 6.76] | -7.93 [-9.14, -6.87] | -7.31 [-8.38, -6.03] | 1.30 [1.05, 1.59] | 14.59 [13.49, 15.79] | -0.09 [-0.11, -0.07] | 0.31 [0.27, 0.34] | 10.51 [10.44, 10.58] | 0.33 [0.29, 0.37] |

## By scenario

| scenario | method | sdr_bg_out | lsd_out | lsd_in | clap_event_target_drop |
|---|---|---|---|---|---|
| dog_env | baseline | 11.10 [8.27, 14.49] | 3.74 [2.93, 4.53] | 5.14 [4.18, 6.53] | 0.24 [0.20, 0.28] |
| dog_env | gen_global@0.6 | -1.59 [-1.90, -1.30] | 13.45 [12.04, 15.10] | 13.99 [12.53, 15.60] | 0.25 [0.19, 0.29] |
| dog_env | gen_global@1 | -6.10 [-6.95, -5.27] | 16.14 [14.76, 17.80] | 16.37 [15.01, 17.92] | 0.27 [0.22, 0.31] |
| dog_env | gen_oracle@0.6 | 11.57 [10.99, 12.19] | 0.72 [0.65, 0.80] | 13.87 [12.40, 15.47] | 0.21 [0.15, 0.25] |
| dog_env | gen_oracle@1 | 7.46 [6.33, 8.58] | 0.84 [0.77, 0.93] | 16.12 [14.72, 17.65] | 0.26 [0.20, 0.30] |
| dog_env | gen_sonic@0.6 | 8.61 [7.01, 10.41] | 1.64 [1.17, 2.21] | 12.55 [10.89, 14.29] | 0.16 [0.11, 0.21] |
| dog_env | gen_sonic@1 | 4.95 [2.71, 7.36] | 1.92 [1.39, 2.55] | 14.51 [12.90, 16.19] | 0.19 [0.13, 0.25] |
| dog_env | oracle | 33.89 [29.24, 38.46] | 0.15 [0.11, 0.19] | 5.14 [4.18, 6.53] | 0.24 [0.20, 0.28] |
| dog_env | sonic | 30.67 [25.01, 35.91] | 0.41 [0.23, 0.63] | 5.05 [4.06, 6.48] | 0.15 [0.11, 0.20] |
| intermittent | baseline | 29.41 [23.72, 35.40] | 1.74 [1.05, 2.56] | 4.55 [3.79, 5.26] | 0.32 [0.27, 0.37] |
| intermittent | gen_global@0.6 | -2.66 [-3.13, -2.16] | 11.70 [9.88, 13.76] | 12.59 [10.88, 14.48] | 0.34 [0.29, 0.39] |
| intermittent | gen_global@1 | -8.52 [-9.28, -7.81] | 15.61 [13.50, 17.88] | 15.27 [13.23, 17.44] | 0.38 [0.33, 0.44] |
| intermittent | gen_oracle@0.6 | 6.27 [4.92, 7.97] | 1.93 [1.54, 2.32] | 12.52 [10.82, 14.39] | 0.35 [0.30, 0.40] |
| intermittent | gen_oracle@1 | 1.03 [-0.41, 2.63] | 2.45 [1.95, 3.01] | 14.22 [11.88, 16.69] | 0.36 [0.31, 0.41] |
| intermittent | gen_sonic@0.6 | 7.34 [5.03, 9.77] | 1.93 [1.58, 2.32] | 10.02 [8.66, 11.66] | 0.26 [0.20, 0.32] |
| intermittent | gen_sonic@1 | 0.84 [-1.34, 3.35] | 2.54 [1.99, 3.16] | 11.05 [9.56, 13.14] | 0.31 [0.24, 0.37] |
| intermittent | oracle | 40.80 [35.68, 45.72] | 0.29 [0.19, 0.41] | 4.55 [3.79, 5.26] | 0.32 [0.27, 0.37] |
| intermittent | sonic | 42.84 [38.80, 47.01] | 0.27 [0.18, 0.39] | 4.78 [3.91, 5.61] | 0.25 [0.19, 0.30] |
| short_event | baseline | 21.82 [16.64, 26.92] | 2.25 [1.64, 2.88] | 5.38 [4.37, 6.45] | 0.22 [0.17, 0.28] |
| short_event | gen_global@0.6 | -2.75 [-3.25, -2.31] | 13.31 [10.91, 15.71] | 13.20 [11.22, 15.15] | 0.26 [0.18, 0.32] |
| short_event | gen_global@1 | -9.14 [-10.00, -8.20] | 16.98 [14.84, 19.02] | 15.29 [13.66, 16.92] | 0.32 [0.25, 0.38] |
| short_event | gen_oracle@0.6 | 13.23 [10.69, 16.69] | 0.55 [0.48, 0.62] | 13.12 [11.13, 15.11] | 0.25 [0.19, 0.31] |
| short_event | gen_oracle@1 | 7.92 [5.82, 10.10] | 0.61 [0.55, 0.68] | 13.43 [11.76, 15.12] | 0.30 [0.23, 0.37] |
| short_event | gen_sonic@0.6 | 7.47 [4.97, 10.60] | 2.08 [1.47, 2.71] | 12.74 [10.89, 14.60] | 0.26 [0.18, 0.32] |
| short_event | gen_sonic@1 | 1.38 [-0.96, 3.83] | 2.39 [1.78, 3.02] | 13.40 [11.85, 15.00] | 0.31 [0.25, 0.38] |
| short_event | oracle | 43.30 [38.96, 47.43] | 0.11 [0.08, 0.15] | 5.38 [4.37, 6.45] | 0.22 [0.17, 0.28] |
| short_event | sonic | 36.70 [30.27, 43.34] | 0.35 [0.25, 0.46] | 5.37 [4.36, 6.45] | 0.22 [0.16, 0.28] |

## Paired Wilcoxon tests (difference = b - a)

| metric | a | b | n | mean_diff | median_diff | p_value |
|---|---|---|---|---|---|---|
| sdr_bg | gen_global@0.6 | gen_sonic@0.6 | 60 | 5.46 | 4.53 | 1.63e-11 |
| sdr_bg_out | gen_global@0.6 | gen_sonic@0.6 | 60 | 10.1 | 9.27 | 1.63e-11 |
| lsd_out | gen_global@0.6 | gen_sonic@0.6 | 60 | -10.9 | -9.92 | 1.63e-11 |
| lsd_in | gen_global@0.6 | gen_sonic@0.6 | 60 | -1.49 | -0.402 | 5.19e-10 |
| target_suppression_db | gen_global@0.6 | gen_sonic@0.6 | 60 | 0.783 | 0.474 | 0.000184 |
| clap_event_target_drop | gen_global@0.6 | gen_sonic@0.6 | 60 | -0.0538 | -0.0335 | 0.000233 |
| sdr_bg | gen_global@0.6 | gen_oracle@0.6 | 60 | 5.93 | 5.25 | 1.63e-11 |
| sdr_bg_out | gen_global@0.6 | gen_oracle@0.6 | 60 | 12.7 | 12.5 | 1.63e-11 |
| lsd_out | gen_global@0.6 | gen_oracle@0.6 | 60 | -11.8 | -10.5 | 1.63e-11 |
| lsd_in | gen_global@0.6 | gen_oracle@0.6 | 60 | -0.0884 | -0.0686 | 1.71e-05 |
| target_suppression_db | gen_global@0.6 | gen_oracle@0.6 | 60 | 0.00411 | -0.122 | 0.224 |
| clap_event_target_drop | gen_global@0.6 | gen_oracle@0.6 | 60 | -0.012 | -0.0157 | 0.135 |
| sdr_bg | gen_sonic@0.6 | gen_oracle@0.6 | 60 | 0.47 | 0.412 | 0.0334 |
| sdr_bg_out | gen_sonic@0.6 | gen_oracle@0.6 | 60 | 2.55 | 3.3 | 4.26e-05 |
| lsd_out | gen_sonic@0.6 | gen_oracle@0.6 | 60 | -0.818 | -0.56 | 4.51e-06 |
| lsd_in | gen_sonic@0.6 | gen_oracle@0.6 | 60 | 1.41 | 0.202 | 1.87e-07 |
| target_suppression_db | gen_sonic@0.6 | gen_oracle@0.6 | 60 | -0.779 | -0.169 | 2.74e-06 |
| clap_event_target_drop | gen_sonic@0.6 | gen_oracle@0.6 | 60 | 0.0418 | 0.00992 | 0.00859 |
| sdr_bg | sonic | gen_sonic@0.6 | 60 | -14.6 | -13.7 | 1.63e-11 |
| sdr_bg_out | sonic | gen_sonic@0.6 | 60 | -28.9 | -27.5 | 1.63e-11 |
| lsd_out | sonic | gen_sonic@0.6 | 60 | 1.54 | 1.2 | 1.71e-11 |
| lsd_in | sonic | gen_sonic@0.6 | 60 | 6.7 | 6.33 | 5.41e-11 |
| target_suppression_db | sonic | gen_sonic@0.6 | 60 | -13.4 | -12.5 | 1.63e-11 |
| clap_event_target_drop | sonic | gen_sonic@0.6 | 60 | 0.0189 | 0.00398 | 0.257 |
| sdr_bg | oracle | gen_oracle@0.6 | 60 | -17.5 | -17 | 1.63e-11 |
| sdr_bg_out | oracle | gen_oracle@0.6 | 60 | -29 | -28.5 | 1.63e-11 |
| lsd_out | oracle | gen_oracle@0.6 | 60 | 0.881 | 0.599 | 1.63e-11 |
| lsd_in | oracle | gen_oracle@0.6 | 60 | 8.15 | 7.26 | 3.82e-11 |
| target_suppression_db | oracle | gen_oracle@0.6 | 60 | -16.7 | -16.3 | 1.63e-11 |
| clap_event_target_drop | oracle | gen_oracle@0.6 | 60 | 0.00473 | 0.00786 | 0.802 |
| sdr_bg | gen_global@1 | gen_sonic@1 | 60 | 6.2 | 5.41 | 1.63e-11 |
| sdr_bg_out | gen_global@1 | gen_sonic@1 | 60 | 10.3 | 8.91 | 1.63e-11 |
| lsd_out | gen_global@1 | gen_sonic@1 | 60 | -14 | -14.1 | 1.63e-11 |
| lsd_in | gen_global@1 | gen_sonic@1 | 60 | -2.65 | -1.86 | 1.13e-10 |
| target_suppression_db | gen_global@1 | gen_sonic@1 | 60 | 1.76 | 1.66 | 0.000102 |
| clap_event_target_drop | gen_global@1 | gen_sonic@1 | 60 | -0.054 | -0.0269 | 0.00294 |
| sdr_bg | gen_global@1 | gen_oracle@1 | 60 | 7.01 | 5.34 | 1.63e-11 |
| sdr_bg_out | gen_global@1 | gen_oracle@1 | 60 | 13.4 | 13 | 1.63e-11 |
| lsd_out | gen_global@1 | gen_oracle@1 | 60 | -14.9 | -14.3 | 1.63e-11 |
| lsd_in | gen_global@1 | gen_oracle@1 | 60 | -1.05 | -0.768 | 7.26e-07 |
| target_suppression_db | gen_global@1 | gen_oracle@1 | 60 | 1.24 | 0.199 | 0.102 |
| clap_event_target_drop | gen_global@1 | gen_oracle@1 | 60 | -0.0183 | -0.0196 | 0.0437 |
| sdr_bg | gen_sonic@1 | gen_oracle@1 | 60 | 0.809 | 0.533 | 0.145 |
| sdr_bg_out | gen_sonic@1 | gen_oracle@1 | 60 | 3.08 | 3.75 | 1.66e-05 |
| lsd_out | gen_sonic@1 | gen_oracle@1 | 60 | -0.983 | -0.831 | 2.13e-06 |
| lsd_in | gen_sonic@1 | gen_oracle@1 | 60 | 1.6 | 0.312 | 0.0013 |
| target_suppression_db | gen_sonic@1 | gen_oracle@1 | 60 | -0.519 | -0.199 | 0.0346 |
| clap_event_target_drop | gen_sonic@1 | gen_oracle@1 | 60 | 0.0356 | 0.00348 | 0.0725 |
| sdr_bg | sonic | gen_sonic@1 | 60 | -19.4 | -18.4 | 1.63e-11 |
| sdr_bg_out | sonic | gen_sonic@1 | 60 | -34.4 | -34.4 | 1.63e-11 |
| lsd_out | sonic | gen_sonic@1 | 60 | 1.94 | 1.58 | 1.63e-11 |
| lsd_in | sonic | gen_sonic@1 | 60 | 7.92 | 8.25 | 2.21e-11 |
| target_suppression_db | sonic | gen_sonic@1 | 60 | -18 | -16.8 | 1.63e-11 |
| clap_event_target_drop | sonic | gen_sonic@1 | 60 | 0.063 | 0.056 | 3.4e-05 |
| sdr_bg | oracle | gen_oracle@1 | 60 | -21.9 | -21.1 | 1.63e-11 |
| sdr_bg_out | oracle | gen_oracle@1 | 60 | -33.9 | -33.4 | 1.63e-11 |
| lsd_out | oracle | gen_oracle@1 | 60 | 1.12 | 0.723 | 1.63e-11 |
| lsd_in | oracle | gen_oracle@1 | 60 | 9.57 | 9.16 | 2.21e-11 |
| target_suppression_db | oracle | gen_oracle@1 | 60 | -21.1 | -20.4 | 1.63e-11 |
| clap_event_target_drop | oracle | gen_oracle@1 | 60 | 0.0426 | 0.0256 | 0.00617 |
