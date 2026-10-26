# Benchmark: all 54 problems on the runner (RTX 5090)

Median of 20 launches after 3 warm-ups, from `python tools/bench.py` on 2026-10-26. No clock locking: indicative only.

| problem | result | median us | registers | max abs err | compile s |
|---|---|---:|---:|---:|---:|
| a01_attention | pass | 84.6 | 136 | 6.6e-07 | 0.64 |
| a02_attention_128 | pass | 35.5 | 80 | 4.5e-07 | 0.59 |
| f01_rmsnorm | pass | 4.8 | 20 | 1.9e-06 | 0.52 |
| f02_row_softmax | pass | 3.1 | 20 | 7.5e-09 | 0.51 |
| f03_l2norm | pass | 3.1 | 20 | 1.5e-08 | 0.68 |
| m01_matmul_512 | pass | 15.9 | 40 | 8.4e-05 | 0.49 |
| m02_matmul_rect | pass | 5.0 | 35 | 1.1e-05 | 0.53 |
| p00_scalar_add | pass | 3.1 | 16 | 0.0e+00 | 0.52 |
| p01_vector_add | pass | 3.1 | 16 | 0.0e+00 | 0.46 |
| p02_scale_shift | pass | 3.2 | 22 | 0.0e+00 | 0.46 |
| p03_relu_mul | pass | 3.2 | 20 | 0.0e+00 | 0.46 |
| r01_array_sum | pass | 3.0 | 18 | 3.8e-06 | 0.52 |
| r02_big_sum | pass | 21.1 | 18 | 1.5e-05 | 0.51 |
| r03_mean | pass | 13.0 | 18 | 9.3e-10 | 0.50 |
| r04_row_sum | pass | 3.1 | 20 | 5.7e-06 | 0.53 |
| xa01_attention_32 | pass | 13.0 | 64 | 2.4e-07 | 0.56 |
| xa02_attention_256 | pass | 562.3 | 255 | 6.6e-07 | 0.52 |
| xa03_attention_rect | pass | 64.1 | 62 | 2.4e-07 | 0.55 |
| xf01_layernorm | pass | 6.8 | 40 | 1.7e-06 | 0.49 |
| xf02_softmax_wide | pass | 4.8 | 20 | 5.6e-09 | 0.56 |
| xf04_center | pass | 3.1 | 20 | 2.4e-07 | 0.56 |
| xf05_rmsnorm_384 | pass | 4.7 | 20 | 9.5e-07 | 0.77 |
| xf06_softmax_scaled | pass | 3.1 | 22 | 7.5e-09 | 0.50 |
| xm01_matmul_256x128 | pass | 7.0 | 32 | 1.0e-05 | 0.58 |
| xm02_matmul_small | pass | 3.3 | 35 | 3.8e-06 | 0.52 |
| xm03_matmul_tall | pass | 9.1 | 40 | 7.6e-06 | 0.83 |
| xm04_matmul_deep | pass | 13.2 | 40 | 6.9e-05 | 0.61 |
| xp01_gelu | pass | 3.2 | 24 | 2.4e-07 | 0.55 |
| xp02_sigmoid_mul | pass | 3.1 | 16 | 1.2e-07 | 0.49 |
| xp03_tanh_exp | pass | 3.1 | 23 | 4.8e-07 | 0.55 |
| xp04_bias_2d | pass | 3.3 | 24 | 0.0e+00 | 0.58 |
| xp05_silu | pass | 3.2 | 17 | 2.4e-07 | 0.72 |
| xp06_div_abs | pass | 3.1 | 23 | 2.4e-07 | 0.51 |
| xp07_where | pass | 3.2 | 20 | 0.0e+00 | 0.47 |
| xp08_clamp_sqrt | pass | 3.2 | 16 | 1.2e-07 | 0.47 |
| xr01_max | pass | 4.8 | 18 | 0.0e+00 | 0.51 |
| xr02_row_mean | pass | 3.0 | 20 | 1.5e-08 | 0.54 |
| xr03_sumsq | pass | 6.0 | 18 | 0.0e+00 | 0.53 |
| xr04_row_max | pass | 2.8 | 18 | 0.0e+00 | 0.52 |
| xr06_abs_sum | pass | 3.0 | 20 | 6.1e-05 | 0.48 |
| ha01_attention_odd | pass | 33.4 | 152 | 2.7e-07 | 0.50 |
| ha02_attention_512 | pass | 191.1 | 224 | 3.9e-07 | 0.53 |
| hf01_global_norm | pass | 4.8 | 18 | 1.2e-10 | 0.50 |
| hf02_softmax_odd | pass | 4.0 | 20 | 7.5e-09 | 0.53 |
| hf03_layernorm_affine | pass | 10.9 | 40 | 2.9e-06 | 0.53 |
| hf04_softmax_3d | pass | 3.3 | 27 | 3.0e-08 | 0.51 |
| hm01_matmul_1024 | pass | 70.3 | 110 | 2.2e-04 | 0.62 |
| hm02_matmul_odd | pass | 6.8 | 36 | 2.9e-06 | 0.60 |
| hm03_matmul_2048x256 | pass | 22.2 | 80 | 5.3e-05 | 0.84 |
| hp01_odd | pass | 3.2 | 14 | 0.0e+00 | 0.53 |
| hp02_3d_bcast | pass | 3.1 | 20 | 0.0e+00 | 0.51 |
| hr01_sum_unpinned | pass | 8.7 | 20 | 1.8e-04 | 0.48 |
| hr02_row_sum_unpinned | pass | 4.7 | 20 | 3.1e-05 | 0.50 |
| hr03_odd_rows | pass | 3.1 | 20 | 3.8e-06 | 0.48 |
