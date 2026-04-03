# PROBE LOG SUMMARY

- Generated: 2026-03-05 10:45:12
- Root: `/home/yoo122333/capstone/chemberta_SAE/artifacts/runs/sae`
- Launch logs (`*.probe.launch.log`): 39
- Metrics (`*/probe/metrics.json`): 30
- Schedule traces (`*/probe/schedule_trace.json`): 30

## Status Counts

- Completed (launch + metrics): **30**
- Interrupted/No final metrics (launch only): **9**
- Metrics without launch log file: **0**

## Interrupted / Launch-Only Runs

- `20260305_104410_nogit` | log: `artifacts/runs/sae/20260305_104410_nogit.probe.launch.log`
- `probe_nlat1536_l0_l0_1e4_w2_r05_20260303_113413` | log: `artifacts/runs/sae/probe_nlat1536_l0_l0_1e4_w2_r05_20260303_113413.probe.launch.log`
- `probe_nlat1536_l0_l0_2e4_none_e30_noes_20260303_213053` | log: `artifacts/runs/sae/probe_nlat1536_l0_l0_2e4_none_e30_noes_20260303_213053.probe.launch.log`
- `probe_nlat1536_l1to5_l0_13e5_none_e30_noes_20260304_140647` | log: `artifacts/runs/sae/probe_nlat1536_l1to5_l0_13e5_none_e30_noes_20260304_140647.probe.launch.log`
- `probe_nlat1536_l1to5_l0_1e4_none_e30_noes_20260304_143044` | log: `artifacts/runs/sae/probe_nlat1536_l1to5_l0_1e4_none_e30_noes_20260304_143044.probe.launch.log`
- `probe_nlat1536_l1to5_l0_2e4_none_e30_noes_20260304_143654` | log: `artifacts/runs/sae/probe_nlat1536_l1to5_l0_2e4_none_e30_noes_20260304_143654.probe.launch.log`
- `probe_nlat2048_l2_l0_2e4_exp_w1_r07_20260303_105142` | log: `artifacts/runs/sae/probe_nlat2048_l2_l0_2e4_exp_w1_r07_20260303_105142.probe.launch.log`
- `probe_nlat2048_l4_l0_12e5_w2_r05_20260303_110632` | log: `artifacts/runs/sae/probe_nlat2048_l4_l0_12e5_w2_r05_20260303_110632.probe.launch.log`
- `probe_sched_l0_1e5_exp_w1_r08` | log: `artifacts/runs/sae/probe_sched_l0_1e5_exp_w1_r08.probe.launch.log`

## Focus Runs Summary

| run_id | status | base_l0 | schedule | epochs | layers | nmse_mean | mean_l0 | dead_ratio_max |
|---|---|---:|---|---:|---|---:|---:|---:|
| `20260305_104410_nogit` | launch_only | - | - | - | - | - | - | - |
| `probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528` | completed | 0.000120 | none | 30 | 0 | 0.006654 | 155.2097 | 0.065755 |
| `probe_nlat1536_l0_l0_12e5_w2_r05_20260303_112151` | completed | 0.000120 | two_step | 8 | 0 | 0.005746 | 194.7518 | 0.054688 |
| `probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236` | completed | 0.000130 | none | 30 | 0 | 0.007178 | 139.8302 | 0.076823 |
| `probe_nlat1536_l0_l0_15e5_none_e30_noes_20260303_213455` | completed | 0.000150 | none | 30 | 0 | 0.007808 | 125.9600 | 0.098307 |
| `probe_nlat1536_l0_l0_15e5_w2_r05_20260303_112451` | completed | 0.000150 | two_step | 8 | 0 | 0.006851 | 162.6305 | 0.085286 |
| `probe_nlat1536_l0_l0_1e4_none_20260303_113440` | completed | 0.000100 | none | 8 | 0 | 0.006025 | 198.6629 | 0.051432 |
| `probe_nlat1536_l0_l0_1e4_none_e20_20260303_113902` | completed | 0.000100 | none | 20 | 0 | 0.005677 | 181.8904 | 0.046875 |
| `probe_nlat1536_l0_l0_1e4_none_e30_20260303_210659` | completed | 0.000100 | none | 30 | 0 | 0.005677 | 181.8904 | 0.046875 |
| `probe_nlat1536_l0_l0_1e4_none_e30_noes_20260303_211715` | completed | 0.000100 | none | 30 | 0 | 0.005677 | 181.8904 | 0.046875 |
| `probe_nlat1536_l0_l0_1e4_w2_r05_20260303_113413` | launch_only | - | - | - | - | - | - | - |
| `probe_nlat1536_l0_l0_2e4_none_e30_noes_20260303_213053` | launch_only | - | - | - | - | - | - | - |
| `probe_nlat1536_l0_l0_5e4_w2_r05_20260303_112928` | completed | 0.000500 | two_step | 8 | 0 | 0.013865 | 60.2462 | 0.242839 |
| `probe_nlat1536_l0_l0_5e4_w2_r10_20260303_113153` | completed | 0.000500 | two_step | 8 | 0 | 0.017008 | 48.0925 | 0.278646 |
| `probe_nlat1536_l0_l0_8e5_w2_r05_20260303_111803` | completed | 0.000080 | two_step | 8 | 0 | 0.004062 | 266.2525 | 0.024740 |
| `probe_nlat1536_l1to5_l0_13e5_none_e30_noes_20260304_140647` | launch_only | - | - | - | - | - | - | - |
| `probe_nlat1536_l1to5_l0_1e4_none_e30_noes_20260304_143044` | launch_only | - | - | - | - | - | - | - |
| `probe_nlat1536_l1to5_l0_2e4_none_e30_noes_20260304_143654` | launch_only | - | - | - | - | - | - | - |

## Top 10 by NMSE (lower is better)

| rank | run_id | nmse_mean | mean_l0 | dead_ratio_max | schedule | base_l0 |
|---:|---|---:|---:|---:|---|---:|
| 1 | `probe_sched_l0_1e5_exp_w1_r08_cached2` | 0.000489 | 940.0245 | 0.011963 | exp | 0.000010 |
| 2 | `probe_nlat2048_l2_l0_3e4_exp_w1_r01_20260303_105412` | 0.001245 | 678.0878 | 0.476562 | exp | 0.000300 |
| 3 | `probe_nlat1536_l0_l0_8e5_w2_r05_20260303_111803` | 0.004062 | 266.2525 | 0.024740 | two_step | 0.000080 |
| 4 | `probe_nlat2048_l0_8e5_w2_r05_20260223_111353` | 0.004070 | 243.6243 | 0.100098 | two_step | 0.000080 |
| 5 | `probe_sched_l0_8e5_w2_r05` | 0.004152 | 195.3022 | 0.312744 | two_step | 0.000080 |
| 6 | `probe_sched_l0_high2low_20260223_110855_r08` | 0.004747 | 164.7299 | 0.358643 | exp | 0.000120 |
| 7 | `probe_nlat2048_l0_1e4_w2_r05_20260223_111946` | 0.004940 | 203.5551 | 0.127930 | two_step | 0.000100 |
| 8 | `probe_sched_l0_8e5_none` | 0.005049 | 169.2897 | 0.350586 | none | 0.000080 |
| 9 | `probe_nlat1536_l0_l0_1e4_none_e20_20260303_113902` | 0.005677 | 181.8904 | 0.046875 | none | 0.000100 |
| 10 | `probe_nlat1536_l0_l0_1e4_none_e30_20260303_210659` | 0.005677 | 181.8904 | 0.046875 | none | 0.000100 |

## Notes

- `launch_only` runs are typically manually stopped runs before `probe/metrics.json` flush.
- Newly aborted run in this chat: `20260305_104410_nogit` (launch log exists, no probe metrics).
