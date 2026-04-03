# Chat Context Transfer (2026-03-04)

## Scope
- Project: `chemberta_SAE`
- Current working line: JumpReLU probe tuning for layer 0
- Goal: find stable tradeoff between reconstruction (`nmse`) and sparsity (`mean_l0`, `dead_ratio`)

## Active Probe Protocol
- Script: `scripts/run_sae_probe.py`
- Fixed settings:
  - `--layers 0`
  - `--n-latents 1536`
  - `--schedule none`
  - `--epochs 30`
  - `--disable-early-stopping`
- Sweep axis used: `--base-l0`

## Important Code Facts
- `--disable-early-stopping` was added to probe runner.
  - File: `scripts/run_sae_probe.py`
  - Effect: sets `cfg.early_stopping_patience = epochs + 1`
- Current JumpReLU implementation:
  - Gate is hard step in forward (`_JumpReLUFn`)
  - L0 surrogate in loss is tanh-smoothed (`l0_step_loss`)
  - File: `src/chem_sae/vendor/jumprelu.py`

## Completed Reference Runs (key)
- `probe_nlat1536_l0_l0_1e4_none_e30_noes_20260303_211715`
  - aggregate: `nmse=0.0056768`, `mean_l0=181.89`, `dead=0.046875`
- `probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528`
  - aggregate: `nmse=0.0066539`, `mean_l0=155.21`, `dead=0.065755`
- `probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236`
  - aggregate: `nmse=0.0071781`, `mean_l0=139.83`, `dead=0.076823`
- `probe_nlat1536_l0_l0_15e5_none_e30_noes_20260303_213455`
  - aggregate: `nmse=0.0078076`, `mean_l0=125.96`, `dead=0.098307`

## Finalization Status
- Both previously interrupted runs are now finalized:
  - `probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528`
  - `probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236`
- `probe/metrics.json` now exists for all four comparison runs.

## Side-by-Side Comparison (layer 0, aggregate)
| base_l0 | run_id | nmse | mean_l0 | dead_ratio |
|---|---|---:|---:|---:|
| `1.0e-4` | `probe_nlat1536_l0_l0_1e4_none_e30_noes_20260303_211715` | `0.0056768` | `181.89` | `0.046875` |
| `1.2e-4` | `probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528` | `0.0066539` | `155.21` | `0.065755` |
| `1.3e-4` | `probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236` | `0.0071781` | `139.83` | `0.076823` |
| `1.5e-4` | `probe_nlat1536_l0_l0_15e5_none_e30_noes_20260303_213455` | `0.0078076` | `125.96` | `0.098307` |

## Current Interpretation
- `1e-4`: best reconstruction, but l0 relatively high.
- `1.5e-4`: lowest l0 of this sweep, but nmse/dead degrade significantly.
- `1.2e-4` looks like the best compromise point from this set.
- `1.3e-4` is a valid middle option if stronger sparsity is preferred over reconstruction.

## Metrics Files
- `artifacts/runs/sae/probe_nlat1536_l0_l0_1e4_none_e30_noes_20260303_211715/probe/metrics.json`
- `artifacts/runs/sae/probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528/probe/metrics.json`
- `artifacts/runs/sae/probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236/probe/metrics.json`
- `artifacts/runs/sae/probe_nlat1536_l0_l0_15e5_none_e30_noes_20260303_213455/probe/metrics.json`

## Suggested First Message in New Chat
```
Continue from `chemberta_SAE/docs/CHAT_CONTEXT_TRANSFER.md`.
From the finalized 1e-4 / 1.2e-4 / 1.3e-4 / 1.5e-4 results,
pick a default l0 setting for layer-0 JumpReLU training and justify it.
```
