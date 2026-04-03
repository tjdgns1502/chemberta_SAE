# Layer 0 Feature Interpretability Plan

Spec: `docs/superpowers/specs/layer0_feature_interpretability_spec.md`

## File Map

Existing files to reuse:

- `src/chem_sae/eval/downstream.py`
- `src/chem_sae/eval/intervention.py`
- `src/chem_sae/modeling/roberta_mlm.py`
- `src/chem_sae/vendor/jumprelu.py`
- `scripts/run.py`

Planned new files:

- `src/chem_sae/eval/feature_audit.py`
- `src/chem_sae/eval/feature_intervention.py`
- `scripts/run_feature_audit.py`
- `scripts/run_feature_intervention.py`
- `tests/test_feature_audit.py`
- `tests/test_feature_intervention.py`

Planned modified files:

- `src/chem_sae/eval/__init__.py`
- `src/chem_sae/modeling/roberta_mlm.py`
- `src/chem_sae/config/intervention.py`

## Task 1: Add Feature Audit Evaluation Module

Files:

- Create `src/chem_sae/eval/feature_audit.py`
- Create `tests/test_feature_audit.py`
- Update `src/chem_sae/eval/__init__.py`

Failing test step:

- Write tests for:
  - coefficient aggregation across seeds
  - sign-consistency filtering
  - top-example selection ordering
  - single-feature metric summary shape

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_audit -v
```

Minimal implementation:

- Extract reusable latent-feature loading logic for a fixed SAE checkpoint.
- Implement per-task coefficient aggregation from multiple seeds.
- Implement feature summary rows and top-activation example collection.
- Keep outputs file-based and deterministic.

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_audit -v
```

## Task 2: Add Feature-Level Latent Edit Hook

Files:

- Modify `src/chem_sae/modeling/roberta_mlm.py`
- Create `tests/test_feature_intervention.py`

Failing test step:

- Write tests for:
  - no-op latent edit preserves current reconstruction path
  - `zero` only edits requested latent indices
  - `mean_clamp` only edits requested latent indices
  - `force_on` only edits requested latent indices

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_intervention -v
```

Minimal implementation:

- Change the layer-level SAE intervention path so it can:
  - encode attention outputs
  - optionally transform selected latent dimensions
  - decode back into reconstructed activations
- Preserve the existing layer-level no-op behavior by default.
- Keep the API narrow: selected indices, mode, and intervention values only.

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_intervention -v
```

## Task 3: Add Feature Intervention Evaluation Module and CLI

Files:

- Create `src/chem_sae/eval/feature_intervention.py`
- Create `scripts/run_feature_intervention.py`
- Modify `src/chem_sae/config/intervention.py`
- Update `src/chem_sae/eval/__init__.py`

Failing test step:

- Add tests covering:
  - intervention spec parsing
  - control-feature sampling reproducibility
  - result row schema for single-feature and grouped-feature runs

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_intervention -v
```

Minimal implementation:

- Implement feature-level intervention evaluation on top of the new latent-edit hook.
- Support modes `zero`, `mean_clamp`, `force_on`.
- Support target sets:
  - one feature
  - explicit feature group
  - matched random control group
- Write structured CSV and JSON outputs under a run-specific report directory.

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_intervention -v
```

## Task 4: Add Feature Audit CLI and Report Generation

Files:

- Create `scripts/run_feature_audit.py`
- Possibly extend `src/chem_sae/eval/feature_audit.py`

Failing test step:

- Add a smoke test or dry-run test for CLI argument parsing and output path resolution.

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_audit -v
```

Minimal implementation:

- Accept the fixed checkpoint path, task list, layer id, and top-k settings.
- Emit:
  - `feature_rankings.csv`
  - `feature_summary.json`
  - per-feature JSON cards
  - HTML or markdown atlas

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_audit -v
```

## Task 5: Run Pilot Audit on the Fixed Candidate

Files:

- No new code if earlier tasks are complete
- Output under a dedicated run directory in `artifacts/runs/sae/`

Pilot command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python scripts/run_feature_audit.py \
  --checkpoint artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt \
  --layer 0 \
  --tasks bbbp,bace_classification \
  --top-k 10 \
  --run-id layer0_audit_20260319
```

Verification step:

- Confirm the ranking and atlas files exist and contain non-empty results for both tasks.

Verification command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
find artifacts/runs/sae/layer0_audit_20260319 -maxdepth 3 -type f | sort
```

## Task 6: Run Pilot Causal Intervention

Files:

- No new code if earlier tasks are complete

Pilot command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python scripts/run_feature_intervention.py \
  --checkpoint artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt \
  --layer 0 \
  --task bbbp \
  --features <top_feature_ids> \
  --mode zero \
  --control matched_random \
  --run-id layer0_causal_pilot_20260319
```

Verification step:

- Confirm:
  - no-op matches baseline behavior
  - target-feature interventions produce non-zero deltas
  - control interventions are smaller on average

Verification command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
tail -n 20 artifacts/runs/sae/layer0_causal_pilot_20260319/reports/feature_intervention_results.csv
```

## Task 7: Full Primary-Task Validation

Files:

- No new code if earlier tasks are complete

Execution:

- Re-run the strongest 3 single features and strongest 2 groups on:
  - `BBBP`
  - `BACE`
- Run both `zero` and `force_on`.
- Run matched random controls with the same group sizes.

Verification:

- Compare mean delta metrics across:
  - task-linked features
  - grouped features
  - matched random controls

Success rule:

- Interpretable candidate is supported only if task-linked interventions are materially stronger than controls.

## Task 8: Final Report

Files:

- Create `artifacts/reports/layer0_feature_interpretability_20260319.md`

Contents:

- fixed checkpoint and task scope
- top features per task
- representative molecules
- causal intervention tables
- control comparison
- final conclusion and limits

Verification command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
sed -n '1,220p' artifacts/reports/layer0_feature_interpretability_20260319.md
```

## Execution Notes

- Keep the candidate fixed to `2048 / 0.05` for this entire phase.
- Treat `ClinTox` as sanity-only.
- Do not expand to multi-layer work until the single-layer causal story is clear.
- Prefer dry-run or smoke-test coverage before long GPU jobs.
- Do not claim interpretability from audit-only results; require intervention evidence.
