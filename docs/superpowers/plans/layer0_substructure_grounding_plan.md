# Layer 0 Substructure Grounding Plan

Spec context:

- fixed SAE candidate: `layer 0 / n_latents=2048 / base_l0=0.05`
- existing audit outputs: `artifacts/runs/sae/layer0_audit_20260319/reports/`

## File Map

Existing files to reuse:

- `src/chem_sae/eval/feature_audit.py`
- `scripts/run_feature_audit.py`
- `tests/test_feature_audit.py`
- feature cards under `artifacts/runs/sae/layer0_audit_20260319/reports/feature_cards/`

Planned new files:

- `src/chem_sae/eval/feature_substructure.py`
- `scripts/run_feature_substructure.py`
- `tests/test_feature_substructure.py`
- `tests/test_feature_substructure_cli.py`

Planned modified files:

- `src/chem_sae/eval/__init__.py`

## Task 1: Add RDKit Substructure Summary Helpers

Files:

- create `src/chem_sae/eval/feature_substructure.py`
- create `tests/test_feature_substructure.py`

Failing test step:

- write tests for:
  - Murcko scaffold extraction on simple aromatic examples
  - MCS summary on a consistent molecule family
  - per-feature card summary using top examples

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_substructure -v
```

Minimal implementation:

- parse feature-card smiles with RDKit
- compute dominant Murcko scaffolds and counts
- compute an MCS SMARTS summary over top examples
- produce deterministic per-feature summaries

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_substructure -v
```

## Task 2: Add CLI for Audit-Card Grounding

Files:

- create `scripts/run_feature_substructure.py`
- create `tests/test_feature_substructure_cli.py`
- update `src/chem_sae/eval/__init__.py`

Failing test step:

- add CLI dry-run test for path resolution and feature filtering

Failure command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_substructure_cli -v
```

Minimal implementation:

- accept an audit `reports` directory
- optionally filter by task and explicit feature ids
- emit JSON and markdown summaries under a new run-specific report directory

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest tests.test_feature_substructure_cli -v
```

## Task 3: Run the Grounding Analysis

Execution:

- run the CLI against `artifacts/runs/sae/layer0_audit_20260319/reports`
- analyze the selected BBBP and BACE cards

Verification:

- confirm the summary JSON and markdown report exist
- inspect whether dominant scaffolds or MCS motifs are non-trivial for the strongest task-linked features

Success command:

```bash
cd /home/yoo122333/capstone/chemberta_SAE
find artifacts/runs/sae/layer0_substructure_grounding_20260319 -maxdepth 3 -type f | sort
```
