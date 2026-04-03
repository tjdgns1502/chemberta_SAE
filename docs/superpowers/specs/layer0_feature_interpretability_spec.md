# Layer 0 Feature Interpretability Spec

## Status

Approved in chat on 2026-03-19.

## Fixed Candidate

- Primary SAE candidate: `layer 0 / JumpReLU / n_latents=2048 / base_l0=0.05`
- Checkpoint: `artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt`
- Baseline comparator: original hidden layer 0
- Secondary comparator: none for this phase

## Goal

Decide whether the selected layer-0 SAE is genuinely interpretable, not merely sparse and downstream-preserving.

This phase must answer two questions:

1. Do a small number of SAE latent features align with stable, task-relevant molecular patterns?
2. Does intervening on those latent features causally change downstream behavior more than matched control features?

## Primary Tasks

- Primary downstream tasks: `BBBP`, `BACE`
- Sanity-only task: `ClinTox`

`ClinTox` must not be used as the main interpretability claim because its scaffold split is too small and unstable.

## Non-Goals

- No multi-layer interpretability sweep yet
- No search over multiple SAE checkpoints in this phase
- No claim of chemistry-grounded causality beyond model-internal causal evidence
- No large UI/dashboard redesign work

## Experiment A: Task-Linked Feature Audit

### Purpose

Find SAE features that are consistently associated with downstream prediction signal.

### Inputs

- Fixed SAE checkpoint above
- Layer-0 attention outputs
- Downstream datasets for `BBBP` and `BACE`

### Method

1. Extract layer-0 latent features for train/test splits.
2. Fit linear probes on latent features for each task using the existing 5-seed convention.
3. Aggregate probe coefficients across seeds.
4. Rank features by task relevance using:
   - coefficient magnitude
   - sign consistency across seeds
   - rank stability across seeds
5. For each selected feature, compute:
   - mean activation on positive vs negative examples
   - single-feature ROC-AUC
   - activation frequency / sparsity
   - overlap or redundancy with nearby high-ranked features
6. For each selected feature, collect top-activating molecules from:
   - train split
   - test split
   - unlabeled or background activation pool if available
7. Emit a feature atlas with one card per feature.

### Output Artifacts

- `feature_rankings.csv`
- `feature_summary.json`
- `feature_cards/feature_<id>.json`
- `feature_atlas.html` or markdown equivalent

### Selection Rule

For each task:

- Keep the top positive and top negative features separately.
- Prefer features with sign consistency `>= 4/5`.
- Prefer features with stable rank across seeds.
- Exclude obviously redundant features when their top examples and decoder similarity are near-duplicates.

### Success Criteria

- At least 5 non-redundant features across `BBBP` and `BACE` pass the stability filter.
- Their top-activating molecules show recognizable recurring structure or chemistry-adjacent pattern.
- Their single-feature signal is meaningfully above random-control features.

## Experiment B: Causal Feature Intervention

### Purpose

Test whether identified features are causally important for task behavior rather than merely correlated.

### Required Capability

The current code supports layer-level SAE replacement, but not feature-level latent editing. This phase will add feature-level latent transforms between SAE encode and decode.

### Intervention Modes

- `zero`: set selected latent features to zero
- `mean_clamp`: set selected latent features to a background mean value
- `force_on`: set selected latent features to a fixed high quantile or calibrated positive value

The implementation must support editing only selected latent dimensions while leaving all others unchanged.

### Experiment Ladder

1. Single-feature ablation on top task-linked features
2. Single-feature force-on on the same features
3. Group ablation on top-k selected features
4. Control interventions on:
   - importance-matched random features
   - low-importance features
   - sign-opposite features where applicable

### Readouts

- Fast readout:
   - probe logit shift
   - class probability shift
   - latent activation shift
- Full readout:
   - downstream ROC-AUC delta on `BBBP`
   - downstream ROC-AUC delta on `BACE`

### Success Criteria

- Top-ranked feature ablations shift prediction behavior more than matched random controls.
- Group ablation has a larger effect than most single-feature ablations.
- Force-on intervention pushes predictions in the expected direction on at least one primary task.
- The observed effect reproduces on both `BBBP` and `BACE`, or strongly on one and directionally on the other.

## Controls

- No-op intervention must exactly reproduce the current latent-probe path.
- Random feature intervention must be included in every causal report.
- `ClinTox` is allowed only as a sanity check, not as the core success metric.

## Reporting

The final report for this phase must include:

- the fixed checkpoint path
- task-specific top features
- representative molecules per feature
- intervention deltas for task-linked vs control features
- a short interpretation for each causal feature candidate
- explicit limitations and failure cases

## Risks

- Features may be predictive but semantically redundant.
- High-ranked features may reflect tokenizer or scaffold artifacts rather than chemistry.
- `force_on` values can create out-of-distribution latent states if not calibrated from observed activations.
- `ClinTox` can misleadingly look strong due to small split size.

## Decision Rule After This Phase

Promote the layer-0 SAE as an interpretable candidate only if both conditions hold:

1. Task-linked audit finds stable, non-trivial, non-redundant features.
2. Feature-level interventions produce causal effects stronger than matched controls.
