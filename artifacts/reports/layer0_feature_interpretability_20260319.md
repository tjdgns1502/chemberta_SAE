# Layer 0 Feature Interpretability Interim Report

Date: 2026-03-19

## Fixed Candidate

- SAE checkpoint: `artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt`
- Layer: `0`
- Primary tasks: `BBBP`, `BACE`
- Sanity-only task: `ClinTox`

## Why This Candidate Was Fixed

Layer 0 `2048 / base_l0=0.05` preserved downstream signal better than the `1536 / 0.05` alternative while staying reasonably sparse.

Reference downstream comparison:

- `BBBP`: baseline `0.7190`, SAE `0.7041`
- `BACE`: baseline `0.7637`, SAE `0.7726`
- `ClinTox`: baseline `0.9993`, SAE `1.0000`

Source: `artifacts/runs/sae/manual_l0_perf_compare_20260318/reports/downstream_records.csv`

## Audit Outputs

Completed audit runs:

- `layer0_audit_20260319`
- `layer0_audit_bbbp_pilot_20260319`

Generated artifacts:

- `feature_rankings.csv`
- `feature_summary.json`
- per-feature cards under `reports/feature_cards/`
- `feature_atlas.html`

Primary report root:

- `artifacts/runs/sae/layer0_audit_20260319/reports/`

## Task-Linked Feature Audit Highlights

### BBBP

Promising positive-direction features from the pilot audit:

- `1237`: coef `2.0021`, single-feature AUC `0.5611`
- `1492`: coef `1.2254`, single-feature AUC `0.5760`
- `1402`: coef `1.2768`, single-feature AUC `0.5261`

Promising negative-direction feature:

- `1011`: coef `-1.0199`, single-feature AUC `0.5706`

Interpretation:

- single features are only weak classifiers on their own
- coefficients are stable across seeds
- top examples show non-random label enrichment, especially for the positive BBBP group

### BACE

Initial positive-direction group chosen for causal testing:

- `327`: coef `1.9294`, single-feature AUC `0.4716`
- `1419`: coef `1.4908`, single-feature AUC `0.4756`
- `1785`: coef `1.4368`, single-feature AUC `0.5735`

Interpretation:

- `1785` is the cleanest single BACE feature so far
- `327` and `1419` are weaker alone, but have stable positive coefficients and are suitable for group-level validation

## Causal Pilot Results

### BBBP Single-Feature Zero Ablation

Run: `layer0_causal_bbbp_f1492_zero_20260319`

- target `1492`: baseline AUC `0.70159`, intervened `0.70214`, delta `+0.00055`
- matched control `1309`: delta `+0.00024`
- target logit shift: `-0.1139`
- control logit shift: `-0.00145`

Run: `layer0_causal_bbbp_f1011_zero_20260319`

- target `1011`: baseline AUC `0.70159`, intervened `0.70133`, delta `-0.00026`
- matched control `1310`: delta `0.00000`
- target logit shift: `+0.01027`
- control logit shift: `-0.00015`

Interpretation:

- single-feature AUC changes are small
- target interventions move logits more than matched controls

### BBBP Group Zero Ablation

Run: `layer0_causal_bbbp_group_zero_20260319`

Target group: `1237,1492,1402`

- baseline AUC `0.70159`
- intervened AUC `0.69826`
- delta `-0.00333`
- mean logit shift `-0.41658`
- mean probability shift `-0.02885`

Matched random control group: `51,228,1310`

- intervened AUC `0.70102`
- delta `-0.00057`
- mean logit shift `+0.00409`
- mean probability shift `+0.00036`

Interpretation:

- this is the clearest causal signal so far
- same-direction task-linked features produce a much larger effect than matched random controls

### BBBP Group Force-On

Run: `layer0_causal_bbbp_group_force_20260319`

Target group: `1237,1492,1402`

- baseline AUC `0.70159`
- intervened AUC `0.69668`
- delta `-0.00491`
- mean logit shift `+1.19618`
- mean probability shift `+0.08481`

Matched random control group: `51,228,1310`

- intervened AUC `0.70139`
- delta `-0.00020`
- mean logit shift `-0.00388`
- mean probability shift `-0.00014`

Interpretation:

- forcing the positive BBBP group strongly shifts prediction scores
- the control group does not reproduce that effect
- the AUC drop suggests over-driving these latents is not neutral; they are behaviorally active

### BACE Single-Feature Zero Ablation

Run: `layer0_causal_bace_f1785_zero_20260319`

- target `1785`: baseline AUC `0.72613`, intervened `0.72593`, delta `-0.00021`
- matched control `1309`: delta `+0.00007`
- target logit shift `-0.03022`
- control logit shift `-0.00591`

Interpretation:

- effect is directionally consistent but weaker than BBBP

### BACE Group Zero Ablation

Run: `layer0_causal_bace_group_zero_20260319`

Target group: `327,1419,1785`

- baseline AUC `0.72613`
- intervened AUC `0.72676`
- delta `+0.00063`
- mean logit shift `+0.02056`
- mean probability shift `+0.00544`

Matched random control group: `51,228,1310`

- intervened AUC `0.72607`
- delta `-0.00007`
- mean logit shift `-0.00210`
- mean probability shift `-0.00006`

Interpretation:

- zeroing this BACE group does not produce a robust degradation
- there is still a modest directional shift relative to control, but the effect is weak

### BACE Group Force-On

Run: `layer0_causal_bace_group_force_20260319`

Target group: `327,1419,1785`

- baseline AUC `0.72613`
- intervened AUC `0.72189`
- delta `-0.00424`
- mean logit shift `+0.04908`
- mean probability shift `+0.00495`

Matched random control group: `51,228,1310`

- intervened AUC `0.72631`
- delta `+0.00017`
- mean logit shift `-0.00175`
- mean probability shift `-0.00007`

Interpretation:

- forcing the BACE-linked group produces a clear task-specific effect
- the matched random control does not reproduce the drop
- this is weaker than the BBBP group signal, but still supports behavioral relevance

## Current Conclusion

Current evidence is enough to say that layer 0 SAE latents are not merely reconstructive bookkeeping.

What is supported now:

- task-linked features can be identified reproducibly from probe coefficients
- grouped BBBP latents have stronger causal impact than matched random features
- `force_on` and `zero` both produce directional behavior changes for the BBBP group
- BACE also shows a task-specific effect for the same-direction group under `force_on`

What is not yet supported:

- a strong BACE claim under both intervention modes
- a claim that any single latent cleanly maps to one semantic concept
- layer-wise generalization beyond layer 0

## Caveats

- causal evaluation currently keeps the linear readout fixed and measures the effect of latent intervention on the downstream representation
- observed AUC deltas are still modest in absolute size
- `ClinTox` is too small and imbalanced to be a primary interpretability benchmark here
