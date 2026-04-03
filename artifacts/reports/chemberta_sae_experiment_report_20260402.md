# ChemBERTa SAE 실험 보고서

작성일: 2026-04-02

이 문서는 `chemberta_SAE` 저장소에 남아 있는 2026-03-18~2026-03-19 실험 산출물을 기준으로 정리한 요약본이다. 새로운 실험을 다시 돌려서 얻은 결과가 아니라, 당시 기록된 CSV/JSON/Markdown 보고서를 재구성한 문서다.

참고:

- 2026-04-02 정리 과정에서 대용량 `artifacts/runs/`와 `artifacts/logs/`는 삭제했다.
- 재현에 필요한 최소 근거 파일은 `artifacts/reports/evidence/` 아래에 별도로 보존했다.

## 1. 이번 실험에서 답하려던 질문

이번 실험의 핵심 질문은 두 가지였다.

1. ChemBERTa의 layer 0 hidden state에 학습한 SAE가 downstream 분류 성능을 크게 무너뜨리지 않으면서도 의미 있는 sparse latent를 만들 수 있는가?
2. 그렇게 얻은 latent 중 일부가 실제로 task와 연결되어 있고, 사람이 읽을 수 있는 화학 구조 또는 부분 패턴과 대응되는가?

관련 계획과 기준은 아래 문서에 남아 있다.

- 실험 계획: `docs/EXPERIMENT_PLAN.md`
- probe 로그 요약: `docs/PROBE_LOG_SUMMARY.md`

## 2. 어떤 후보를 고정해서 봤는가

해석 실험의 고정 후보는 layer 0의 `JumpReLU SAE`, `n_latents=2048`, `base_l0=0.05`였다.

- 체크포인트: `artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt`
- 선택 근거: `1536 / 0.05`보다 downstream 보존이 더 좋았고, 이후 audit/intervention/substructure grounding을 모두 같은 후보 위에서 진행했다.

근거 파일:

- 후보 선정 메모: `artifacts/reports/layer0_feature_interpretability_20260319.md`
- downstream 비교 원본: `artifacts/reports/evidence/manual_l0_perf_compare_20260318/downstream_records.csv`

## 3. downstream 성능 요약

Layer 0 `2048 / 0.05` 후보는 baseline 대비 큰 붕괴 없이 유지됐고, task별 차이는 다음과 같았다.

| Task | Baseline ROC-AUC | SAE ROC-AUC | Delta |
| --- | ---: | ---: | ---: |
| BBBP | 0.7190 | 0.7041 | -0.0149 |
| BACE | 0.7637 | 0.7726 | +0.0089 |
| ClinTox | 0.9993 | 1.0000 | +0.0007 |

해석:

- BBBP에서는 약간 손실이 있었지만, 완전히 무너지는 수준은 아니었다.
- BACE에서는 오히려 baseline보다 소폭 상승했다.
- ClinTox는 수치상 거의 동일하지만 데이터 규모가 작아 주된 해석 근거로 쓰기에는 보수적으로 보는 편이 맞다.

원본 위치:

- `artifacts/reports/evidence/manual_l0_perf_compare_20260318/downstream_records.csv`

## 4. SAE 자체 학습 품질

Layer 0 probe sweep 요약 기준으로, layer 0에서는 낮은 NMSE와 적당한 sparsity가 함께 나오는 sweet spot이 관찰됐다.

대표적인 reference는 다음과 같다.

- `base_l0=1.2e-4`, schedule `none`, layer `0`
- `nmse_mean=0.006654`
- `mean_l0=155.2097`
- `dead_ratio_max=0.065755`

이 값은 layer 0 sweet spot 후보를 재확인하는 기준으로 계속 인용됐다.

원본 위치:

- `docs/PROBE_LOG_SUMMARY.md`
- `docs/EXPERIMENT_PLAN.md`

추가로 2026-03-19 W&B 로컬 summary들에서도 layer 0 sweep이 이어졌고, `base_l0=0.05` 부근 후보들에 대해 `probe_nmse_mean`, `probe_mean_l0`, `probe_dead_ratio_max`가 기록되어 있다.

예:

- `wandb/run-20260319_104543-5czap3i9/files/wandb-summary.json`: `probe_nmse_mean=0.0052738812`
- `wandb/run-20260319_104544-i1hilyv8/files/wandb-summary.json`: `probe_nmse_mean=0.0066936128`
- `wandb/run-20260319_104546-z2ejjr5u/files/wandb-summary.json`: `probe_nmse_mean=0.0081382224`

## 5. Feature audit에서 찾은 핵심 latent

고정 후보(`2048 / 0.05`)에 대해 BBBP와 BACE에서 probe coefficient가 큰 latent를 추려 feature audit을 수행했다.

### BBBP

양의 방향 주요 feature:

- `1237`: coef `2.0021`, single-feature AUC `0.5611`
- `1492`: coef `1.2254`, single-feature AUC `0.5760`
- `1402`: coef `1.2768`, single-feature AUC `0.5261`

음의 방향 feature:

- `1011`: coef `-1.0199`, single-feature AUC `0.5706`

### BACE

주요 양의 방향 feature:

- `327`: coef `1.9294`, single-feature AUC `0.4716`
- `1419`: coef `1.4908`, single-feature AUC `0.4756`
- `1785`: coef `1.4368`, single-feature AUC `0.5735`

해석:

- 단일 feature 하나만으로는 강한 classifier라고 보긴 어렵다.
- 그래도 coefficient 방향성이 안정적이고, top example의 label enrichment가 있어 group-level causal test 후보로는 충분했다.

원본 위치:

- `artifacts/reports/layer0_feature_interpretability_20260319.md`
- `artifacts/reports/layer0_feature_interpretability_20260319.md`
- `artifacts/reports/lab_meeting_layer0_sae_source_20260319.md`

## 6. Causal intervention 결과

이 단계에서는 선택된 latent를 `zero` 또는 `force_on`으로 개입했을 때 downstream ROC-AUC와 logit이 실제로 바뀌는지 확인했다. 비교를 위해 matched random control도 같이 측정했다.

### BBBP group intervention

Target group: `1237,1492,1402`

- zero: `0.70159 -> 0.69826`, delta `-0.00333`
- matched control zero: delta `-0.00057`
- force_on: `0.70159 -> 0.69668`, delta `-0.00491`
- matched control force_on: delta `-0.00020`

해석:

- BBBP에서는 task-linked feature group이 random control보다 훨씬 큰 변화를 만들었다.
- 즉 이 그룹은 단순한 재구성 bookkeeping이 아니라 실제 downstream 행동과 연결된 축이라고 볼 수 있다.

### BACE group intervention

Target group: `327,1419,1785`

- zero: `0.72613 -> 0.72676`, delta `+0.00063`
- matched control zero: delta `-0.00007`
- force_on: `0.72613 -> 0.72189`, delta `-0.00424`
- matched control force_on: delta `+0.00017`

해석:

- BACE는 zero ablation에서는 신호가 약했다.
- 하지만 force-on에서는 control이 재현하지 못하는 하락이 보여, feature group이 task-specific effect를 가진다는 점은 지지된다.

원본 위치:

- `artifacts/reports/layer0_feature_interpretability_20260319.md`
- `artifacts/reports/evidence/layer0_causal_bbbp_group_zero_20260319/feature_intervention_summary.json`
- `artifacts/reports/evidence/layer0_causal_bbbp_group_force_20260319/feature_intervention_summary.json`
- `artifacts/reports/evidence/layer0_causal_bace_group_zero_20260319/feature_intervention_summary.json`
- `artifacts/reports/evidence/layer0_causal_bace_group_force_20260319/feature_intervention_summary.json`

## 7. 사람이 읽을 수 있는 구조 해석이 가능했는가

결론부터 말하면, 일부는 가능했고 특히 BACE feature `1419`가 가장 설득력 있었다.

### BACE feature 1419

- coef `1.4908`
- single-feature AUC `0.4756`
- `mcs_num_atoms=18`
- dominant scaffold: `O=C1NC=NC1(c1ccccc1)c1ccccc1` (`5/10`)

token localization에서도 `O`, `NC`, `CN` 같은 토큰이 반복적으로 높은 activation을 보였다.

이 점 때문에 feature `1419`는 단순한 CLS 전역 요약 축이라기보다, diaryl + carbonyl-containing N-rich heterocycle 계열에 가까운 구조 family를 반영하는 latent 후보로 해석됐다.

### BACE feature 1785

- single-feature AUC는 `0.5735`로 더 좋았지만
- dominant scaffold 일치가 `2/10`, `mcs_num_atoms=9`에 그쳤고
- token localization은 대부분 `0`에 가까워 local motif보다는 global signal 쪽 해석이 더 강했다.

### BBBP 핵심 feature들

BBBP의 `1011`, `1237`, `1492`, `1402`는 causal effect는 있었지만,

- dominant scaffold 일치가 낮고
- MCS가 짧거나 없고
- token localization도 거의 잡히지 않았다.

즉 BBBP에서는 “특정 substructure detector”라기보다 “분자 전체 성질을 압축한 CLS-global factor”에 가까운 latent가 더 강하게 보였다.

원본 위치:

- `artifacts/reports/layer0_feature_token_localization_20260319.md`
- `artifacts/reports/lab_meeting_layer0_sae_source_20260319.md`
- `artifacts/reports/bace_decision_memo_source_20260319.md`

## 8. 이번 실험에서 말할 수 있는 것

이번 layer 0 파일럿 결과만으로 지지되는 주장은 아래 정도다.

1. Layer 0 SAE latent는 단순 재구성용 bookkeeping 이상으로 downstream behavior와 연결돼 있다.
2. BBBP에서는 feature group intervention이 random control보다 훨씬 큰 변화를 만들어 causal relevance를 보여준다.
3. BACE에서도 force-on intervention 기준으로 task-specific effect가 있다.
4. 사람이 읽을 수 있는 구조 해석은 일부 가능하며, 가장 좋은 예시는 BACE feature `1419`다.
5. 반면 BBBP 핵심 latent는 local motif보다 CLS-global molecular factor 성격이 더 강하다.

## 9. 아직 말하면 안 되는 것

현재 기록만으로는 다음 주장을 강하게 하기는 어렵다.

1. “특정 single latent가 하나의 명확한 화학 개념을 완전히 분리해서 표현한다.”
2. “BACE에서도 zero/force-on 모두에서 강한 causal effect가 일관되게 나온다.”
3. “Layer 0에서 본 현상이 layer 1~2 이상에서도 그대로 일반화된다.”

즉 이번 결과는 완성 결론이라기보다, “계속 파볼 가치가 있는 신호를 확보한 layer 0 파일럿”으로 표현하는 것이 가장 정확하다.

## 10. 다음 단계 제안

기록에 남아 있는 후속 제안은 아래와 같다.

1. BBBP global feature와 분자 descriptor(`molecular weight`, `logP`, `aromatic ring count`, `hetero atom count`) 상관 분석
2. layer 1~2에도 같은 audit/intervention/substructure 파이프라인 복제
3. counterfactual molecule pair를 만들어 motif 변화와 latent/logit 변화를 같이 확인

관련 문서:

- `artifacts/reports/lab_meeting_layer0_sae_source_20260319.md`
- `docs/EXPERIMENT_PLAN.md`

## 11. 짧은 전달용 문장

친구에게 아주 짧게 보낼 때는 아래 정도로 요약하면 된다.

> ChemBERTa layer 0에 SAE를 붙여서 본 파일럿에서, downstream 성능은 크게 안 무너졌고 특히 BACE에서는 baseline보다 약간 좋아졌다.  
> BBBP/BACE 둘 다 task-linked latent group은 random control보다 큰 causal effect를 보였고, BACE에서는 사람도 읽을 수 있는 scaffold 계열 feature 하나(1419번)가 나왔다.  
> 다만 BBBP 쪽은 local substructure보다는 분자 전체 성질을 담은 global latent가 더 강해서, 지금 단계 결론은 “해석 가능성 신호를 찾은 파일럿” 정도가 가장 정확하다.
