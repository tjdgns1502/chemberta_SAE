# ChemBERTa SAE Layer 0 랩미팅 정리

## 한 줄 결론

Layer 0 결과는 **완성된 결론이 아니라 파일럿 결과**다.

다만 그 파일럿에서, BACE 쪽에서는 사람이 이해할 수 있는 feature 후보 하나가 보였다.

## 발표용 30초 버전

- 이번에는 layer 0 SAE를 `2048 / l0=0.05`로 고정해서 해석 실험을 진행했다.
- 성능은 아주 크게 무너지지 않았고, BACE에서는 baseline보다 약간 좋아졌다.
- BACE에서 한 개의 feature는 실제 구조 계열과 연결됐다.
- 반면 BBBP는 특정 부분구조보다 분자 전체 성질을 요약한 신호가 더 강했다.
- 그래서 지금 단계의 결론은 “성공 확정”이 아니라 “계속 볼 만한 신호를 찾았다”이다.

## 왜 이 후보를 고정했는가

Layer 0 후보들 중 `2048 / l0=0.05`가 가장 균형이 좋았다.

- BBBP: baseline `0.7190` -> SAE `0.7041`
- BACE: baseline `0.7637` -> SAE `0.7726`
- ClinTox: baseline `0.9993` -> SAE `1.0000`

비교용 `1536 / l0=0.05`는 BBBP와 BACE에서 더 불리했다.

## 이번에 실제로 한 작업

1. Feature audit
- BBBP, BACE에서 probe coefficient가 큰 latent를 뽑아 task-linked feature를 정리했다.

2. Causal intervention
- 선택된 latent를 `zero`, `force_on`으로 개입해서 random control보다 더 큰 변화가 나는지 확인했다.

3. Substructure grounding
- top-activating molecule에서 공통 scaffold와 MCS를 뽑아서 사람이 읽을 수 있는 구조 family가 있는지 확인했다.

4. Token localization
- latent가 특정 토큰/부분구조에 붙는지, 아니면 CLS 전역 요약에만 실리는지 확인했다.

## 핵심 결과 1: downstream 성능

- `2048 / 0.05`는 baseline 대비 큰 붕괴 없이 유지됐다.
- BBBP는 약 `-0.015` ROC-AUC 하락
- BACE는 약 `+0.009` ROC-AUC 상승
- ClinTox는 거의 동일하지만 데이터가 작아서 해석 비중은 낮게 둔다.

## 핵심 결과 2: causal intervention

### BBBP

Target group: `1237,1492,1402`

- zero: `0.70159 -> 0.69826`, delta `-0.00333`
- matched random control zero: delta `-0.00057`
- force_on: `0.70159 -> 0.69668`, delta `-0.00491`
- matched random control force_on: delta `-0.00020`

해석:
- BBBP에서는 task-linked feature group이 실제 예측 행동에 영향을 준다.
- 단, 이 feature들은 뒤에서 보듯 local substructure 하나라기보다 global factor에 가깝다.

### BACE

Target group: `327,1419,1785`

- zero: `0.72613 -> 0.72676`, delta `+0.00063`
- matched random control zero: delta `-0.00007`
- force_on: `0.72613 -> 0.72189`, delta `-0.00424`
- matched random control force_on: delta `+0.00017`

해석:
- BACE에서도 force_on에서는 분명한 task-specific effect가 있다.
- zero는 BBBP만큼 강하지 않다.

## 핵심 결과 3: 구조 해석

### 가장 설득력 있는 feature: BACE feature 1419

- dominant scaffold가 `5/10`
- MCS가 `18 atoms`
- diaryl + carbonyl-containing N-rich heterocycle / amidine-like core에 가깝다.
- 토큰 수준에서도 `O`, `NC`, `CN`이 강하게 활성화된다.

해석:
- 이 feature는 사람이 medicinal chemistry에서 보는 scaffold/headgroup family와 정렬된다고 볼 수 있다.

### BACE feature 1785

- 구조 family는 보이지만 1419보다 덜 깔끔하다.
- 일부 aza-heteroaryl / bulky aromatic scaffold를 잡는 것으로 보인다.
- 다만 token-local feature라기보다 CLS-global 신호에 더 가깝다.

### BBBP feature 1011 / 1237 / 1492

- causal effect는 컸다.
- 하지만 substructure는 퍼져 있고, token localization도 거의 안 잡혔다.
- 실제로는 특정 원자 하나보다 CLS에 압축된 전역 분자 특성을 담는 latent로 보는 게 맞다.

## 쉬운 설명: CLS-global feature란?

- local feature: “여기 carbonyl, 여기 amidine 때문에 켜진다”
- CLS-global feature: “이 분자 전체가 이런 타입이라 켜진다”

이번 layer 0에서는 BBBP 쪽 핵심 feature 다수가 후자였다.

즉 지금 결과는:
- “substructure-level disentanglement가 일부 있다”
- 동시에
- “CLS summary 안의 global molecular factor disentanglement가 더 강하다”

## 사람이 보는 관점과 연관이 있는가

있다.

특히 BACE 쪽은 사람이 실제로 보는 기준과 겹친다.

- carbonyl / amidine-like headgroup
- diaryl scaffold family
- heterocycle core

즉 모델이 완전히 임의의 패턴을 잡은 것이 아니라, medicinal chemistry 관점과 겹치는 방향을 일부 latent가 담고 있다.

## 현재까지의 정리

1. `2048 / l0=0.05`는 layer 0 파일럿 후보로는 유지할 만하다.
2. layer 0 latent는 단순 재구성용이 아니라 downstream behavior와 연결된다.
3. BACE에서는 사람이 설명할 수 있는 구조 계열 feature가 하나 보였다.
4. BBBP 핵심 latent는 local motif보다 CLS-global factor에 가깝다.

## 다음 단계

1. global feature와 descriptor 상관 보기
- molecular weight
- logP
- aromatic ring count
- hetero atom count

2. layer 1~2에도 같은 해석 파이프라인 복제
- 더 local한 feature가 나오는지 확인

3. counterfactual molecule pair 분석
- 특정 motif 유무에 따라 latent와 logit이 같이 움직이는지 확인

## 발표 때 강조하면 좋은 문장

- “Layer 0 결과는 완성본이라기보다 파일럿입니다.”
- “그래도 BACE에서는 사람이 이해할 수 있는 구조 계열 feature 하나가 나왔습니다.”
- “반면 BBBP는 특정 부분구조보다 분자 전체 요약 신호가 더 강했습니다.”
