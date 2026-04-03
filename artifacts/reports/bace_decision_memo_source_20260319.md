# BACE 중심 의사결정 메모

## 질문

이 SAE 해석 실험을 계속해야 하는가?

## 짧은 답

조심스러운 예.

다만 지금 단계에서 말할 수 있는 것은 **"layer 0 전체가 잘 해석된다"** 가 아니라, **"BACE feature 1419라는 첫 번째 성공 사례가 보였다"** 에 가깝다.

## 왜 BACE가 핵심인가

현재 결과 중에서 사람이 이해할 수 있는 화학적 의미와 가장 잘 연결되는 신호가 BACE에서 나왔다.

특히 `feature 1419`는 다음 조건을 동시에 만족한다.

1. task-linked
- BACE probe coefficient가 안정적으로 양수다.

2. causal
- `force_on` 했을 때 BACE AUC가 `0.72613 -> 0.72189`로 내려간다.
- matched random control은 거의 안 변한다.

3. 구조 family 존재
- dominant scaffold가 top 10 중 `5/10`
- MCS가 `18 atoms`

4. token-locality 존재
- CLS 평균보다 non-CLS 토큰 최대 활성 평균이 훨씬 크다.
- 즉 단순한 CLS 전역 요약 축이 아니라, 실제 substructure 쪽 신호를 갖고 있다.

## 이 feature가 의미하는 것

현재 해석으로는 `feature 1419`는 대략 다음 구조 family와 연결된다.

- carbonyl-containing N-rich heterocycle
- amidine / imine-like core
- diaryl scaffold family

쉽게 말하면 이 노드는

- "BACE 쪽 분자에서 자주 보이는 구조 계열"

을 어느 정도 따로 떼어낸 feature로 보인다.

## 왜 사람이 보는 관점과 연결된다고 말할 수 있나

BACE medicinal chemistry에서는 원래 이런 걸 본다.

- catalytic aspartates와 상호작용하는 headgroup
- aryl / heteroaryl scaffold
- S pocket을 치는 치환기
- potency와 함께 BBB, P-gp, pKa 같은 물성 문제

즉 사람도 “이 분자가 BACE inhibitor답게 생겼는가”를 볼 때,

- carbonyl
- N-rich core
- diaryl / heteroaryl scaffold

같은 축을 본다.

이번 feature 1419는 그 관점과 맞닿아 있다.

## 왜 이 추론을 믿을 수 있나

한 가지 증거가 아니라 **서로 다른 네 종류의 증거가 같은 방향**을 가리킨다.

1. task linkage
- BACE에서만 중요하게 뜬다.

2. causal effect
- random control보다 target group 개입 효과가 크다.

3. structure grounding
- 공통 scaffold와 큰 MCS가 잡힌다.

4. token localization
- 실제 비-CLS 토큰에서도 활성화된다.

이 네 가지가 동시에 맞으면 “우연히 상관만 있는 축”일 가능성은 많이 줄어든다.

## 아직 조심해야 하는 점

이건 아직 다음을 의미하지는 않는다.

- 이 feature가 정확한 결합 모드 하나를 완전히 표현한다.
- 이 feature 하나만으로 BACE 활성을 완벽히 설명한다.
- layer 0 전체가 다 해석 가능하다.

즉 현재 수준은:

- “좋은 첫 증거”
- 하지만 아직 “최종 증명”은 아님

## BBBP와 비교했을 때 왜 BACE가 더 중요하나

BBBP도 causal effect는 나왔다.
하지만 BBBP 쪽 핵심 feature들은 대부분 `CLS-global` 축이었다.

즉:

- BBBP: “분자 전체 타입을 요약한 latent” 성격이 강함
- BACE: “실제 구조 family와 연결된 latent”가 보임

해석 실험의 목적이 “사람이 이해할 수 있는 개념을 latent에서 찾는 것”이라면, 현재는 BACE가 훨씬 강한 성공 사례다.

## 그래서 계속해야 하나

### 내 판단

계속하는 게 맞다.

단, 조건이 있다.

- 전 레이어 전체를 넓게 치는 방식으로 계속하면 안 된다.
- **BACE 중심으로 짧고 강하게 검증하는 2차 단계**로 들어가야 한다.

## 다음 단계 제안

### 계속할 가치가 있는 이유

- 이미 feature 1419라는 “사람이 설명 가능한 latent” 후보가 하나 나왔다.
- 이건 SAE가 단순 압축기인지, 해석 가능한 representation 분해기인지 가르는 중요한 신호다.

### 2차 검증에서 꼭 해야 할 것

1. feature 1419 enrichment 확대 검증
- top 10이 아니라 더 넓은 상위 샘플에서 scaffold family가 유지되는지 확인

2. BACE에서 추가 localizable feature 찾기
- 1419만 있는지, 2~3개 더 나오는지 확인

3. descriptor / fingerprint 상관
- feature activation이 어떤 화학 descriptor와 연결되는지 확인

4. negative control 강화
- 비슷한 activation 크기지만 BACE와 무관한 feature와 비교

## 중단해야 하는 경우

다음 중 하나면 여기서 멈추는 게 맞다.

- 1419 외에는 비슷한 localizable feature가 전혀 안 나옴
- 더 넓은 샘플에서 scaffold enrichment가 무너짐
- stronger control을 두면 causal / structural signal이 사라짐

## GO / NO-GO 기준

### GO

- BACE에서 최소 2개 이상 feature가
- task-linked
- causal
- scaffold-grounded

세 조건을 동시에 만족하면 계속할 가치가 충분하다.

### NO-GO

- 1419 하나만 예외적으로 보이고 나머지는 전부 CLS-global이면
- “layer 0 BACE에 우연한 한 케이스가 있었다”로 정리하고 멈추는 게 낫다.

## 발표 때 바로 말할 문장

- “오늘 결과는 layer 0 전체 성공 보고가 아니라, BACE에서 의미 있는 첫 사례를 본 정도로 이해하면 됩니다.”
- “그 사례가 바로 BACE feature 1419이고, 이 feature는 실제 구조 계열과 연결됩니다.”
- “그래서 실험을 접을 단계는 아니지만, 아직 과하게 일반화하면 안 되고 BACE 중심으로 더 확인해야 합니다.”
