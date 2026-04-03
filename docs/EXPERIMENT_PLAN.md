# 실험 재실행 계획 (2026-03-17)

## 배경

### 코드 수정 사항 (이번 리뷰에서 수정됨)
1. **ActivationChunkDataset 에폭별 셔플링** — 매 에폭 동일 순서로 학습하던 버그 수정
2. **TopK/BatchTopK copy-paste 버그** — TopK 실행 시 실제로는 BatchTopK가 돌아가던 문제
3. **Intervention best model 복원** — 마지막 에폭 대신 최적 에폭 모델 반환
4. **단일 클래스 ROC-AUC 크래시** — 가드 추가
5. **NMSE division by zero** — clamp(min=1e-8) 추가
6. **L0 sweep에서 0.0 제거** — 스파시티 비활성화 값 제거
7. **W&B 메트릭 키 SAELens 호환** — prefix 제거, 공유 함수 통합

### 현재까지 유의미한 결과
- **Layer 0 sweet spot**: base_l0=1.2e-4, NMSE=0.00665, L0=155, dead_ratio=0.066
- **SAELens 정렬 후 Layer 1**: 실질적 스파시티 달성 (L0=655/4096)
- **P0 bug fix (encode)가 스파시티 성패를 결정**

---

## Phase 1: 기반 검증 (코드 수정 후 재현성 확인)

### 1.1 Layer 0 Sweet Spot 재현
```bash
python3 scripts/run_sae_probe.py \
    --layers 0 \
    --base-l0 1.2e-4 \
    --schedule none \
    --epochs 30 \
    --log-to-wandb \
    --wandb-project sae_lens_training \
    --wandb-run-name "v2_layer0_l0_1.2e-4_none"
```
**목적**: 셔플링 버그 수정 후 기존 sweet spot이 재현되는지 확인
**기대치**: NMSE ~0.006, L0 ~150, dead_ratio < 0.10
**판단 기준**: 기존 대비 ±20% 이내면 재현 성공

### 1.2 Layer 0 Schedule 비교 (수정된 코드)
```bash
# two_step schedule
python3 scripts/run_sae_probe.py \
    --layers 0 --base-l0 1.2e-4 --schedule two_step \
    --warmup-epochs 3 --epochs 30 \
    --log-to-wandb --wandb-run-name "v2_layer0_l0_1.2e-4_twostep"

# exp schedule
python3 scripts/run_sae_probe.py \
    --layers 0 --base-l0 1.2e-4 --schedule exp \
    --warmup-epochs 3 --decay-ratio 0.5 --epochs 30 \
    --log-to-wandb --wandb-run-name "v2_layer0_l0_1.2e-4_exp"
```
**목적**: L0 스케줄링이 dead neuron 역학에 미치는 영향 확인

---

## Phase 2: L0-NMSE Pareto Frontier 정밀 탐색

### 2.1 Layer 0 Fine Grid Search
```bash
for L0 in 5e-5 8e-5 1e-4 1.2e-4 1.5e-4 2e-4 3e-4; do
    python3 scripts/run_sae_probe.py \
        --layers 0 --base-l0 $L0 --schedule none --epochs 30 \
        --log-to-wandb --wandb-run-name "v2_layer0_grid_${L0}"
done
```
**목적**: 5e-5 ~ 3e-4 구간에서 최적 L0 계수 정밀 탐색
**분석**: W&B에서 NMSE vs L0 vs dead_ratio Pareto curve 생성

### 2.2 Expansion Ratio 실험
```bash
for N in 1536 2048 3072 4096; do
    python3 scripts/run_sae_probe.py \
        --layers 0 --base-l0 1.2e-4 --schedule none \
        --n-latents $N --epochs 30 \
        --log-to-wandb --wandb-run-name "v2_layer0_latents_${N}"
done
```
**목적**: n_latents 증가가 해석 가능성에 미치는 영향
**기대**: n_latents↑ → L0↑ but 개별 feature가 더 specific해질 수 있음

---

## Phase 3: 다층 레이어 비교

### 3.1 전 레이어 동일 조건 평가
Phase 2에서 확정된 최적 L0로 레이어 0-5 전부 실행:
```bash
python3 scripts/run_sae_probe.py \
    --layers 0,1,2,3,4,5 \
    --base-l0 <BEST_L0> --schedule <BEST_SCHEDULE> \
    --epochs 30 \
    --log-to-wandb --wandb-run-name "v2_all_layers"
```
**목적**: 어떤 레이어에서 SAE가 가장 잘 작동하는지 비교
**메트릭**: NMSE, L0, dead_ratio를 레이어별 비교

---

## Phase 4: 다운스트림 재평가

### 4.1 JumpReLU SAE 다운스트림 (P0 수정 후)
```bash
python3 scripts/run.py sae \
    --arch jumprelu --layers 0 \
    --log-to-wandb --wandb-run-name "v2_downstream_jumprelu"
```
**목적**: P0 encode 버그 수정 + 셔플링 수정 후 다운스트림 ROC-AUC 재측정
**비교 대상**: 기존 baseline-frozen (BBBP=0.720, BACE=0.810, ClinTox=0.859)

---

## 실험 순서 및 GPU 배분

| 우선순위 | 실험 | 예상 시간 | GPU |
|----------|------|-----------|-----|
| **1** | 1.1 Sweet spot 재현 | ~2h | GPU 0 |
| **1** | 1.2 Schedule 비교 (2개) | ~4h | GPU 1-2 |
| **2** | 2.1 Grid search (7개) | ~14h | GPU 0-5 병렬 |
| **2** | 2.2 Expansion ratio (4개) | ~8h | GPU 0-3 병렬 |
| **3** | 3.1 전 레이어 | ~12h | GPU 0-5 |
| **4** | 4.1 다운스트림 | ~6h | GPU 0 |

---

## 성공 기준

### SAE 학습 품질
- NMSE < 0.01 (Layer 0 기준)
- L0 / n_latents < 0.15 (스파시티 15% 이하)
- dead_ratio < 0.10

### 다운스트림 보존
- Baseline-frozen 대비 ROC-AUC 손실 < 5%

### W&B 대시보드
- SAELens 표준 메트릭 키로 로깅 확인
- 실시간 학습 역학 모니터링 가능

---

## W&B 사용 가이드 (수정 후)

```bash
# 최초 로그인
wandb login

# Online 모드 (기본)
python3 scripts/run_sae_probe.py --log-to-wandb ...

# Offline → 나중에 sync
WANDB_MODE=offline python3 scripts/run_sae_probe.py --log-to-wandb ...
wandb sync wandb/offline-run-*

# 기존 offline run 정리
rm -rf wandb/offline-run-20260305_*
```

메트릭 키가 이제 SAELens와 동일 (`losses/overall_loss`, `metrics/l0`, `sparsity/dead_features` 등)하므로 SAELens W&B 대시보드 템플릿을 그대로 사용 가능.
