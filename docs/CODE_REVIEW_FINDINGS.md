# CODE REVIEW FINDINGS

정렬 기준: `P0(치명) -> P1(높음) -> P2(중간) -> P3(낮음)`

업데이트(2026-03-04):
- 수정 완료: `P0-1`, `P1-1`, `P1-2`, `P1-3`, `P2-1`, `P2-2`, `P2-3`, `P3-1`, `P3-2`
- 잔여 이슈: 없음 (코드/테스트 반영 완료)

---

## P0

### P0-1
- 심각도: `P0`
- 증상: JumpReLU downstream 평가가 게이팅된 latent가 아니라 pre-activation을 사용할 수 있어, JumpReLU 실험의 downstream ROC-AUC 해석이 무효화될 위험이 큼.
- 재현 조건:
  - `arch=jumprelu`로 `run_all` 또는 `run_sweep(stage3 winner)` 실행 후 `evaluate_downstream` 호출 경로 진입.
- 파일/라인:
  - `src/chem_sae/eval/downstream.py:49-50`
  - `src/chem_sae/vendor/sae_core.py:66-68`
  - `src/chem_sae/vendor/jumprelu.py:101-106`
  - `src/chem_sae/vendor/jumprelu.py:122-126`
- SAELens 기준:
  - `sae_lens/saes/jumprelu_sae.py:265-273` (feature는 `JumpReLU.apply(...)` 결과)
- 영향:
  - JumpReLU arch 후보 비교/스윕에서 downstream 성능 지표가 실제 sparsified representation을 반영하지 않을 수 있음.
- 수정안(단일):
  - `JumpReLUAutoencoder.encode()`를 오버라이드해 `_JumpReLUFn.apply(hidden_pre, threshold, bandwidth)`를 반환하도록 변경.
  - 대안으로 `compute_latent_features`에서 `ae(flat)`의 두 번째 반환값(latents)을 사용.
- 검증케이스:
  - 동일 입력에서 `encode(x)`와 `forward(x)`의 latent가 일치해야 함.
  - `threshold` 상향 시 활성 latent 수(`mean_l0`) 감소 단조성 확인.
  - SAELens `JumpReLUTrainingSAE.encode_with_hidden_pre` 출력과 방향성 일치 확인.
- 관찰 증거:
  - JumpReLU class는 `activation=Identity`로 초기화되고(`jumprelu.py:101-106`), downstream은 `ae.encode` 경로를 사용(`downstream.py:49-50`).
  - `chemberta-repro` 런타임 재현:
    - `encode_out` = `[[0.2,0.7,0.1,1.0],[1.0,0.1,0.9,2.0]]`
    - `forward_lat` = `[[0.0,0.7,0.0,1.0],[1.0,0.0,0.9,2.0]]`
    - `allclose=False`

---

## P1

### P1-1
- 심각도: `P1`
- 증상: sweep trial의 `seed`가 실질적으로 학습 루프에 반영되지 않아, 기록된 seed와 실제 난수 상태가 불일치.
- 재현 조건:
  - `python scripts/run.py sae --sweep --arch jumprelu` 또는 `--arch both` 실행 후 registry의 `seed`와 실행 결과 재현성 비교.
- 파일/라인:
  - `src/chem_sae/train/sweep.py:49`
  - `src/chem_sae/train/sweep.py:61`
  - `src/chem_sae/train/sae_training.py:471-503`
  - `src/chem_sae/train/sae_training.py:267-271`
- SAELens 기준:
  - SAELens는 학습 config 기반 coefficient/step을 명시적으로 적용하며 실행 컨텍스트와 학습 상태 일관성을 강하게 유지.
- 영향:
  - 재현성 저하, registry 상의 `seed` 필드 신뢰도 하락.
- 수정안(단일):
  - `run_architecture_trial` 시작 시 `set_seed(trial_cfg.seed)` 호출하고, `ActivationChunkDataset` seed도 `trial_cfg.seed`를 사용.
- 검증케이스:
  - 동일 `trial_cfg.seed` 두 번 실행 시 `metrics.json` 핵심 지표 동일.
  - 서로 다른 seed에서 지표 분산 발생 확인.
- 관찰 증거:
  - trial seed는 sweep에서 생성되지만(`sweep.py:49,61`), 실제 학습 함수 경로에서 해당 seed를 적용하는 호출이 없음.

### P1-2
- 심각도: `P1`
- 증상: resume 시 `patience_counter` 및 RNG 상태를 복원하지 않아 중단-재개 결과가 연속 실행과 달라질 수 있음.
- 재현 조건:
  - 학습 중 checkpoint 생성 후 프로세스 중단, 동일 run에서 재개.
- 파일/라인:
  - `src/chem_sae/train/sae_training.py:281-299`
  - `src/chem_sae/train/sae_training.py:337-391`
  - `src/chem_sae/train/sae_training_probe.py:212-230`
  - `src/chem_sae/train/sae_training_probe.py:285-327`
- SAELens 기준:
  - SAELens는 step/coefficient 기반 학습 상태 일관성을 전제로 함.
- 영향:
  - early stopping 재현성/종료 epoch 불일치.
- 수정안(단일):
  - checkpoint에 `patience_counter`와 `python/numpy/torch/cuda RNG state` 저장/복원 추가.
- 검증케이스:
  - epoch N에서 강제 중단 후 resume한 결과가 uninterrupted run과 동일 종료 epoch/`best_nmse`를 보장.
- 관찰 증거:
  - checkpoint 로드시 `start_epoch/global_step/best_nmse`만 복원하고 `patience_counter`는 재초기화됨.

### P1-3
- 심각도: `P1`
- 증상: 필수 회귀 테스트 부재.
- 재현 조건:
  - `python3 -m pytest -q` 실행.
- 파일/라인:
  - `tests/__init__.py:1` (실질 테스트 없음)
- SAELens 기준:
  - SAELens 수준의 손실/훈련 contract 비교 시 최소 단위테스트 필요.
- 영향:
  - 손실/gradient/체크포인트 회귀가 조기에 탐지되지 않음.
- 수정안(단일):
  - `tests/`에 JumpReLU loss/gradient, resume 무결성, early-stopping 계약, layers_spec 파서, probe output 계약 테스트 추가.
- 검증케이스:
  - 하단 “테스트 명세” 항목 참조.
- 관찰 증거:
  - 현 환경 실행 결과: `/usr/bin/python3: No module named pytest` (테스트 러너/테스트셋 부재 상태).

---

## P2

### P2-1
- 심각도: `P2`
- 증상: downstream CSV에 `sae_type="TopK"`가 하드코딩되어 JumpReLU 결과가 잘못 기록됨.
- 재현 조건:
  - `arch=jumprelu` 실행 후 `reports/downstream_records.csv` 확인.
- 파일/라인:
  - `src/chem_sae/eval/downstream.py:280`
- SAELens 기준:
  - 아키텍처별 metric tagging 일관성 필요.
- 영향:
  - 후처리/리더보드/분석 리포트 오판.
- 수정안(단일):
  - `extra_fields["arch"]` 또는 명시 인자 기반으로 `sae_type` 동적 설정.
- 검증케이스:
  - JumpReLU 실행 후 CSV 행에서 `sae_type=JumpReLU` 확인.
- 관찰 증거:
  - 결과 row 생성 코드에 `sae_type` 문자열이 상수 `"TopK"`로 고정됨.

### P2-2
- 심각도: `P2`
- 증상: probe는 `--resume` 옵션 없이 기존 checkpoint가 있으면 자동 재개되어 의도치 않은 continuation 가능.
- 재현 조건:
  - 동일 `run_id`에서 probe를 중단 후 같은 명령 재실행.
- 파일/라인:
  - `src/chem_sae/train/sae_training_probe.py:220-223`
  - `scripts/run_sae_probe.py` (resume 제어 인자 부재)
- SAELens 기준:
  - 실험 재시작/재개가 명시적으로 구분되어야 함.
- 영향:
  - 동일 run_id 재실행 시 실험 프로토콜 혼선.
- 수정안(단일):
  - `scripts/run_sae_probe.py`에 `--resume` 플래그 추가, 기본은 fresh start.
- 검증케이스:
  - checkpoint 존재 run에서 `--resume` 없으면 epoch1부터 재시작, `--resume`이면 재개 확인.
- 관찰 증거:
  - `latest_checkpoint`가 존재하면 조건 없이 로드하는 경로만 존재.

### P2-3
- 심각도: `P2`
- 증상: `cfg.local_only`가 MolNet dataset 로딩에 반영되지 않음.
- 재현 조건:
  - 네트워크 차단 환경에서 downstream/final-hidden/intervention 실행.
- 파일/라인:
  - `src/chem_sae/eval/final_hidden.py:67`
  - `src/chem_sae/eval/intervention.py:74`
  - `src/chem_sae/eval/downstream.py:120,217` (`local_only` 인자 미전달)
- SAELens 기준:
  - 실행 config의 IO 제약(local/offline) 일관 반영 필요.
- 영향:
  - 오프라인 환경 재현 실패, 네트워크 의존성 증가.
- 수정안(단일):
  - `load_molnet_dataset(..., local_only=cfg.local_only)`로 통일.
- 검증케이스:
  - 네트워크 차단 환경에서 local cache만으로 평가 루프 동작 확인.
- 관찰 증거:
  - 일부 경로는 `local_only=False`를 하드코딩하고, 일부 경로는 인자 자체를 전달하지 않음.

---

## P3

### P3-1
- 심각도: `P3`
- 증상: `_StepSTE`는 현재 코드 경로에서 사용되지 않고, SAELens `Step`과 gradient 정의도 다름.
- 재현 조건:
  - 코드 검색: `_StepSTE` 호출 지점 없음.
- 파일/라인:
  - `src/chem_sae/vendor/jumprelu.py:23-53`
  - `sae_lens/saes/jumprelu_sae.py:23-51`
- SAELens 기준:
  - Step 모드 gradient는 threshold 중심(`x_grad=None`).
- 영향:
  - 향후 Step 모드 도입 시 예기치 않은 gradient 동작 가능.
- 수정안(단일):
  - `_StepSTE` 제거 또는 SAELens와 동일 gradient로 정렬 후 명시적으로 사용.
- 검증케이스:
  - Step 모드 단위테스트에서 threshold gradient 부호/스케일 일치 확인.
- 관찰 증거:
  - `rg "_StepSTE|Step.apply"` 기준 프로젝트 학습 경로에서 `_StepSTE` 참조 없음.

### P3-2
- 심각도: `P3`
- 증상: `torch.load(..., weights_only=False)` 경고 다수 발생.
- 재현 조건:
  - probe 실행 로그 확인.
- 파일/라인:
  - `src/chem_sae/utils/hf.py:17`
  - `src/chem_sae/data/datasets.py:79`
  - `src/chem_sae/train/sae_training.py:274,291,393`
  - `src/chem_sae/train/sae_training_probe.py:199,222,329`
- SAELens 기준:
  - 저장/로드 경로 안정성 및 보안성 고려 필요.
- 영향:
  - 신뢰되지 않은 체크포인트 로딩 시 보안 리스크.
- 수정안(단일):
  - 체크포인트/weights 로딩 경로에 `weights_only=True` 적용 가능한 곳 우선 적용.
- 검증케이스:
  - launch log에서 FutureWarning 소거 확인.
- 관찰 증거:
  - `probe_nlat1536_l1to5_l0_13e5_none_e30_noes_20260304_140647.probe.launch.log` 상단 FutureWarning 반복 출력.

---

## 코드 이슈 vs 하이퍼파라미터 이슈 분리

- 코드 이슈:
  - P0-1, P1-1, P1-2, P2-1, P2-2, P2-3, P3-1, P3-2
- 하이퍼파라미터 이슈(코드 결함 아님):
  - `base_l0` 상승 시 `dead_ratio` 증가 및 `nmse` 악화는 로그에서 일관 관찰됨.
  - 근거:
    - `artifacts/runs/sae/probe_nlat1536_l0_l0_1e4_none_e30_noes_20260303_211715/probe/metrics.json`
    - `artifacts/runs/sae/probe_nlat1536_l0_l0_12e5_none_e30_noes_20260303_214528/probe/metrics.json`
    - `artifacts/runs/sae/probe_nlat1536_l0_l0_13e5_none_e30_noes_20260303_215236/probe/metrics.json`
    - `artifacts/runs/sae/probe_nlat1536_l0_l0_15e5_none_e30_noes_20260303_213455/probe/metrics.json`

## 테스트 명세(실행 가능 수준)

1. JumpReLU 손실/표현 검증
- 입력/threshold를 고정했을 때 `JumpReLUAutoencoder.encode` 결과가 `forward` latent와 일치.
- `l0_coefficient` 증가 시 동일 batch에서 `mean_l0` 비증가 경향 확인.

2. Resume 무결성
- 중단 후 재개 시 `start_epoch`, `global_step`, `best_nmse`, `patience_counter` 연속성 확인.

3. Early stopping 계약
- `--disable-early-stopping` 사용 시 `early_stopping_patience = epochs + 1` 강제 확인.

4. Layer spec 파싱 계약
- `all`, `0,1,2`, 빈 문자열, 음수, 범위초과 입력 각각 기대 예외/결과 확인.

5. Probe 결과 파일 완결성
- 정상 종료 시 `probe/metrics.json` + `probe/schedule_trace.json` 생성 보장.
- 중단(run kill) 시 partial 상태를 명시적으로 식별 가능한 마커 파일 생성(개선안 반영 시).

6. 로그 계약
- epoch 로그에서 `nmse/l0/dead/l0_coef_eff` 키 존재를 정규식으로 검사.
