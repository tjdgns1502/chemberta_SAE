# SAELENS TRACE MATRIX

기준선: `SAELens@e294a1c6`

상태 태그:
- `동일`: 기능 의미가 사실상 동일
- `의도적 차이`: 구조/공식이 다르지만 문서화 가능한 설계 차이
- `불명확 차이`: 의도 근거 부족 또는 동작 의미 훼손 가능

| ID | chemberta_SAE 구현 | SAELens 대응 구현 | 상태 | 근거/리스크 |
|---|---|---|---|---|
| T01 | `src/chem_sae/vendor/jumprelu.py:19-21` `rectangle` | `sae_lens/saes/jumprelu_sae.py:19-21` `rectangle` | 동일 | 구현/조건식 동일 |
| T02 | `src/chem_sae/vendor/jumprelu.py:53-84` `_JumpReLUFn` | `sae_lens/saes/jumprelu_sae.py:54-85` `JumpReLU` | 동일 | forward/backward 식과 threshold gradient 형태 일치 |
| T03 | `src/chem_sae/vendor/jumprelu.py:116-118` `log_threshold -> exp` | `sae_lens/saes/jumprelu_sae.py:241-264` | 동일 | threshold parameterization 동일 |
| T04 | `src/chem_sae/vendor/jumprelu.py:145-153` forward에서 hard gate 적용 | `sae_lens/saes/jumprelu_sae.py:265-273` | 동일 | training forward에서 hard gating feature 사용 |
| T05 | `src/chem_sae/vendor/jumprelu.py:166-182` `step/tanh` 분기 + `W_dec_norm` 반영 | `sae_lens/saes/jumprelu_sae.py:287-301` (`step`/`tanh`) | 동일 | SAELens와 동일한 sparsity loss mode 분기 구조 반영 |
| T06 | `src/chem_sae/vendor/jumprelu.py:23-50` `_StepFn` | `sae_lens/saes/jumprelu_sae.py:23-51` `Step` | 동일 | x gradient 차단(`None`) + threshold gradient 식 동일 |
| T07 | `src/chem_sae/vendor/sae_core.py:73-97` normalize in/out + `apply_b_dec_to_input` | `sae_lens/saes/sae.py:329-349,465-475,908-916` | 동일 | runtime layer-norm in/out과 `b_dec` 적용 플래그를 반영해 수식 의미 동등 |
| T08 | `src/chem_sae/vendor/sae_core.py:233-252` `autoencoder_loss`(MSE+L1) | `sae_lens/saes/standard_sae.py` `mse_loss + l1_loss` | 동일 | 손실 스케일을 SAELens 계열(MSE+L1)로 정렬 |
| T09 | `src/chem_sae/vendor/jumprelu.py:154-196` `jumprelu_loss` + `calculate_pre_act_loss` + `src/chem_sae/train/sae_training.py`, `sae_training_probe.py`의 dead-mask/warmup 연결 | `sae_lens/saes/jumprelu_sae.py:276-313` + `sae_lens/training/sae_trainer.py` coefficient/dead-neuron 흐름 | 동일 | dead-neuron mask와 l0 warmup(steps) 흐름을 학습 루프에 반영 |
| T10 | `src/chem_sae/vendor/batchtopk_ext.py` `BatchTopK` (core 외부 확장으로 분리) | `sae_lens/saes/batchtopk_sae.py:13-33` `BatchTopK` | 의도적 차이 | forward 수식은 SAELens와 동일하게 정렬, 파일 위치만 확장 모듈로 분리 |
| T11 | `src/chem_sae/eval/downstream.py:61` `ae.encode(flat)` 사용 + `src/chem_sae/vendor/jumprelu.py:146-153` JumpReLU `encode` 오버라이드 | SAELens JumpReLU 학습 feature는 `JumpReLU.apply(...)` 결과 | 동일 | 2026-03-04 수정으로 `encode` 경로가 hard-gated latent를 반환하도록 정렬 |
| T12 | `src/chem_sae/vendor/jumprelu.py:129-140` state_dict threshold/log 변환 훅 구현 | `sae_lens/saes/jumprelu_sae.py:340-353` | 동일 | 저장 시 `threshold`, 로드 시 `log_threshold` 변환 경로 반영 |
| T13 | `src/chem_sae/vendor/jumprelu.py:120-128` `fold_W_dec_norm` threshold 보정 | `sae_lens/saes/jumprelu_sae.py:323-339` | 동일 | decoder norm folding 시 threshold 동치 보정 반영 |
| T14 | `src/chem_sae/vendor/sae_core.py:136-139` plain `state_dict()` 저장 | SAELens도 내부 weight state를 별도 가공 없이 보관 후 arch-specific hook 적용 | 동일 | activation 메타 주입 제거로 저장 경로 정렬 |

## 판정 요약
- `동일`: T01~T09, T11~T14
- `의도적 차이`: T10
- `불명확 차이`: 없음

`불명확 차이` 항목은 모두 해소됨.
