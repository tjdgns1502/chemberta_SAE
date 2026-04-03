# NEXT CHAT HANDOFF

## 현재 상태 요약
- 전수 검수 문서 작성 완료:
  - `docs/CODE_REVIEW_MASTER.md`
  - `docs/CODE_REVIEW_FINDINGS.md`
  - `docs/SAELENS_TRACE_MATRIX.md`
  - `docs/NEXT_CHAT_HANDOFF.md`
- 코드 수정 진행:
  - `P0-1` 완료 (JumpReLU `encode`를 gate 적용 latent 경로로 정렬)
  - 회귀 테스트 추가: `tests/test_jumprelu_encode.py`
  - `P1-1` 완료 (trial seed를 학습 루프/activation chunk seed에 반영)
  - `P1-2` 완료 (resume 시 `patience_counter` + RNG state 저장/복원)
  - `P2-1` 완료 (`sae_type`를 arch 기준 동적 기록)
  - `P2-2` 완료 (`run_sae_probe.py`에 `--resume` 명시 계약 추가)
  - `P2-3` 완료 (MolNet 로더에 `local_only` 전 경로 반영)
  - `P3-1` 완료 (`_StepSTE` dead code 제거)
  - `P3-2` 완료 (`torch.load` 경로별 `weights_only` 명시)
  - SAELens 원본 정렬 추가 반영:
    - JumpReLU `step/tanh` sparsity loss mode + `W_dec_norm` 스케일 반영
    - Step gradient 계약(`x_grad=None`) 반영
    - `threshold<->log_threshold` 저장/로드 변환 훅 반영
    - `fold_W_dec_norm` 시 threshold 보정 반영
    - dead-neuron mask + l0 warmup(step) 학습 루프 연결
    - `apply_b_dec_to_input` 플래그 추가로 SAE core 입력 중심화 경로 정렬
    - `BatchTopK` forward 수식을 SAELens와 동일하게 정렬(위치만 확장 모듈 유지)
  - 테스트 추가:
    - `tests/test_randomness_state.py`
    - `tests/test_downstream_sae_type.py`
    - `tests/test_layers_spec_contract.py`
    - `tests/test_probe_cli_resume_contract.py`
    - `tests/test_probe_output_contract.py`
    - `tests/test_train_resume_contract.py`
    - `tests/test_jumprelu_sae_lens_contract.py`
    - `tests/test_jumprelu_dead_mask_contract.py`
    - `tests/test_l0_warmup_contract.py`
    - `tests/test_apply_b_dec_contract.py`
    - `tests/test_batchtopk_sae_lens_contract.py`
  - 테스트 실행:
    - `PYTHONPATH=src /home/yoo122333/micromamba/envs/chemberta-repro/bin/python -m unittest discover -s tests -v` 통과 (`Ran 25 tests`, `OK`)

## 다음 채팅에서 바로 할 일 (우선순위 고정)
1. 코드 이슈는 정리 완료 상태.
2. 다음은 실험 단계:
   - probe 재실행(`--resume` 계약 준수) 후 metrics/schedule 검증.
   - l0 튜닝 재개.
3. SAELens 추가 정렬 후보:
   - `T10` 제거 여부 결정(현재는 위치만 의도적 차이).

## 고정된 의사결정
- SAELens는 구조 복제 대상이 아니라 기능 기준선.
- `동일`보다 `의미 동등성` 우선.
- 하이퍼파라미터 문제와 코드 결함은 분리 판정.

## 구현 순서 제안 (커밋 단위)
1. `fix(jumprelu-eval): use gated latents for downstream` (완료)
2. `fix(repro): honor trial seed + resume state parity` (완료)
3. `fix(cli-logging): probe resume contract + sae_type tags + local_only` (완료)
4. `fix(cleanup): remove dead StepSTE + explicit torch.load weights_only` (완료)
5. `test(core-contracts): jumprelu/resume/earlystop/layerspec/probe/log` (완료)

## 시작용 지시문 (다음 채팅 첫 메시지)
```text
Continue from docs/NEXT_CHAT_HANDOFF.md.
Run probe and sweep experiments with updated contracts, then compare metrics by layer/l0.
```

## 빠른 점검 커맨드
```bash
cd /home/yoo122333/capstone/chemberta_SAE
git diff -- docs/CODE_REVIEW_MASTER.md docs/CODE_REVIEW_FINDINGS.md docs/SAELENS_TRACE_MATRIX.md docs/NEXT_CHAT_HANDOFF.md
```

## 리스크 메모
- 현재 환경에서 `python3 -m pytest`는 `pytest` 미설치로 실패.
- 런타임 동적 검증은 환경의 `torch` 설치 상태 확인 후 진행 필요.
