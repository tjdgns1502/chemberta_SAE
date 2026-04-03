# CODE REVIEW MASTER

## 1) 검수 메타
- 검수 대상 루트: `/home/yoo122333/capstone/chemberta_SAE`
- 기준선 루트: `/home/yoo122333/capstone/SAELens`
- 기준선 커밋: `e294a1c6`
- 검수 일시: 2026-03-04
- 검수 방식: 정적 코드 리뷰 + 산출 로그/메트릭 교차검증(코드 수정 없음)

## 2) 범위/제외
- 포함:
  - `src/chem_sae/*`
  - `scripts/*`
  - `configs/*`
  - `README.md`, `docs/*`
  - `references/legacy_source/*`
- 제외:
  - `artifacts/runs/*` 대용량 산출물 자체 품질평가
  - 단, 코드 동작 증거로 로그/`metrics.json`/`schedule_trace.json`은 사용

## 3) 단계별 진행 현황
| 단계 | 내용 | 상태 | 산출물 |
|---|---|---|---|
| 1 | 인벤토리/의존성 맵 | 완료 | 본 문서 4~5절 |
| 2 | SAELens Trace Matrix | 완료 | `SAELENS_TRACE_MATRIX.md` |
| 3 | 정적 정확성 검수 | 완료 | `CODE_REVIEW_FINDINGS.md` |
| 4 | SAELens 의미 보존 검수 | 완료 | `SAELENS_TRACE_MATRIX.md` + Findings |
| 5 | 재현성/프로토콜 검수 | 완료 | Findings P1/P2 + 본 문서 6절 |
| 6 | 테스트 전략/케이스 명세 | 완료 | Findings + `NEXT_CHAT_HANDOFF.md` |
| 7 | 개선안 설계(코드 수정 없음) | 완료 | Findings의 수정안/검증케이스 |
| 8 | 최종 심각도 이슈리스트 | 완료 | `CODE_REVIEW_FINDINGS.md` |
| 9 | 인수인계 문서 세트 | 완료 | 본 문서 + 3개 문서 |

## 4) 검수 대상 목록(인벤토리)
| 분류 | 파일/모듈 |
|---|---|
| CLI | `scripts/run.py`, `scripts/run_sae_probe.py`, `scripts/run_*.py` 래퍼 |
| Config | `src/chem_sae/config/{experiment,intervention,paths,runtime}.py` |
| Data | `src/chem_sae/data/{datasets,loaders}.py` |
| Modeling | `src/chem_sae/modeling/{roberta_mlm,build}.py` |
| Train | `src/chem_sae/train/{sae_training,sae_training_probe,sweep,intervention_training,quality_metrics}.py` |
| Eval | `src/chem_sae/eval/{downstream,final_hidden,intervention}.py` |
| Vendor | `src/chem_sae/vendor/{sae_core,jumprelu,molnet_loader}.py` |
| Utils | `src/chem_sae/utils/{checkpoint,io,randomness,hf}.py` |
| Legacy 참고 | `references/legacy_source/{sae_experiment_original,sae_intervention_experiment_original}.py` |
| 문서 | `README.md`, `docs/CHAT_CONTEXT_TRANSFER.md` |

## 5) 호출 흐름 다이어그램(텍스트)
```text
scripts/run.py
  ├─ sae
  │   ├─ run_all()                    [src/chem_sae/train/sae_training.py]
  │   │   ├─ build_mlm_model()
  │   │   ├─ resolve_layers_from_model()
  │   │   ├─ prepare_activation_cache()
  │   │   │   └─ extract_attn_activations() -> artifacts/runs/sae/{run_id}/acts/layer_*/chunk_*.pt
  │   │   └─ run_architecture_trial()
  │   │       ├─ train_sae_for_layer()
  │   │       ├─ evaluate_downstream()
  │   │       └─ save checkpoints/metrics/plots/csv
  │   └─ run_sweep()                  [src/chem_sae/train/sweep.py]
  │       └─ stage1 -> stage2 -> stage3 winner + reference eval
  ├─ baseline
  │   └─ evaluate_baseline_frozen()
  ├─ final-hidden
  │   └─ evaluate_final_hidden_state()
  └─ intervention
      └─ run_intervention_experiment()

scripts/run_sae_probe.py
  └─ run_probe()                      [src/chem_sae/train/sae_training_probe.py]
      ├─ prepare_activation_cache()
      ├─ train_probe_for_layer() x selected layers
      └─ write probe/schedule_trace.json + probe/metrics.json (정상 종료 시)
```

## 6) 재현성/실험 프로토콜 점검 요약
- `run_id` 격리 구조는 `SaeExperimentConfig.ensure_dirs()` 기준으로 정상 구성됨.
- 중단된 probe run(`l1to5` 계열)은 `probe/metrics.json`, `probe/schedule_trace.json`이 생성되지 않음(정상 종료 전 중단).
- 과희소화 현상은 로그상 `l0_coefficient` 증가와 함께 `dead_ratio` 상승, `nmse` 악화로 일관되게 관찰되어 하이퍼파라미터 영향이 큼.
  - 예: `probe_nlat1536_l1to5_l0_2e4_none_e30_noes_20260304_143654.probe.launch.log`의 layer1 epoch1~19 구간에서 `dead` 지속 상승.

## 7) SAELens 대비 총평
- `jumprelu` 핵심 forward/backward 골격은 SAELens와 유사하나, 손실 정의와 학습/평가 경로에서 의미 차이가 존재.
- 특히 JumpReLU downstream feature 추출 경로는 현재 코드가 게이팅 전 pre-activation을 사용할 수 있어 의미 보존 실패(P0).
- 상세 매핑은 `SAELENS_TRACE_MATRIX.md`, 심각도별 결과는 `CODE_REVIEW_FINDINGS.md` 참조.

## 8) 실행/검증 환경 메모
- 검증 기본 환경: `micromamba env: chemberta-repro` (`/home/yoo122333/micromamba/envs/chemberta-repro/bin/python`)
- `python -m compileall src scripts`: 성공
- `python -m pytest -q`: 실패 (`No module named pytest`)
- 런타임 검증:
  - `torch 2.5.1`, `cuda True`
  - JumpReLU `encode` vs `forward` latent 불일치 재현(`allclose=False`)로 P0-1 실증
