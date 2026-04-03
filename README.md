# chemberta_SAE

ChemBERTa + SAE 실험 코드를 `단일 루트`에서 관리하기 위한 신규 프로젝트 골격입니다.

## 디렉토리 구조

```text
chemberta_SAE/
  src/chem_sae/
    config/
    data/
    eval/
    modeling/
    train/
    vendor/
  scripts/
    run.py
  configs/
  artifacts/
    runs/
    logs/
  tests/
```

## 원칙

- 외부 git clone 디렉토리 전체 import 금지
- 필요한 코드만 `src/chem_sae/vendor` 또는 내부 모듈로 내재화
- 산출물은 항상 `artifacts/` 하위로 저장
- 실행 진입점은 `scripts/run.py` 단일 CLI를 기준으로 관리

## 현재 상태

- 모듈 분리 진행 중 (`config`, `data`, `modeling`, `train`, `eval`)
- Downstream 데이터: Hugging Face Hub(`scikit-fingerprints/MoleculeNet_*`) 사용
- MLM 데이터: `data/100k_rndm_zinc_drugs_clean.txt` 사용

## 실행 예시

- SAE 학습(대표 명령): `python scripts/run.py sae --arch both --layers all`
- SAE 스윕(8 -> 20 -> 25): `python scripts/run.py sae --sweep --arch both --layers all`
- 특정 run 이어서 재개: `python scripts/run.py sae --run-id <RUN_ID> --resume`
- JumpReLU probe(대시보드 3 epoch 갱신): `python scripts/run_sae_probe.py --layers 1 --base-l0 1e-4 --dashboard-every-epochs 3`
- SAE/W&B 로깅: `python scripts/run.py sae --arch jumprelu --layers 1 --log-to-wandb --wandb-project sae_lens_training --wandb-run-name chemberta-sae`
- Probe/W&B 로깅: `python scripts/run_sae_probe.py --layers 1 --base-l0 1e-3 --log-to-wandb --wandb-project sae_lens_training --wandb-run-name chemberta-probe`
- Baseline 평가: `python scripts/run.py baseline --layers all`
- Final hidden 평가: `python scripts/run.py final-hidden --layers all`
- Intervention 평가: `python scripts/run.py intervention --pattern_ids 0,1 --gpu_id 0`

기존 `scripts/run_*.py` 파일은 하위 호환 래퍼이며 내부적으로 모두 `scripts/run.py`를 호출합니다.

## W&B 가이드 (SAELens logger 계약 반영)

- 실시간(online) 로그 시작:
  - `wandb login` (최초 1회)
  - `python scripts/run.py sae --arch jumprelu --layers 1 --log-to-wandb --wandb-project sae_lens_training --wandb-run-name chemberta-sae`
- Probe 실시간 로그:
  - `python scripts/run_sae_probe.py --layers 1 --base-l0 1e-3 --log-to-wandb --wandb-project sae_lens_training --wandb-run-name chemberta-probe`
- 오프라인 수집 후 업로드:
  - `WANDB_MODE=offline python scripts/run_sae_probe.py ... --log-to-wandb`
  - `wandb sync wandb/offline-run-*`
- 재개(같은 W&B run id 유지):
  - `python scripts/run.py sae --run-id <RUN_ID> --resume --log-to-wandb --wandb-id <WANDB_ID>`
  - probe도 동일하게 `--run-id`, `--resume`, `--wandb-id` 조합 지원

지원 플래그(두 CLI 공통):
- `--log-to-wandb`, `--disable-wandb`
- `--wandb-project`, `--wandb-entity`, `--wandb-id`, `--wandb-run-name`
- `--wandb-log-frequency`, `--eval-every-n-wandb-logs`
- `--disable-wandb-log-weights`
- `--log-optimizer-state-to-wandb`
- `--log-activations-store-to-wandb`

참고:
- online 모드에서만 W&B 웹 대시보드가 실시간 갱신됩니다.
- 실행 완료 시 콘솔에 `wandb_run_url=...`가 출력됩니다(online 기준).
- run 메타는 `artifacts/runs/sae/<run_id>/reports/wandb_run.json`에 저장됩니다.

## run_id 격리 구조

실험 산출물은 항상 `run_id` 하위로 저장됩니다.

```text
artifacts/runs/sae/{run_id}/
  acts/
  checkpoints/
  sweep/
  final/
  reports/
  plots/
  run_meta.json
```

`--resume`는 반드시 `--run-id`와 함께 사용해야 하며, 해당 run의 checkpoint만 복원합니다.

Probe 실행 시 대시보드는 `artifacts/runs/sae/{run_id}/probe/dashboards/`에 생성됩니다.
고정 미러 대시보드는 `artifacts/runs/sae/probe_dashboard_live/`에서 같은 파일을 계속 갱신합니다.
