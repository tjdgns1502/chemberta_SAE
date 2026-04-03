from dataclasses import dataclass

from .paths import INTERVENTION_RUN_ROOT, LOGS_ROOT, RUNS_ROOT, SAE_RUN_ROOT


@dataclass(frozen=True)
class RuntimeDirs:
    runs_dir: str = str(RUNS_ROOT)
    logs_dir: str = str(LOGS_ROOT)
    # Backward-compatible default checkpoint path for SAE.
    checkpoints_dir: str = str(SAE_RUN_ROOT / "checkpoints")
    intervention_checkpoints_dir: str = str(INTERVENTION_RUN_ROOT / "checkpoints")
