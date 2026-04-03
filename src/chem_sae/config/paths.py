from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "chem_sae"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
SAE_RUN_ROOT = RUNS_ROOT / "sae"
INTERVENTION_RUN_ROOT = RUNS_ROOT / "sae_intervention"

# Legacy root kept for compatibility with older scripts.
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"

# Legacy project roots used during migration.
CAPSTONE_ROOT = PROJECT_ROOT.parent
LEGACY_CHEMBERTA_ROOT = CAPSTONE_ROOT / "chemberta_repro_final"
