"""Run baseline evaluation only (no SAE required)."""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "bert-loves-chemistry"))
sys.path.insert(0, project_root)

from sae_experiment import (
    SaeExperimentConfig,
    build_mlm_model,
    evaluate_baseline_frozen,
    set_seed,
)
import torch
from transformers import RobertaTokenizerFast


if __name__ == "__main__":
    cfg = SaeExperimentConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)
    
    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION (Frozen backbone)")
    print(f"{'='*60}")
    
    for layer in cfg.layers:
        print(f"\n--- Layer {layer} ---")
        evaluate_baseline_frozen(cfg, model, tokenizer, layer, device)
    
    print("\n✅ Baseline evaluation complete!")
