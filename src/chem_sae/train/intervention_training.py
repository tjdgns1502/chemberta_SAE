from __future__ import annotations

import itertools

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import SaeInterventionConfig
from chem_sae.data import prepare_mlm_loader
from chem_sae.eval import evaluate_intervention
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.utils import append_csv_row, save_checkpoint, set_seed
from chem_sae.vendor import Autoencoder, TopK, autoencoder_loss


def generate_intervention_patterns(num_layers: int = 6):
    patterns = list(itertools.product([False, True], repeat=num_layers))
    return [list(pattern) for pattern in patterns]


def pattern_to_string(pattern: list[bool]) -> str:
    return "".join(["1" if bit else "0" for bit in pattern])


def get_intervened_layers(pattern: list[bool]) -> str:
    layers = [str(i) for i, intervene in enumerate(pattern) if intervene]
    return ",".join(layers) if layers else ""


@torch.no_grad()
def extract_activations_with_intervention(
    model: RobertaForMaskedLM,
    loader: DataLoader,
    target_layer: int,
    sae_dict: dict[int, Autoencoder],
    pattern: list[bool],
    device: torch.device,
):
    model.eval()
    for sae in sae_dict.values():
        sae.eval()

    current_pattern = [pattern[i] if i < target_layer else False for i in range(len(pattern))]
    buffered = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        _, attn_outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sae_dict=sae_dict,
            intervention_pattern=current_pattern,
            return_attn_outputs=True,
            attn_output_layers={target_layer},
        )

        attn = attn_outputs[target_layer]
        flat = attn[attention_mask.bool()].detach().cpu().to(torch.float16)
        buffered.append(flat)

    return torch.cat(buffered, dim=0)


def train_sae(
    activations: torch.Tensor,
    cfg: SaeInterventionConfig,
    device: torch.device,
) -> Autoencoder:
    d_model = activations.shape[-1]
    num_samples = activations.shape[0]

    train_size = int(num_samples * (1 - cfg.val_fraction))
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_acts = activations[train_indices].to(torch.float32)
    val_acts = activations[val_indices].to(torch.float32)

    train_loader = DataLoader(train_acts, batch_size=cfg.sae_batch_size, shuffle=True)
    val_loader = DataLoader(val_acts, batch_size=cfg.sae_batch_size, shuffle=False)

    ae = Autoencoder(
        n_latents=cfg.n_latents,
        n_inputs=d_model,
        activation=TopK(cfg.topk),
        normalize=True,
    ).to(device)

    optimizer = torch.optim.AdamW(ae.parameters(), lr=cfg.sae_lr)
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(cfg.sae_epochs):
        ae.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            _, latents, recons = ae(batch)
            loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, latents, recons = ae(batch)
                loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
                val_loss += loss.item()

        train_loss_avg = train_loss / max(1, len(train_loader))
        val_loss_avg = val_loss / max(1, len(val_loader))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state_dict = {k: v.clone() for k, v in ae.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"      epoch {epoch + 1}/{cfg.sae_epochs} train_loss={train_loss_avg:.4f} "
            f"val_loss={val_loss_avg:.4f} (best={best_val_loss:.4f})"
        )

        if patience_counter >= cfg.early_stopping_patience:
            print(f"      Early stopping at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        ae.load_state_dict(best_state_dict)
    return ae


def train_sae_with_intervention(
    cfg: SaeInterventionConfig,
    pattern: list[bool],
    pattern_id: int,
    device: torch.device,
) -> dict[int, Autoencoder]:
    model, _ = build_mlm_model(cfg, device)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    mlm_loader = prepare_mlm_loader(cfg, tokenizer)

    sae_dict: dict[int, Autoencoder] = {}
    for layer in cfg.layers:
        intervene_this_layer = pattern[layer]

        if intervene_this_layer:
            print(f"  Training SAE for layer {layer}...")
            activations = extract_activations_with_intervention(
                model, mlm_loader, layer, sae_dict, pattern, device
            )
            sae = train_sae(activations, cfg, device)
            sae_dict[layer] = sae

            save_path = cfg.ckpt_dir / f"pattern_{pattern_id}" / f"layer_{layer}.pt"
            save_checkpoint(save_path, {"model": sae.state_dict()})
        else:
            print(f"  Skipping SAE training for layer {layer} (no intervention)")

    return sae_dict


def run_intervention_experiment(pattern_ids: list[int], gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cfg = SaeInterventionConfig()
    cfg.ensure_dirs()
    patterns = generate_intervention_patterns(num_layers=len(cfg.layers))

    for pattern_id in pattern_ids:
        pattern = patterns[pattern_id]
        pattern_str = pattern_to_string(pattern)
        num_intervened = sum(pattern)
        intervened_layers = get_intervened_layers(pattern)

        print(f"\n{'=' * 70}")
        print(f"GPU {gpu_id}: Pattern {pattern_id}/{len(patterns) - 1}")
        print(f"Binary: {pattern_str}")
        print(
            f"Intervened layers: {num_intervened}/{len(cfg.layers)} "
            f"({intervened_layers if intervened_layers else 'none'})"
        )
        print(f"{'=' * 70}")

        set_seed(cfg.seed)
        print("Training SAEs...")
        sae_dict = train_sae_with_intervention(cfg, pattern, pattern_id, device)

        print("Evaluating downstream tasks...")
        results = evaluate_intervention(cfg, sae_dict, pattern, pattern_id, device)

        for row in results:
            row["pattern_binary"] = pattern_str
            row["num_intervened"] = num_intervened
            row["intervened_layers"] = intervened_layers
            row["gpu_id"] = gpu_id
            append_csv_row(cfg.log_path, row)

        print(f"Pattern {pattern_id} complete!")

