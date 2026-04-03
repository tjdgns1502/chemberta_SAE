from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig

if TYPE_CHECKING:
    from chem_sae.vendor import Autoencoder


def get_activation(name: str):
    if name in ("gelu", "gelu_new", "gelu_fast"):
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(name)


def _coerce_feature_values(
    latents: torch.Tensor,
    *,
    feature_indices: torch.Tensor,
    feature_values: float | list[float] | torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(feature_indices.numel(), device=latents.device, dtype=latents.dtype)
    if feature_values is None:
        raise ValueError(f"feature_values are required for mode='{mode}'")

    values = torch.as_tensor(feature_values, device=latents.device, dtype=latents.dtype).flatten()
    if values.numel() == 1:
        return values.expand(feature_indices.numel())
    if values.numel() != feature_indices.numel():
        raise ValueError("feature_values must be scalar or match feature_indices length")
    return values


def apply_feature_intervention(
    latents: torch.Tensor,
    *,
    feature_indices: list[int] | torch.Tensor | None,
    mode: str | None,
    feature_values: float | list[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if mode is None or feature_indices is None:
        return latents
    if mode not in {"zero", "mean_clamp", "force_on"}:
        raise ValueError(f"unsupported feature intervention mode: {mode}")

    index_tensor = torch.as_tensor(feature_indices, device=latents.device, dtype=torch.long).flatten()
    if index_tensor.numel() == 0:
        return latents

    values = _coerce_feature_values(
        latents,
        feature_indices=index_tensor,
        feature_values=feature_values,
        mode=mode,
    )

    edited = latents.clone()
    edited[:, index_tensor] = values.unsqueeze(0).expand(latents.shape[0], -1)
    return edited


def _decode_sae_latents(
    sae: "Autoencoder",
    latents: torch.Tensor,
    info: dict[str, Any] | None,
) -> torch.Tensor:
    try:
        return sae.decode(latents, info)
    except TypeError:
        return sae.decode(latents)


def apply_sae_latent_intervention(
    attn_output: torch.Tensor,
    sae: "Autoencoder",
    *,
    latent_intervention: dict[str, Any] | None,
) -> torch.Tensor:
    bsz, seq_len, d_model = attn_output.shape
    flat = attn_output.reshape(-1, d_model)
    latents, info = sae.encode(flat)
    edited_latents = apply_feature_intervention(
        latents,
        feature_indices=(
            latent_intervention.get("feature_indices")
            if latent_intervention is not None
            else None
        ),
        mode=latent_intervention.get("mode") if latent_intervention is not None else None,
        feature_values=(
            latent_intervention.get("feature_values")
            if latent_intervention is not None
            else None
        ),
    )
    recons = _decode_sae_latents(sae, edited_latents, info)
    return recons.reshape(bsz, seq_len, d_model)


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pad_token_id = config.pad_token_id

    def create_position_ids_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.create_position_ids_from_input_ids(input_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        return context_layer, attn_probs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        self_outputs = self.self(hidden_states, attention_mask=attention_mask)
        attn_output = self.output(self_outputs[0], hidden_states)
        return attn_output, self_outputs[1]


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sae: "Autoencoder" | None = None,
        apply_intervention: bool = False,
        latent_intervention: dict[str, Any] | None = None,
        return_attn_output: bool = False,
    ):
        attn_output, _ = self.attention(hidden_states, attention_mask=attention_mask)
        if apply_intervention and sae is not None:
            attn_output = apply_sae_latent_intervention(
                attn_output,
                sae,
                latent_intervention=latent_intervention,
            )
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        if return_attn_output:
            return layer_output, attn_output
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sae_dict: dict[int, "Autoencoder"] | None = None,
        intervention_pattern: list[bool] | None = None,
        latent_intervention_dict: dict[int, dict[str, Any]] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        attn_outputs = {} if return_attn_outputs else None
        target_layers = set(attn_output_layers or [])

        for i, layer_module in enumerate(self.layer):
            apply_intervention = intervention_pattern[i] if intervention_pattern else False
            sae = sae_dict.get(i) if sae_dict else None
            latent_intervention = latent_intervention_dict.get(i) if latent_intervention_dict else None
            if return_attn_outputs and (i in target_layers):
                hidden_states, attn_output = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    sae=sae,
                    apply_intervention=apply_intervention,
                    latent_intervention=latent_intervention,
                    return_attn_output=True,
                )
                attn_outputs[i] = attn_output
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    sae=sae,
                    apply_intervention=apply_intervention,
                    latent_intervention=latent_intervention,
                )

        if return_attn_outputs:
            return hidden_states, attn_outputs
        return hidden_states


class RobertaModel(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        sae_dict: dict[int, "Autoencoder"] | None = None,
        intervention_pattern: list[bool] | None = None,
        latent_intervention_dict: dict[int, dict[str, Any]] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        embeddings = self.embeddings(input_ids, token_type_ids=token_type_ids)
        encoder_out = self.encoder(
            embeddings,
            attention_mask=extended_mask,
            sae_dict=sae_dict,
            intervention_pattern=intervention_pattern,
            latent_intervention_dict=latent_intervention_dict,
            return_attn_outputs=return_attn_outputs,
            attn_output_layers=attn_output_layers,
        )
        return encoder_out


class RobertaLMHead(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class RobertaForMaskedLM(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

    def tie_weights(self):
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sae_dict: dict[int, "Autoencoder"] | None = None,
        intervention_pattern: list[bool] | None = None,
        latent_intervention_dict: dict[int, dict[str, Any]] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        encoder_out = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            sae_dict=sae_dict,
            intervention_pattern=intervention_pattern,
            latent_intervention_dict=latent_intervention_dict,
            return_attn_outputs=return_attn_outputs,
            attn_output_layers=attn_output_layers,
        )
        if return_attn_outputs:
            sequence_output, attn_outputs = encoder_out
        else:
            sequence_output = encoder_out
            attn_outputs = None

        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_attn_outputs:
            return logits, loss, attn_outputs
        return logits, loss
