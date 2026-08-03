from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class EncodedTextControl:
    features: torch.Tensor
    mask: torch.Tensor


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be torch.nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        return cls(
            linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B(self.lora_A(self.dropout(value)))
        return self.base(value) + delta * self.scaling


def _transformer_layers(model: nn.Module) -> Tuple[str, Sequence[nn.Module]]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return "transformer.layer", model.transformer.layer
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return "encoder.layer", model.encoder.layer
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return "encoder.layers", model.encoder.layers
    raise ValueError("text backbone exposes no supported transformer layer stack")


def install_distilbert_attention_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    last_n_layers: int,
) -> List[str]:
    """Install LoRA on q/k/v/out projections in the final transformer layers."""

    stack_name, layers = _transformer_layers(model)
    if last_n_layers <= 0 or last_n_layers > len(layers):
        raise ValueError(
            f"last_n_layers must be in [1, {len(layers)}], got {last_n_layers}"
        )
    target_names = ("q_lin", "k_lin", "v_lin", "out_lin")
    installed: List[str] = []
    start = len(layers) - last_n_layers
    for layer_index in range(start, len(layers)):
        layer = layers[layer_index]
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError(
                f"{stack_name}.{layer_index} has no DistilBERT attention module"
            )
        for target_name in target_names:
            projection = getattr(attention, target_name, None)
            if not isinstance(projection, nn.Linear):
                raise ValueError(
                    f"{stack_name}.{layer_index}.attention.{target_name} "
                    "is not torch.nn.Linear"
                )
            setattr(
                attention,
                target_name,
                LoRALinear.from_linear(
                    projection,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            installed.append(
                f"{stack_name}.{layer_index}.attention.{target_name}"
            )
    expected = last_n_layers * len(target_names)
    if len(installed) != expected:
        raise RuntimeError(f"installed {len(installed)} LoRA modules, expected {expected}")
    return installed


class TextPromptEncoder(nn.Module):
    """Frozen DistilBERT with attention LoRA and a learned static projection."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        output_dim: int = 256,
        max_length: int = 384,
        mean_pool: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 0.4,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 6,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for text control; install the pinned "
                "version from install/requirements.txt"
            ) from exc

        load_options = {"local_files_only": bool(local_files_only)}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **load_options
        )
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, **load_options
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.lora_module_names = install_distilbert_attention_lora(
            self.backbone,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
            last_n_layers=int(lora_last_n_layers),
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.output_dim = int(output_dim)
        self.max_length = int(max_length)
        self.mean_pool = bool(mean_pool)
        self.projection = (
            nn.Linear(hidden_size, self.output_dim)
            if hidden_size != self.output_dim
            else nn.Identity()
        )
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_uniform_(self.projection.weight, gain=0.5)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not prompts:
            return torch.zeros(
                (0, self.output_dim),
                device=device,
                dtype=next(self.backbone.parameters()).dtype,
            )
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("TextPromptEncoder accepts only non-empty prompts")
        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = self.backbone(**encoded)
        hidden = output.last_hidden_state
        attention_mask = encoded.get("attention_mask")
        if self.mean_pool:
            if attention_mask is None:
                raise ValueError("mean pooling requires an attention_mask")
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        else:
            pooled = hidden[:, 0]
        return self.projection(pooled)


class FiLMLayer(nn.Module):
    """ECoSim FiLM MLP with an explicit per-agent direct-control mask."""

    def __init__(
        self,
        *,
        feature_dim: int,
        conditioning_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        init_mode: str = "identity",
        identity_noise_std: float = 0.001,
    ) -> None:
        super().__init__()
        if min(feature_dim, conditioning_dim, hidden_dim) <= 0:
            raise ValueError("FiLM dimensions must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if identity_noise_std < 0:
            raise ValueError("identity_noise_std must be non-negative")
        init_mode = str(init_mode).lower()
        if init_mode not in {"identity", "random"}:
            raise ValueError("init_mode must be 'identity' or 'random'")
        self.feature_dim = int(feature_dim)
        self.conditioning_dim = int(conditioning_dim)
        layers: List[nn.Module] = [
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2 * feature_dim))
        self.film_mlp = nn.Sequential(*layers)

        final = self.film_mlp[-1]
        with torch.no_grad():
            if init_mode == "identity":
                nn.init.xavier_uniform_(final.weight, gain=0.01)
                final.bias[:feature_dim].fill_(1.0)
                final.bias[feature_dim:].zero_()
                if identity_noise_std:
                    final.bias.add_(
                        torch.randn_like(final.bias) * float(identity_noise_std)
                    )
            else:
                nn.init.xavier_uniform_(final.weight, gain=1.0)
                final.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        conditioning: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim not in {2, 3}:
            raise ValueError("features must have rank 2 or 3")
        if conditioning.ndim != 2:
            raise ValueError("conditioning must have rank 2")
        if mask.dtype != torch.bool:
            raise ValueError("control mask must be Boolean")
        if mask.ndim != 1:
            raise ValueError("control mask must have rank 1")
        n_agent = features.shape[0]
        if conditioning.shape[0] != n_agent or mask.shape[0] != n_agent:
            raise ValueError("features, conditioning, and mask agent counts differ")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature width must be {self.feature_dim}, got {features.shape[-1]}"
            )
        if conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(
                "conditioning width must be "
                f"{self.conditioning_dim}, got {conditioning.shape[-1]}"
            )

        if features.ndim == 2:
            film_parameters = self.film_mlp(conditioning)
            gamma, beta = film_parameters.split(self.feature_dim, dim=-1)
            conditioned = gamma * features + beta
            expanded_mask = mask.to(features.device).unsqueeze(-1)
        else:
            n_step = features.shape[1]
            expanded_conditioning = conditioning.unsqueeze(1).expand(
                -1, n_step, -1
            )
            film_parameters = self.film_mlp(expanded_conditioning)
            gamma, beta = film_parameters.split(self.feature_dim, dim=-1)
            conditioned = gamma * features + beta
            expanded_mask = mask.to(features.device).view(n_agent, 1, 1)
        return torch.where(expanded_mask, conditioned, features)


class TextControlAdapter(nn.Module):
    """Static text encoding and one masked FiLM module per CatK agent block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_blocks: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = int(num_blocks)
        output_dim = int(config.get("output_dim", 256))
        self.encoder = TextPromptEncoder(
            model_name_or_path=str(config.get("model_name_or_path")),
            output_dim=output_dim,
            max_length=int(config.get("max_length", 384)),
            mean_pool=bool(config.get("mean_pool", False)),
            lora_rank=int(config.get("lora_rank", 16)),
            lora_alpha=float(config.get("lora_alpha", 0.4)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            lora_last_n_layers=int(config.get("lora_last_n_layers", 6)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        control_dropout = float(config.get("control_dropout", 0.3))
        if control_dropout < 0 or control_dropout > 1:
            raise ValueError("control_dropout must be in [0, 1]")
        self.control_dropout = control_dropout

        blocks = [int(index) for index in config.get("film_blocks", range(num_blocks))]
        if len(set(blocks)) != len(blocks):
            raise ValueError("film_blocks must not contain duplicates")
        if any(index < 0 or index >= num_blocks for index in blocks):
            raise ValueError(f"film_blocks must be within [0, {num_blocks - 1}]")
        self.film_layers = nn.ModuleDict(
            {
                str(index): FiLMLayer(
                    feature_dim=hidden_dim,
                    conditioning_dim=output_dim,
                    hidden_dim=int(config.get("film_hidden_dim", 256)),
                    dropout=float(config.get("film_dropout", 0.0)),
                    init_mode=str(config.get("film_init_mode", "identity")),
                    identity_noise_std=float(
                        config.get("film_identity_noise_std", 0.001)
                    ),
                )
                for index in blocks
            }
        )

    def encode(
        self,
        prompts: Sequence[str],
        mask: torch.Tensor,
        device: torch.device,
        *,
        apply_control_dropout: bool = False,
    ) -> Optional[EncodedTextControl]:
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("text control mask must be a rank-1 Boolean tensor")
        if len(prompts) != mask.shape[0]:
            raise ValueError("prompt and mask agent counts differ")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("every prompt must be a string")

        effective_mask = mask.to(device=device).clone()
        nonempty = torch.tensor(
            [bool(prompt.strip()) for prompt in prompts],
            dtype=torch.bool,
            device=device,
        )
        effective_mask &= nonempty
        if apply_control_dropout and self.control_dropout:
            keep = torch.rand(effective_mask.shape, device=device) >= self.control_dropout
            effective_mask &= keep
        active_indices = torch.where(effective_mask)[0]
        if active_indices.numel() == 0:
            return None

        index_list = active_indices.detach().cpu().tolist()
        active_prompts = [prompts[index] for index in index_list]
        active_features = self.encoder(active_prompts, device=device)
        features = torch.zeros(
            (len(prompts), active_features.shape[-1]),
            device=device,
            dtype=active_features.dtype,
        )
        features = features.index_copy(0, active_indices, active_features)
        return EncodedTextControl(features=features, mask=effective_mask)

    def condition(
        self,
        features: torch.Tensor,
        encoded: Optional[EncodedTextControl],
        block_index: int,
    ) -> torch.Tensor:
        if block_index < 0 or block_index >= self.num_blocks:
            raise IndexError(
                f"block_index must be within [0, {self.num_blocks - 1}]"
            )
        layer = self.film_layers[str(block_index)] if str(block_index) in self.film_layers else None
        if encoded is None or layer is None:
            return features
        return layer(features, encoded.features, encoded.mask)

    def unfreeze_control_parameters(self) -> List[str]:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.encoder.backbone.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True
        for parameter in self.encoder.projection.parameters():
            parameter.requires_grad = True
        for parameter in self.film_layers.parameters():
            parameter.requires_grad = True
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]


__all__ = [
    "EncodedTextControl",
    "FiLMLayer",
    "LoRALinear",
    "TextControlAdapter",
    "TextPromptEncoder",
    "install_distilbert_attention_lora",
]
