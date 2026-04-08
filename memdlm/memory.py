from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryConfig:
    enabled: bool = False
    num_inner_steps: int = 2
    num_inner_epochs: int = 1
    masking_strategy: str = "pmc"  # "progressive_memory", "progressive_memory_consistent"/"pmc", "pre_only", or "no_progressive"
    mask_ratio_epsilon: float = 1e-3
    prompt_mask_ratio: float = 0.2
    progressive_consistent_start_scale: float = 1.5
    inner_loss_type: str = "ce"  # "ce" or "distill" or "distill_reverse" or "distill_hidden"
    distill_temperature: float = 1.0
    distill_hidden_type: str = "mse"  # "mse" or "cosine"
    inner_loss_mask_mode: str = "student_masked"  # Loss mask mode for distill/distill_hidden/ce: "student_masked" or "newly_revealed"
    inner_lr: float = 0.1
    inner_grad_clip: float = 1.0
    inner_rank: int = 32
    inner_alpha: float = 64.0
    inner_dropout: float = 0.0
    inner_layer_fraction: float = 0.1  # Negative: distill_hidden uses early layers
    inner_target_modules: str = "gate_proj,up_proj,down_proj"
    inner_adapter_type: str = "lora"  # "lora" or "full"
    pad_unmask_mode: str = "none"  # "none", "inner", "both"
    pad_token_id: int | None = None
    sync_inner: bool = True
    inner_gradient_checkpointing: bool = True


class InnerLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.mem_inner_A = nn.Parameter(
            torch.zeros(rank, in_features, device=device, dtype=dtype)
        )
        self.mem_inner_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_inner()

    def reset_inner(self) -> None:
        nn.init.kaiming_uniform_(self.mem_inner_A, a=5**0.5)
        nn.init.zeros_(self.mem_inner_B)

    def inner_parameters(self) -> List[nn.Parameter]:
        return [self.mem_inner_A, self.mem_inner_B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_inner = x.to(self.mem_inner_A.dtype) if x.dtype != self.mem_inner_A.dtype else x
        inner_out = self.dropout(x_inner) @ self.mem_inner_A.T @ self.mem_inner_B.T
        return base_out + self.scaling * inner_out.to(base_out.dtype)


class InnerFullLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        dropout: float,
    ):
        super().__init__()
        self.base_layer = base_layer
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.mem_inner_delta = nn.Parameter(
            torch.zeros_like(base_layer.weight, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_inner()

    def reset_inner(self) -> None:
        nn.init.zeros_(self.mem_inner_delta)

    def inner_parameters(self) -> List[nn.Parameter]:
        return [self.mem_inner_delta]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_inner = (
            x.to(self.mem_inner_delta.dtype)
            if x.dtype != self.mem_inner_delta.dtype
            else x
        )
        inner_out = self.dropout(x_inner) @ self.mem_inner_delta.T
        return base_out + inner_out.to(base_out.dtype)


class InnerLoRAManager:
    def __init__(
        self,
        model: nn.Module,
        config: MemoryConfig,
    ):
        self.model = model
        self.config = config
        self.adapter_layers: List[nn.Module] = []
        self._inject()
        if len(self.adapter_layers) == 0:
            raise ValueError(
                "No inner adapter layers were injected. Check inner_target_modules "
                "and model module names (e.g., gate_proj/up_proj/down_proj) or "
                "ensure PEFT/quantized layers expose a base_layer."
            )

    def _resolve_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for attr in ("module", "model", "base_model", "model"):
            model = getattr(model, attr, model)
        # Support multiple backbone layouts:
        # - LLaMA/Qwen style: model.layers / layers
        # - LLaDA style: model.transformer.blocks / transformer.blocks
        for path in (
            "model.layers",
            "layers",
            "model.transformer.blocks",
            "transformer.blocks",
        ):
            current = model
            ok = True
            for part in path.split("."):
                if not hasattr(current, part):
                    ok = False
                    break
                current = getattr(current, part)
            if ok:
                return current
        raise AttributeError("Cannot locate transformer layers for inner LoRA injection.")

    def _inject(self) -> None:
        layers = self._resolve_layers(self.model)

        num_layers = len(layers)
        layer_fraction = max(0.0, min(1.0, abs(self.config.inner_layer_fraction)))
        start_idx = max(0, int(num_layers * (1.0 - layer_fraction)))
        target_layers = range(start_idx, num_layers)
        target_modules = [
            m.strip()
            for m in self.config.inner_target_modules.split(",")
            if m.strip()
        ]

        for layer_idx in target_layers:
            layer = layers[layer_idx]
            self._inject_on_layer(layer, target_modules)

    def _inject_on_layer(self, layer: nn.Module, target_modules: List[str]) -> None:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts:
                self._inject_on_module(expert, target_modules)
            if hasattr(layer.mlp, "shared_experts"):
                self._inject_on_module(layer.mlp.shared_experts, target_modules)
            return

        if hasattr(layer, "mlp"):
            self._inject_on_module(layer.mlp, target_modules)
        if hasattr(layer, "self_attn"):
            self._inject_on_module(layer.self_attn, target_modules)
        if hasattr(layer, "attention") and not hasattr(layer, "self_attn"):
            self._inject_on_module(layer.attention, target_modules)
        self._inject_on_module(layer, target_modules)

    def _inject_on_module(self, parent: nn.Module, target_modules: List[str]) -> None:
        adapter_type = self.config.inner_adapter_type.lower()
        if adapter_type not in ("lora", "full"):
            raise ValueError(
                "inner_adapter_type must be 'lora' or 'full', "
                f"got '{self.config.inner_adapter_type}'."
            )
        for module_name in target_modules:
            if not hasattr(parent, module_name):
                continue
            base_layer = getattr(parent, module_name)
            target_linear = None
            if isinstance(base_layer, (InnerLoRALinear, InnerFullLinear)):
                continue
            if isinstance(base_layer, nn.Linear):
                target_linear = base_layer
            elif hasattr(base_layer, "base_layer") and isinstance(
                base_layer.base_layer, nn.Linear
            ):
                target_linear = base_layer.base_layer

            if target_linear is None or isinstance(
                target_linear, (InnerLoRALinear, InnerFullLinear)
            ):
                continue

            if adapter_type == "lora":
                inner_layer = InnerLoRALinear(
                    base_layer=target_linear,
                    rank=self.config.inner_rank,
                    alpha=self.config.inner_alpha,
                    dropout=self.config.inner_dropout,
                )
            else:
                inner_layer = InnerFullLinear(
                    base_layer=target_linear,
                    dropout=self.config.inner_dropout,
                )
            if target_linear is base_layer:
                setattr(parent, module_name, inner_layer)
            else:
                base_layer.base_layer = inner_layer
            self.adapter_layers.append(inner_layer)

    def reset_inner(self) -> None:
        for layer in self.adapter_layers:
            layer.reset_inner()

    def get_inner_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self.adapter_layers:
            params.extend(layer.inner_parameters())
        return params


class ParametricMemory:
    def __init__(
        self,
        model: nn.Module,
        config: MemoryConfig,
        mask_token_id: int,
        scheduler_weight_fn,
        loss_weight_type: str,
        loss_norm_type: str,
        postprocess_outputs,
    ):
        self.model = model
        self.config = config
        self.mask_token_id = mask_token_id
        self.scheduler_weight_fn = scheduler_weight_fn
        self.loss_weight_type = loss_weight_type
        self.loss_norm_type = loss_norm_type
        self.postprocess_outputs = postprocess_outputs
        self.inner_manager = InnerLoRAManager(model, config)
        self.inner_params = self.inner_manager.get_inner_parameters()
        self.inner_param_ids = {id(p) for p in self.inner_params}
        self.inner_optimizer = torch.optim.SGD(
            self.inner_params,
            lr=self.config.inner_lr,
        )

    def reset_inner(self) -> None:
        self.inner_manager.reset_inner()

    def _compute_loss_weights(self, t: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        b, l = input_ids.shape
        if self.loss_weight_type == "scheduler":
            return self.scheduler_weight_fn(t).unsqueeze(1).repeat(1, l)
        if self.loss_weight_type == "uniform":
            return torch.ones_like(input_ids)
        raise NotImplementedError

    def _normalize_loss(self, token_nll: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = token_nll.size(0)
        denom = mask.sum()
        if self.loss_norm_type == "token":
            token_nll = token_nll / denom.clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError("Invalid loss_norm_type.")
        return token_nll.sum()

    def _compute_inner_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        masked_mask: torch.Tensor,
        t: torch.Tensor,
        teacher_masked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self._forward_inner(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.postprocess_outputs(outputs)
        logits = outputs.logits

        if teacher_masked is None or self.config.inner_loss_mask_mode == "student_masked":
            loss_mask = masked_mask
        else:
            loss_mask = masked_mask & (~teacher_masked)

        loss_weights = self._compute_loss_weights(t=t, input_ids=input_ids)
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
        )
        token_nll = token_nll * loss_weights * loss_mask.to(token_nll.dtype)
        return self._normalize_loss(token_nll, loss_mask)

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_masked: torch.Tensor,
        teacher_masked: torch.Tensor | None,
        t: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_masked is None or self.config.inner_loss_mask_mode == "student_masked":
            loss_mask = student_masked
        else:
            loss_mask = student_masked & (~teacher_masked)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        temperature = max(self.config.distill_temperature, 1e-5)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)
        loss_weights = self._compute_loss_weights(t=t, input_ids=input_ids)
        token_kl = token_kl * loss_weights * loss_mask.to(token_kl.dtype)
        return self._normalize_loss(token_kl, loss_mask)

    def _compute_reverse_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_masked: torch.Tensor,
        teacher_masked: torch.Tensor | None,
        t: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_masked is None or self.config.inner_loss_mask_mode == "student_masked":
            loss_mask = student_masked
        else:
            loss_mask = student_masked & (~teacher_masked)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        temperature = max(self.config.distill_temperature, 1e-5)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_probs = student_log_probs.exp()
        token_kl = F.kl_div(teacher_log_probs, student_probs, reduction="none").sum(-1)
        loss_weights = self._compute_loss_weights(t=t, input_ids=input_ids)
        token_kl = token_kl * loss_weights * loss_mask.to(token_kl.dtype)
        return self._normalize_loss(token_kl, loss_mask)

    def _compute_hidden_distill_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_masked: torch.Tensor,
        teacher_masked: torch.Tensor | None,
        t: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_masked is None or self.config.inner_loss_mask_mode == "student_masked":
            loss_mask = student_masked
        else:
            loss_mask = student_masked & (~teacher_masked)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=student_hidden.device, requires_grad=True)
        if self.config.distill_hidden_type == "cosine":
            student_norm = F.normalize(student_hidden, dim=-1)
            teacher_norm = F.normalize(teacher_hidden, dim=-1)
            token_loss = 1.0 - (student_norm * teacher_norm).sum(-1)
        else:
            token_loss = F.mse_loss(student_hidden, teacher_hidden, reduction="none").mean(
                -1
            )
        loss_weights = self._compute_loss_weights(t=t, input_ids=input_ids)
        token_loss = token_loss * loss_weights * loss_mask.to(token_loss.dtype)
        return self._normalize_loss(token_loss, loss_mask)

    def _resolve_decoder_model(self) -> nn.Module | None:
        model: nn.Module = self.model
        for attr in ("module", "model", "base_model", "model"):
            model = getattr(model, attr, model)
        if not hasattr(model, "layers"):
            return None
        if not hasattr(model, "embed_tokens") and not hasattr(model, "word_embeddings"):
            return None
        if not hasattr(model, "rotary_emb"):
            return None
        return model

    @staticmethod
    def _get_embed_tokens(decoder: nn.Module):
        if hasattr(decoder, "embed_tokens"):
            return decoder.embed_tokens
        if hasattr(decoder, "word_embeddings"):
            return decoder.word_embeddings
        raise AttributeError("Cannot locate embedding layer (embed_tokens or word_embeddings).")

    def _run_decoder_layer(
        self,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        position_ids: torch.Tensor,
        position_embeddings: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        try:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        except TypeError:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return layer_outputs[0]

    def _compute_partial_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        layer_fraction: float,
    ) -> torch.Tensor | None:
        model = self._resolve_decoder_model()
        if model is None:
            return None
        layers = model.layers
        num_layers = len(layers)
        if num_layers == 0:
            return None
        fraction = max(0.0, min(1.0, layer_fraction))
        if fraction <= 0.0:
            return None
        num_to_run = max(1, int(num_layers * fraction))
        num_to_run = min(num_layers, num_to_run)

        embed_fn = self._get_embed_tokens(model)
        inputs_embeds = embed_fn(input_ids)
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(num_to_run):
            hidden_states = self._run_decoder_layer(
                layers[layer_idx],
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )

        if hasattr(model, "norm") and model.norm is not None:
            hidden_states = model.norm(hidden_states)
        return hidden_states

    def _resolve_lm_head(self) -> nn.Module | None:
        """Find the language model head (final linear projection to vocabulary)."""
        model = self.model
        for attr in ("base_model", "model"):
            if hasattr(model, "lm_head"):
                return model.lm_head
            candidate = getattr(model, attr, None)
            if candidate is not None:
                model = candidate
        if hasattr(model, "lm_head"):
            return model.lm_head
        return None

    def _forward_inner(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool = False,
    ):
        """Forward pass with gradients only for layers containing LoRA adapters.

        Runs early layers (without LoRA) under torch.no_grad() and detaches
        the hidden state at the boundary, so only late layers (with LoRA)
        store activations for backpropagation. Optionally applies gradient
        checkpointing to those late layers for additional memory savings.

        Falls back to self.model(...) if the decoder or LM head cannot be
        resolved (e.g. non-standard model architectures).
        """
        decoder = self._resolve_decoder_model()
        lm_head = self._resolve_lm_head()
        if decoder is None or lm_head is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )

        layers = decoder.layers
        num_layers = len(layers)
        layer_fraction = max(0.0, min(1.0, abs(self.config.inner_layer_fraction)))
        start_idx = max(0, int(num_layers * (1.0 - layer_fraction)))

        with torch.no_grad():
            embed_fn = self._get_embed_tokens(decoder)
            inputs_embeds = embed_fn(input_ids)
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)
            hidden_states = inputs_embeds
            position_embeddings = decoder.rotary_emb(hidden_states, position_ids)
            for layer_idx in range(start_idx):
                hidden_states = self._run_decoder_layer(
                    layers[layer_idx],
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                )

        hidden_states = hidden_states.detach()

        all_hidden_states = [] if output_hidden_states else None
        use_checkpointing = self.config.inner_gradient_checkpointing
        for layer_idx in range(start_idx, num_layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            if use_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._run_decoder_layer,
                    layers[layer_idx],
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                    cache_position,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._run_decoder_layer(
                    layers[layer_idx],
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                )

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        if hasattr(decoder, "norm") and decoder.norm is not None:
            hidden_states = decoder.norm(hidden_states)
        logits = lm_head(hidden_states)

        result = SimpleNamespace(logits=logits)
        if all_hidden_states is not None:
            result.hidden_states = tuple(all_hidden_states)
        return result

    def adapt(self, inputs: dict[str, torch.Tensor]) -> Tuple[List[float], List[float]]:
        if not self.config.enabled:
            return [], []

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        maskable_mask = labels != -100
        if (
            self.config.pad_unmask_mode in ("inner", "both")
            and self.config.pad_token_id is not None
        ):
            maskable_mask = maskable_mask & (input_ids != self.config.pad_token_id)
        b, l = input_ids.shape
        device = input_ids.device

        if self.config.masking_strategy not in (
            "progressive_memory",
            "progressive_memory_consistent",
            "pmc",
            "pre_only",
            "no_progressive",
        ):
            raise ValueError(
                "masking_strategy must be 'progressive_memory', "
                "'progressive_memory_consistent'/'pmc', 'pre_only', or 'no_progressive'."
            )

        no_progressive = self.config.masking_strategy == "no_progressive"
        progressive_consistent = self.config.masking_strategy in (
            "progressive_memory_consistent", "pmc",
        )
        pre_only = self.config.masking_strategy == "pre_only"
        current_input = input_ids.clone()
        mem_masked_mask = inputs.get("mem_masked_mask", None)
        mem_noised_input_ids = inputs.get("mem_noised_input_ids", None)
        mem_t = inputs.get("mem_t", None)
        if mem_masked_mask is not None:
            mem_masked_mask = mem_masked_mask & maskable_mask
            if mem_noised_input_ids is not None:
                mem_noised_input_ids = torch.where(
                    mem_masked_mask, self.mask_token_id, input_ids
                )
        if (progressive_consistent or pre_only) and (
            mem_masked_mask is None or mem_noised_input_ids is None
        ):
            if mem_t is not None and mem_t.dim() == 1 and mem_t.numel() == b:
                final_mask_prob = mem_t.unsqueeze(1).expand(b, l)
            else:
                final_mask_prob = torch.full(
                    (b, l), self.config.prompt_mask_ratio, device=device
                )
            mem_masked_mask = (
                torch.rand((b, l), device=device) < final_mask_prob
            ) & maskable_mask
            mem_noised_input_ids = torch.where(
                mem_masked_mask, self.mask_token_id, input_ids
            )
        progressive_consistent_active = (
            (progressive_consistent or pre_only)
            and mem_masked_mask is not None
            and mem_noised_input_ids is not None
        )
        if no_progressive and mem_masked_mask is not None and mem_noised_input_ids is not None:
            current_input = mem_noised_input_ids
            is_masked = mem_masked_mask
        elif progressive_consistent_active:
            final_masked = mem_masked_mask.clone()
            total_maskable = maskable_mask.sum(dim=1).clamp_min(1)
            final_ratio = final_masked.sum(dim=1).to(torch.float32) / total_maskable.to(
                torch.float32
            )
            start_ratio = final_ratio * self.config.progressive_consistent_start_scale
            start_ratio = torch.maximum(start_ratio, final_ratio)
            start_ratio = torch.minimum(start_ratio, torch.ones_like(start_ratio))

            is_masked = final_masked.clone()
            for b_idx in range(b):
                maskable_indices = maskable_mask[b_idx].nonzero(as_tuple=True)[0]
                total = maskable_indices.numel()
                if total == 0:
                    continue
                target_masked_total = max(
                    int(final_masked[b_idx].sum().item()),
                    int(total * start_ratio[b_idx].item()),
                )
                target_masked_total = min(total, target_masked_total)
                num_to_add = target_masked_total - int(final_masked[b_idx].sum().item())
                if num_to_add <= 0:
                    continue
                candidate_indices = (
                    maskable_mask[b_idx] & (~final_masked[b_idx])
                ).nonzero(as_tuple=True)[0]
                if candidate_indices.numel() == 0:
                    continue
                num_to_add = min(num_to_add, candidate_indices.numel())
                perm = torch.randperm(candidate_indices.numel(), device=device)
                add_indices = candidate_indices[perm[:num_to_add]]
                is_masked[b_idx, add_indices] = True
            current_input[is_masked] = self.mask_token_id
        else:
            if not no_progressive and self.config.num_inner_steps > 1:
                initial_mask_ratio = 0.8
            else:
                initial_mask_ratio = self.config.prompt_mask_ratio
            is_masked = (
                torch.rand((b, l), device=device) < initial_mask_ratio
            ) & maskable_mask
            current_input[is_masked] = self.mask_token_id

        inner_params = self.inner_params
        inner_lr = self.config.inner_lr
        grad_clip = self.config.inner_grad_clip

        losses: List[float] = []
        grad_norms: List[float] = []

        base_input = current_input.clone()
        base_masked = is_masked.clone()

        _gc_was_enabled = getattr(self.model, "is_gradient_checkpointing", False)
        _was_training = self.model.training
        if self.config.inner_gradient_checkpointing and not _gc_was_enabled:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.model.gradient_checkpointing_enable()
                if not _was_training:
                    self.model.train()

        num_steps = 1 if (no_progressive or pre_only) else self.config.num_inner_steps
        num_epochs = max(1, self.config.num_inner_epochs)
        for _epoch in range(num_epochs):
            current_input = base_input.clone()
            is_masked = base_masked.clone()
            for step in range(num_steps):
                if inner_params and self.config.sync_inner:
                    self.inner_optimizer.zero_grad(set_to_none=True)
                if no_progressive:
                    if mem_t is None:
                        t = torch.full(
                            (b,), self.config.prompt_mask_ratio, device=device
                        )
                    else:
                        t = mem_t
                    next_input = None
                    next_masked = None
                else:
                    if progressive_consistent_active:
                        total_maskable = maskable_mask.sum(dim=1).clamp_min(1)
                        current_ratio = is_masked.sum(dim=1).to(torch.float32) / (
                            total_maskable.to(torch.float32)
                        )
                        t = current_ratio
                    else:
                        if self.config.num_inner_steps > 1:
                            progress = step / (self.config.num_inner_steps - 1)
                            mask_ratio = 0.8 + (
                                self.config.prompt_mask_ratio - 0.8
                            ) * progress
                        else:
                            mask_ratio = self.config.prompt_mask_ratio
                        t = torch.full((b,), mask_ratio, device=device)
                    next_input = None
                    next_masked = None
                    if not pre_only and step < self.config.num_inner_steps - 1:
                        next_input = current_input.clone()
                        next_masked = is_masked.clone()
                        with torch.no_grad():
                            if progressive_consistent_active:
                                final_masked = mem_masked_mask
                                for b_idx in range(b):
                                    if final_masked is None:
                                        continue
                                    current_masked = int(next_masked[b_idx].sum().item())
                                    final_masked_count = int(final_masked[b_idx].sum().item())
                                    if current_masked <= final_masked_count:
                                        continue
                                    remaining_steps = self.config.num_inner_steps - 1 - step
                                    need_to_reveal_total = current_masked - final_masked_count
                                    num_to_reveal_now = (
                                        need_to_reveal_total + remaining_steps - 1
                                    ) // remaining_steps
                                    reveal_candidates = (
                                        next_masked[b_idx] & (~final_masked[b_idx])
                                    ).nonzero(as_tuple=True)[0]
                                    if reveal_candidates.numel() == 0:
                                        continue
                                    num_to_reveal_now = min(
                                        num_to_reveal_now, reveal_candidates.numel()
                                    )
                                    perm = torch.randperm(
                                        reveal_candidates.numel(), device=device
                                    )
                                    reveal_indices = reveal_candidates[
                                        perm[:num_to_reveal_now]
                                    ]
                                    next_masked[b_idx, reveal_indices] = False
                                    next_input[b_idx, reveal_indices] = labels[
                                        b_idx, reveal_indices
                                    ]
                                if mem_noised_input_ids is not None and (
                                    step == self.config.num_inner_steps - 2
                                ):
                                    next_input = mem_noised_input_ids.clone()
                                    if mem_masked_mask is not None:
                                        next_masked = mem_masked_mask.clone()
                            else:
                                if self.config.num_inner_steps > 1:
                                    next_progress = (step + 1) / (self.config.num_inner_steps - 1)
                                    next_mask_ratio = 0.8 + (
                                        self.config.prompt_mask_ratio - 0.8
                                    ) * next_progress
                                else:
                                    next_mask_ratio = self.config.prompt_mask_ratio
                                for b_idx in range(b):
                                    maskable_indices = maskable_mask[b_idx].nonzero(
                                        as_tuple=True
                                    )[0]
                                    total_maskable = len(maskable_indices)
                                    target_masked_total = int(total_maskable * next_mask_ratio)
                                    current_masked = next_masked[b_idx].sum().item()
                                    num_to_reveal_now = max(
                                        0, current_masked - target_masked_total
                                    )
                                    if num_to_reveal_now == 0 or current_masked == 0:
                                        continue
                                    masked_indices = next_masked[b_idx].nonzero(
                                        as_tuple=True
                                    )[0]
                                    perm = torch.randperm(len(masked_indices), device=device)
                                    reveal_indices = masked_indices[perm[:num_to_reveal_now]]
                                    next_masked[b_idx, reveal_indices] = False
                                    next_input[b_idx, reveal_indices] = labels[
                                        b_idx, reveal_indices
                                    ]

                if pre_only and mem_noised_input_ids is not None:
                    next_input = mem_noised_input_ids
                    next_masked = mem_masked_mask

                with torch.enable_grad():
                    if self.config.inner_loss_type in ("distill", "distill_reverse"):
                        _distill_fn = (
                            self._compute_reverse_distill_loss
                            if self.config.inner_loss_type == "distill_reverse"
                            else self._compute_distill_loss
                        )
                        if no_progressive:
                            teacher_input_ids = input_ids
                            with torch.no_grad():
                                teacher_outputs = self.model(
                                    input_ids=teacher_input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=False,
                                )
                                teacher_outputs = self.postprocess_outputs(teacher_outputs)
                                teacher_logits = teacher_outputs.logits.detach()

                            student_outputs = self._forward_inner(
                                input_ids=current_input,
                                attention_mask=attention_mask,
                                output_hidden_states=False,
                            )
                            student_outputs = self.postprocess_outputs(student_outputs)
                            loss = _distill_fn(
                                student_logits=student_outputs.logits,
                                teacher_logits=teacher_logits,
                                student_masked=is_masked,
                                teacher_masked=None,
                                t=t,
                                input_ids=current_input,
                            )
                        elif next_input is None or next_masked is None:
                            teacher_input_ids = input_ids
                            with torch.no_grad():
                                teacher_outputs = self.model(
                                    input_ids=teacher_input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=False,
                                )
                                teacher_outputs = self.postprocess_outputs(teacher_outputs)
                                teacher_logits = teacher_outputs.logits.detach()

                            student_outputs = self._forward_inner(
                                input_ids=current_input,
                                attention_mask=attention_mask,
                                output_hidden_states=False,
                            )
                            student_outputs = self.postprocess_outputs(student_outputs)
                            loss = _distill_fn(
                                student_logits=student_outputs.logits,
                                teacher_logits=teacher_logits,
                                student_masked=is_masked,
                                teacher_masked=None,
                                t=t,
                                input_ids=current_input,
                            )
                        else:
                            with torch.no_grad():
                                teacher_outputs = self.model(
                                    input_ids=next_input,
                                    attention_mask=attention_mask,
                                    output_hidden_states=False,
                                )
                                teacher_outputs = self.postprocess_outputs(teacher_outputs)
                                teacher_logits = teacher_outputs.logits.detach()

                            student_outputs = self._forward_inner(
                                input_ids=current_input, attention_mask=attention_mask, output_hidden_states=False
                            )
                            student_outputs = self.postprocess_outputs(student_outputs)
                            loss = _distill_fn(
                                student_logits=student_outputs.logits,
                                teacher_logits=teacher_logits,
                                student_masked=is_masked,
                                teacher_masked=next_masked,
                                t=t,
                                input_ids=current_input,
                            )
                    elif self.config.inner_loss_type == "distill_hidden":
                        if no_progressive:
                            layer_fraction = self.config.inner_layer_fraction
                            use_partial = layer_fraction < 0.0
                            partial_fraction = abs(layer_fraction)
                            teacher_input_ids = input_ids
                            with torch.no_grad():
                                teacher_hidden = None
                                if use_partial:
                                    teacher_hidden = self._compute_partial_hidden(
                                        input_ids=teacher_input_ids,
                                        attention_mask=attention_mask,
                                        layer_fraction=partial_fraction,
                                    )
                                if teacher_hidden is None:
                                    teacher_outputs = self.model(
                                        input_ids=teacher_input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True,
                                    )
                                    teacher_hidden = teacher_outputs.hidden_states[-1].detach()
                                else:
                                    teacher_hidden = teacher_hidden.detach()

                            student_hidden = None
                            if use_partial:
                                student_hidden = self._compute_partial_hidden(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    layer_fraction=partial_fraction,
                                )
                            if student_hidden is None:
                                student_outputs = self._forward_inner(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                )
                                student_hidden = student_outputs.hidden_states[-1]
                            loss = self._compute_hidden_distill_loss(
                                student_hidden=student_hidden,
                                teacher_hidden=teacher_hidden,
                                student_masked=is_masked,
                                teacher_masked=None,
                                t=t,
                                input_ids=current_input,
                            )
                        elif next_input is None or next_masked is None:
                            layer_fraction = self.config.inner_layer_fraction
                            use_partial = layer_fraction < 0.0
                            partial_fraction = abs(layer_fraction)
                            teacher_input_ids = input_ids
                            with torch.no_grad():
                                teacher_hidden = None
                                if use_partial:
                                    teacher_hidden = self._compute_partial_hidden(
                                        input_ids=teacher_input_ids,
                                        attention_mask=attention_mask,
                                        layer_fraction=partial_fraction,
                                    )
                                if teacher_hidden is None:
                                    teacher_outputs = self.model(
                                        input_ids=teacher_input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True,
                                    )
                                    teacher_hidden = teacher_outputs.hidden_states[-1].detach()
                                else:
                                    teacher_hidden = teacher_hidden.detach()

                            student_hidden = None
                            if use_partial:
                                student_hidden = self._compute_partial_hidden(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    layer_fraction=partial_fraction,
                                )
                            if student_hidden is None:
                                student_outputs = self._forward_inner(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                )
                                student_hidden = student_outputs.hidden_states[-1]
                            loss = self._compute_hidden_distill_loss(
                                student_hidden=student_hidden,
                                teacher_hidden=teacher_hidden,
                                student_masked=is_masked,
                                teacher_masked=None,
                                t=t,
                                input_ids=current_input,
                            )
                        else:
                            layer_fraction = self.config.inner_layer_fraction
                            use_partial = layer_fraction < 0.0
                            partial_fraction = abs(layer_fraction)
                            with torch.no_grad():
                                teacher_hidden = None
                                if use_partial:
                                    teacher_hidden = self._compute_partial_hidden(
                                        input_ids=next_input,
                                        attention_mask=attention_mask,
                                        layer_fraction=partial_fraction,
                                    )
                                if teacher_hidden is None:
                                    teacher_outputs = self.model(
                                        input_ids=next_input,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True,
                                    )
                                    teacher_hidden = teacher_outputs.hidden_states[-1].detach()
                                else:
                                    teacher_hidden = teacher_hidden.detach()

                            student_hidden = None
                            if use_partial:
                                student_hidden = self._compute_partial_hidden(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    layer_fraction=partial_fraction,
                                )
                            if student_hidden is None:
                                student_outputs = self._forward_inner(
                                    input_ids=current_input,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                )
                                student_hidden = student_outputs.hidden_states[-1]
                            loss = self._compute_hidden_distill_loss(
                                student_hidden=student_hidden,
                                teacher_hidden=teacher_hidden,
                                student_masked=is_masked,
                                teacher_masked=next_masked,
                                t=t,
                                input_ids=current_input,
                            )
                    else:
                        loss = self._compute_inner_loss(
                            input_ids=current_input,
                            attention_mask=attention_mask,
                            labels=labels,
                            masked_mask=is_masked,
                            t=t,
                            teacher_masked=next_masked,
                        )

                if torch.isnan(loss) or torch.isinf(loss):
                    losses.append(float("nan"))
                    grad_norms.append(0.0)
                    continue

                losses.append(loss.item())

                if inner_params:
                    if self.config.sync_inner:
                        torch.autograd.backward(loss, inputs=inner_params)

                        total_norm_sq = 0.0
                        for p in inner_params:
                            if p.grad is not None:
                                total_norm_sq += p.grad.norm().item() ** 2
                        grad_norms.append(total_norm_sq**0.5)

                        if grad_clip > 0:
                            for p in inner_params:
                                if p.grad is None:
                                    continue
                                gnorm = p.grad.norm().item()
                                if gnorm > grad_clip:
                                    p.grad.mul_(grad_clip / (gnorm + 1e-12))
                        self.inner_optimizer.step()
                    else:
                        grads = torch.autograd.grad(
                            loss,
                            inner_params,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True,
                        )
                        total_norm_sq = 0.0
                        for g in grads:
                            if g is not None:
                                total_norm_sq += g.norm().item() ** 2
                        grad_norms.append(total_norm_sq**0.5)

                        with torch.no_grad():
                            for param, g in zip(inner_params, grads):
                                if g is None:
                                    continue
                                gnorm = g.norm().item()
                                if grad_clip > 0 and gnorm > grad_clip:
                                    g = g * (grad_clip / (gnorm + 1e-12))
                                param.data.sub_(inner_lr * g)
                else:
                    grad_norms.append(0.0)

                if next_input is not None and next_masked is not None:
                    current_input = next_input
                    is_masked = next_masked

        if self.config.inner_gradient_checkpointing and not _gc_was_enabled:
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if not _was_training:
                self.model.eval()

        if inner_params and self.config.sync_inner:
            self.inner_optimizer.zero_grad(set_to_none=True)
        return losses, grad_norms
