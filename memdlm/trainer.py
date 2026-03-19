"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from memdlm.memory import ParametricMemory, MemoryConfig
from dllm.utils.configs import TrainingArguments
from dllm.utils.data import prepend_bos
from dllm.core.trainers.utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback


class MemDLMTrainer(transformers.Trainer):

    @dataclass
    class MemDLMConfig(TrainingArguments):
        time_epsilon: float = 1e-3
        loss_weight_type: str = "scheduler"  # "scheduler", "uniform"
        loss_norm_type: str = "token"  # "batch", "sequence", "token"
        right_shift_logits: bool = False
        mem_enabled: bool = False
        mem_num_inner_steps: int = 2
        mem_num_inner_epochs: int = 1
        mem_masking_strategy: str = "pmc"  # "progressive_memory", "progressive_memory_consistent"/"pmc", "pre_only", or "no_progressive"
        mem_mask_ratio_epsilon: float = 1e-3
        mem_prompt_mask_ratio: float = 0.2
        mem_progressive_consistent_start_scale: float = 1.5
        mem_inner_loss_type: str = "ce"
        mem_distill_temperature: float = 1.0
        mem_distill_hidden_type: str = "mse"
        mem_inner_loss_mask_mode: str = "student_masked"
        mem_inner_lr: float = 0.1
        mem_inner_grad_clip: float = 1.0
        mem_inner_rank: int = 32
        mem_inner_alpha: float = 64.0
        mem_inner_dropout: float = 0.0
        mem_inner_layer_fraction: float = 0.1  # Negative: distill_hidden uses early layers
        mem_inner_target_modules: str = "gate_proj,up_proj,down_proj"
        mem_inner_adapter_type: str = "lora"  # "lora" or "full"
        mem_pad_unmask_mode: str = "none"  # "none", "inner", "both"
        mem_sync_inner: bool = True
        mem_inner_gradient_checkpointing: bool = True

    def __init__(
        self,
        args: MemDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits
        if args.mem_pad_unmask_mode not in ("none", "inner", "both"):
            raise ValueError("mem_pad_unmask_mode must be one of: none, inner, both")
        self.memory_pad_unmask_mode = args.mem_pad_unmask_mode
        self.pad_token_id = self.processing_class.pad_token_id
        self.memory_config = MemoryConfig(
            enabled=args.mem_enabled,
            num_inner_steps=args.mem_num_inner_steps,
            num_inner_epochs=args.mem_num_inner_epochs,
            masking_strategy=args.mem_masking_strategy,
            mask_ratio_epsilon=args.mem_mask_ratio_epsilon,
            prompt_mask_ratio=args.mem_prompt_mask_ratio,
            progressive_consistent_start_scale=args.mem_progressive_consistent_start_scale,
            inner_loss_type=args.mem_inner_loss_type,
            distill_temperature=args.mem_distill_temperature,
            distill_hidden_type=args.mem_distill_hidden_type,
            inner_loss_mask_mode=args.mem_inner_loss_mask_mode,
            inner_lr=args.mem_inner_lr,
            inner_grad_clip=args.mem_inner_grad_clip,
            inner_rank=args.mem_inner_rank,
            inner_alpha=args.mem_inner_alpha,
            inner_dropout=args.mem_inner_dropout,
            inner_layer_fraction=args.mem_inner_layer_fraction,
            inner_target_modules=args.mem_inner_target_modules,
            inner_adapter_type=args.mem_inner_adapter_type,
            pad_unmask_mode=args.mem_pad_unmask_mode,
            pad_token_id=self.pad_token_id,
            sync_inner=args.mem_sync_inner,
            inner_gradient_checkpointing=args.mem_inner_gradient_checkpointing,
        )
        self.memory = None
        if self.memory_config.enabled:
            if self.processing_class.mask_token_id is None:
                raise ValueError("mask_token_id is required for parametric memory.")
            self.memory = ParametricMemory(
                model=self.model,
                config=self.memory_config,
                mask_token_id=self.processing_class.mask_token_id,
                scheduler_weight_fn=self.scheduler.weight,
                loss_weight_type=self.loss_weight_type,
                loss_norm_type=self.loss_norm_type,
                postprocess_outputs=self._postprocess_outputs,
            )

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            if labels is not None:
                if torch.all(labels[:, 0] == -100):
                    return inputs

            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def create_optimizer(self):
        if self.optimizer is None and self.memory is not None:
            inner_params = self.memory.inner_manager.get_inner_parameters()
            for p in inner_params:
                p.requires_grad = False
            super().create_optimizer()
            for p in inner_params:
                p.requires_grad = True
            return self.optimizer
        return super().create_optimizer()

    def _compute_mdlm_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion language modeling loss.

        Applies stochastic masking to input tokens based on a diffusion timestep,
        then computes the weighted cross-entropy loss for predicting the original tokens.

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = self._build_maskable_mask(
            input_ids=input_ids,
            labels=labels,
            apply_pad_unmask=self.memory_pad_unmask_mode == "both",
        )  # [b, l]

        t = inputs.get("mem_t", None)
        masked_mask = inputs.get("mem_masked_mask", None)
        noised_input_ids = inputs.get("mem_noised_input_ids", None)
        if t is None or masked_mask is None or noised_input_ids is None:
            t, masked_mask, noised_input_ids = self._prepare_noised_inputs(
                input_ids=input_ids,
                labels=labels,
            )

        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)  # [b, l]

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()

        return (loss, outputs) if return_outputs else loss

    def _prepare_noised_inputs(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape
        maskable_mask = self._build_maskable_mask(
            input_ids=input_ids,
            labels=labels,
            apply_pad_unmask=self.memory_pad_unmask_mode == "both",
        )
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )
        return t, masked_mask, noised_input_ids

    def _build_maskable_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        apply_pad_unmask: bool,
    ) -> torch.Tensor:
        maskable_mask = labels != -100
        if apply_pad_unmask and self.pad_token_id is not None:
            maskable_mask = maskable_mask & (input_ids != self.pad_token_id)
        return maskable_mask

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        if self.memory is not None and self.memory_config.enabled:
            self.memory.reset_inner()
            if self.memory_config.masking_strategy in (
                "no_progressive",
                "progressive_memory_consistent",
                "pmc",
                "pre_only",
            ):
                t, masked_mask, noised_input_ids = self._prepare_noised_inputs(
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"],
                )
                inputs = dict(inputs)
                inputs["mem_t"] = t
                inputs["mem_masked_mask"] = masked_mask
                inputs["mem_noised_input_ids"] = noised_input_ids
            self.memory.adapt(inputs)

        return self._compute_mdlm_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            **kwargs,
        )
