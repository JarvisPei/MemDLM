"""
Interactive chat / sampling script for LLaDA models.

Examples
--------
# Chat mode (multi-turn, chat template)
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH"

# Raw single-turn sampling
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH" --chat_template False
"""

import sys
from dataclasses import dataclass

import torch
import transformers
from peft import PeftConfig, PeftModel

import dllm
import memdlm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "inclusionAI/LLaDA-MoE-7B-A1B-Base"
    adapter_model_name_or_path: str | None = None
    base_model_name_or_path: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    mem_enabled: bool = False
    mem_num_inner_steps: int = 2
    mem_num_inner_epochs: int = 1
    mem_masking_strategy: str = "pmc"
    mem_mask_ratio_epsilon: float = 1e-3
    mem_prompt_mask_ratio: float = 0.2
    mem_inner_loss_type: str = "ce"
    mem_distill_temperature: float = 1.0
    mem_distill_hidden_type: str = "mse"
    mem_inner_loss_mask_mode: str = "student_masked"
    mem_inner_lr: float = 0.1
    mem_inner_grad_clip: float = 1.0
    mem_inner_rank: int = 32
    mem_inner_alpha: float = 64.0
    mem_inner_dropout: float = 0.0
    mem_inner_layer_fraction: float = 0.1
    mem_inner_target_modules: str = "gate_proj,up_proj,down_proj"
    mem_inner_adapter_type: str = "lora"
    mem_sync_inner: bool = True
    mem_loss_weight_type: str = "uniform"
    mem_loss_norm_type: str = "token"
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        # same base-path resolution logic as in sample.py
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        if self.adapter_model_name_or_path:
            self.adapter_model_name_or_path = dllm.utils.resolve_with_base_env(
                self.adapter_model_name_or_path, "BASE_MODELS_DIR"
            )
        if self.base_model_name_or_path:
            self.base_model_name_or_path = dllm.utils.resolve_with_base_env(
                self.base_model_name_or_path, "BASE_MODELS_DIR"
            )


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


class MemoryWrappedSampler:
    def __init__(self, sampler, memory):
        self.sampler = sampler
        self.memory = memory
        self.model = sampler.model
        self.tokenizer = sampler.tokenizer

    def _adapt_on_prompt(self, inputs: list[torch.Tensor | list]) -> None:
        if self.memory is None:
            return
        prompt = inputs[0]
        if isinstance(prompt, list):
            prompt = torch.as_tensor(prompt, dtype=torch.long, device=self.model.device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        attention_mask = torch.ones_like(prompt, device=prompt.device)
        labels = prompt.clone()
        self.memory.reset_inner()
        self.memory.adapt(
            {
                "input_ids": prompt,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    def sample(self, inputs, config=None, **kwargs):
        self._adapt_on_prompt(inputs)
        return self.sampler.sample(inputs, config=config, **kwargs)


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    if script_args.adapter_model_name_or_path:
        peft_cfg = PeftConfig.from_pretrained(script_args.adapter_model_name_or_path)
        base_id = script_args.base_model_name_or_path or getattr(
            peft_cfg, "base_model_name_or_path", None
        )
        if not base_id:
            raise ValueError(
                "adapter_config.json lacks base_model_name_or_path; "
                "pass --base_model_name_or_path."
            )
        model_args = ScriptArguments(
            model_name_or_path=base_id,
            dtype=script_args.dtype,
            load_in_4bit=script_args.load_in_4bit,
            attn_implementation=script_args.attn_implementation,
            seed=script_args.seed,
            chat_template=script_args.chat_template,
            visualize=script_args.visualize,
        )
        model = memdlm.models.get_model(model_args=model_args).eval()
        model = PeftModel.from_pretrained(model, script_args.adapter_model_name_or_path)
        tokenizer = memdlm.models.get_tokenizer(model_args=model_args)
    else:
        model = memdlm.models.get_model(model_args=script_args).eval()
        tokenizer = memdlm.models.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

    memory = None
    if script_args.mem_enabled:
        if tokenizer.mask_token_id is None:
            raise ValueError("mask_token_id is required for parametric memory.")
        memory_config = memdlm.MemoryConfig(
            enabled=True,
            num_inner_steps=script_args.mem_num_inner_steps,
            num_inner_epochs=script_args.mem_num_inner_epochs,
            masking_strategy=script_args.mem_masking_strategy,
            mask_ratio_epsilon=script_args.mem_mask_ratio_epsilon,
            prompt_mask_ratio=script_args.mem_prompt_mask_ratio,
            inner_loss_type=script_args.mem_inner_loss_type,
            distill_temperature=script_args.mem_distill_temperature,
            distill_hidden_type=script_args.mem_distill_hidden_type,
            inner_loss_mask_mode=script_args.mem_inner_loss_mask_mode,
            inner_lr=script_args.mem_inner_lr,
            inner_grad_clip=script_args.mem_inner_grad_clip,
            inner_rank=script_args.mem_inner_rank,
            inner_alpha=script_args.mem_inner_alpha,
            inner_dropout=script_args.mem_inner_dropout,
            inner_layer_fraction=script_args.mem_inner_layer_fraction,
            inner_target_modules=script_args.mem_inner_target_modules,
            inner_adapter_type=script_args.mem_inner_adapter_type,
            sync_inner=script_args.mem_sync_inner,
        )
        scheduler_weight_fn = (
            dllm.core.schedulers.LinearAlphaScheduler().weight
            if script_args.mem_loss_weight_type == "scheduler"
            else (lambda t: torch.ones_like(t))
        )
        memory = memdlm.ParametricMemory(
            model=model,
            config=memory_config,
            mask_token_id=tokenizer.mask_token_id,
            scheduler_weight_fn=scheduler_weight_fn,
            loss_weight_type=script_args.mem_loss_weight_type,
            loss_norm_type=script_args.mem_loss_norm_type,
            postprocess_outputs=lambda o: o,
        )
        sampler = MemoryWrappedSampler(sampler, memory)

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn sampling (no chat template).")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
