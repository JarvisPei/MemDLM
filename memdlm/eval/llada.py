"""
accelerate launch \
    --num_processes 4 \
    memdlm/eval/llada.py \
    --tasks gsm8k_cot \
    --model llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=inclusionAI/LLaDA-MoE-7B-A1B-Base,max_new_tokens=512,steps=512,block_size=512,cfg=0.0"
"""

from types import SimpleNamespace
from dataclasses import dataclass

import accelerate
import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

import dllm
import memdlm.models
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.schedulers import LinearAlphaScheduler
from memdlm.memory import ParametricMemory, MemoryConfig


@dataclass
class LLaDAEvalConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    max_length: int = 4096
    steps: int = 1024
    block_size: int = 1024

    pretrained: str = ""
    adapter_model_name_or_path: str | None = None
    base_model_name_or_path: str | None = None
    dtype: str | torch.dtype = "auto"
    load_in_4bit: bool = True
    attn_implementation: str | None = None
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"
    mem_enabled: bool = False
    mem_num_inner_steps: int = 2
    mem_num_inner_epochs: int = 1
    mem_masking_strategy: str = "pmc"
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
    mem_inner_layer_fraction: float = 0.1
    mem_inner_target_modules: str = "gate_proj,up_proj,down_proj"
    mem_inner_adapter_type: str = "lora"
    mem_sync_inner: bool = False
    mem_inner_gradient_checkpointing: bool = True
    mem_loss_weight_type: str = "uniform"
    mem_loss_norm_type: str = "token"


@register_model("llada")
class LLaDAEvalHarness(LM):
    @staticmethod
    def _parse_token_list(value):
        """Parse token list from string format like '[126081;126348]' or list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            if not value:
                return []
            return [int(x.strip()) for x in value.split(";") if x.strip()]
        elif isinstance(value, list):
            return value
        elif value is None:
            return []
        return []

    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = LLaDAEvalConfig()

        pretrained = kwargs.get("pretrained", config.pretrained)
        adapter_model_name_or_path = kwargs.get(
            "adapter_model_name_or_path", config.adapter_model_name_or_path
        )
        base_model_name_or_path = kwargs.get(
            "base_model_name_or_path", config.base_model_name_or_path
        )
        dtype = kwargs.get("dtype", config.dtype)
        load_in_4bit = kwargs.get("load_in_4bit", config.load_in_4bit)
        attn_implementation = kwargs.get(
            "attn_implementation", config.attn_implementation
        )
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = self._parse_token_list(
            kwargs.get("suppress_tokens", config.suppress_tokens)
        )
        begin_suppress_tokens = self._parse_token_list(
            kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        mem_enabled = kwargs.get("mem_enabled", config.mem_enabled)
        mem_num_inner_steps = kwargs.get("mem_num_inner_steps", config.mem_num_inner_steps)
        mem_num_inner_epochs = kwargs.get(
            "mem_num_inner_epochs", config.mem_num_inner_epochs
        )
        mem_masking_strategy = kwargs.get(
            "mem_masking_strategy", config.mem_masking_strategy
        )
        mem_mask_ratio_epsilon = kwargs.get(
            "mem_mask_ratio_epsilon", config.mem_mask_ratio_epsilon
        )
        mem_prompt_mask_ratio = kwargs.get(
            "mem_prompt_mask_ratio", config.mem_prompt_mask_ratio
        )
        mem_progressive_consistent_start_scale = kwargs.get(
            "mem_progressive_consistent_start_scale",
            config.mem_progressive_consistent_start_scale,
        )
        mem_inner_loss_type = kwargs.get(
            "mem_inner_loss_type", config.mem_inner_loss_type
        )
        mem_distill_temperature = kwargs.get(
            "mem_distill_temperature", config.mem_distill_temperature
        )
        mem_distill_hidden_type = kwargs.get(
            "mem_distill_hidden_type", config.mem_distill_hidden_type
        )
        mem_inner_loss_mask_mode = kwargs.get(
            "mem_inner_loss_mask_mode", config.mem_inner_loss_mask_mode
        )
        mem_inner_lr = kwargs.get("mem_inner_lr", config.mem_inner_lr)
        mem_inner_grad_clip = kwargs.get(
            "mem_inner_grad_clip", config.mem_inner_grad_clip
        )
        mem_inner_rank = kwargs.get("mem_inner_rank", config.mem_inner_rank)
        mem_inner_alpha = kwargs.get("mem_inner_alpha", config.mem_inner_alpha)
        mem_inner_dropout = kwargs.get("mem_inner_dropout", config.mem_inner_dropout)
        mem_inner_layer_fraction = kwargs.get(
            "mem_inner_layer_fraction", config.mem_inner_layer_fraction
        )
        mem_inner_target_modules = kwargs.get(
            "mem_inner_target_modules", config.mem_inner_target_modules
        )
        mem_inner_adapter_type = kwargs.get(
            "mem_inner_adapter_type", config.mem_inner_adapter_type
        )
        mem_sync_inner = kwargs.get("mem_sync_inner", config.mem_sync_inner)
        mem_inner_gradient_checkpointing = kwargs.get(
            "mem_inner_gradient_checkpointing",
            config.mem_inner_gradient_checkpointing,
        )
        mem_loss_weight_type = kwargs.get(
            "mem_loss_weight_type", config.mem_loss_weight_type
        )
        mem_loss_norm_type = kwargs.get(
            "mem_loss_norm_type", config.mem_loss_norm_type
        )

        accelerator = accelerate.Accelerator()

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        if adapter_model_name_or_path:
            peft_cfg = PeftConfig.from_pretrained(adapter_model_name_or_path)
            base_id = base_model_name_or_path or getattr(
                peft_cfg, "base_model_name_or_path", None
            )
            if not base_id:
                raise ValueError(
                    "adapter_config.json lacks base_model_name_or_path; "
                    "pass base_model_name_or_path."
                )
            model_args = SimpleNamespace(
                model_name_or_path=base_id,
                dtype=get_dtype(dtype),
                load_in_4bit=load_in_4bit,
                attn_implementation=attn_implementation,
            )
            self.model = memdlm.models.get_model(model_args=model_args)
            self.model = PeftModel.from_pretrained(
                self.model, adapter_model_name_or_path
            )
            tokenizer_model_id = base_id
        else:
            model_args = SimpleNamespace(
                model_name_or_path=pretrained,
                dtype=get_dtype(dtype),
                load_in_4bit=load_in_4bit,
                attn_implementation=attn_implementation,
            )
            self.model = memdlm.models.get_model(model_args=model_args)
            tokenizer_model_id = pretrained
        self.model.eval()

        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = memdlm.models.get_tokenizer(
            SimpleNamespace(model_name_or_path=tokenizer_model_id, model=self.model)
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.right_shift_logits = right_shift_logits

        self.memory = None
        if mem_enabled:
            if self.tokenizer.mask_token_id is None:
                raise ValueError("mask_token_id is required for parametric memory.")
            memory_config = MemoryConfig(
                enabled=True,
                num_inner_steps=mem_num_inner_steps,
                num_inner_epochs=mem_num_inner_epochs,
                masking_strategy=mem_masking_strategy,
                mask_ratio_epsilon=mem_mask_ratio_epsilon,
                prompt_mask_ratio=mem_prompt_mask_ratio,
                progressive_consistent_start_scale=mem_progressive_consistent_start_scale,
                inner_loss_type=mem_inner_loss_type,
                distill_temperature=mem_distill_temperature,
                distill_hidden_type=mem_distill_hidden_type,
                inner_loss_mask_mode=mem_inner_loss_mask_mode,
                inner_lr=mem_inner_lr,
                inner_grad_clip=mem_inner_grad_clip,
                inner_rank=mem_inner_rank,
                inner_alpha=mem_inner_alpha,
                inner_dropout=mem_inner_dropout,
                inner_layer_fraction=mem_inner_layer_fraction,
                inner_target_modules=mem_inner_target_modules,
                inner_adapter_type=mem_inner_adapter_type,
                sync_inner=mem_sync_inner,
                inner_gradient_checkpointing=mem_inner_gradient_checkpointing,
            )
            scheduler_weight_fn = (
                LinearAlphaScheduler().weight
                if mem_loss_weight_type == "scheduler"
                else (lambda t: torch.ones_like(t))
            )
            self.memory = ParametricMemory(
                model=self.model,
                config=memory_config,
                mask_token_id=self.tokenizer.mask_token_id,
                scheduler_weight_fn=scheduler_weight_fn,
                loss_weight_type=mem_loss_weight_type,
                loss_norm_type=mem_loss_norm_type,
                postprocess_outputs=self._postprocess_outputs,
            )

        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _maybe_memory_adapt(self, prompt_ids: torch.Tensor) -> None:
        if self.memory is None:
            return
        self.memory.reset_inner()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        labels = prompt_ids.clone()
        self.memory.adapt(
            {
                "input_ids": prompt_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> bool:
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        out = []
        for instance in tqdm(requests, desc="Computing likelihood..."):
            context, continuation = self._encode_pair(*instance.args)
            assert len(context) + len(continuation) <= self.max_length, (
                f"Context + continuation length exceeds {self.max_length} tokens: "
                f"{len(context)} + {len(continuation)}"
            )

            context = torch.tensor(context, device=self.device, dtype=torch.long)
            continuation = torch.tensor(
                continuation, device=self.device, dtype=torch.long
            )
            self._maybe_memory_adapt(context)
            with torch.no_grad():
                logprob = self.get_loglikelihood(context, continuation)
                isgreedy = self.suffix_greedy_prediction(context, continuation)
            out.append((logprob, isgreedy))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        out = []
        sampler = MDLMSampler(model=self.model, tokenizer=self.tokenizer)

        for instance in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = instance.args  # type: ignore
            prompt_ids = self.tokenizer(context)["input_ids"]
            prompt = [torch.tensor(prompt_ids, device=self.device, dtype=torch.long)]
            stop_tokens = gen_kwargs["until"]
            gen_max_new_tokens = gen_kwargs.get("max_gen_toks", self.max_new_tokens)
            self._maybe_memory_adapt(prompt[0])
            generated_ids = sampler.sample(
                inputs=prompt,
                steps=gen_max_new_tokens,
                max_new_tokens=gen_max_new_tokens,
                block_size=gen_max_new_tokens,
                temperature=0.0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                suppress_tokens=self.suppress_tokens,
                begin_suppress_tokens=self.begin_suppress_tokens,
                right_shift_logits=self.right_shift_logits,
            )
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt[0].shape[0] :], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    cli_evaluate()
