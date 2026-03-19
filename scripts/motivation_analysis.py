"""
Motivation Analysis: Quantifying Denoising Exposure Bias in dLLMs.

This script measures the cross-entropy loss under two (optionally three) trajectories,
for one or two models simultaneously:

  Trajectory A (Static / Training Condition):
    Ground-truth response tokens are randomly masked at fixed ratios.
    The model predicts masked tokens from pristine, ground-truth context.

  Trajectory B (Sequential / Inference Condition):
    Starting from 100% masked response, the model iteratively predicts and
    unmasks tokens using its own predictions, measuring loss at each step.

  Trajectory C (Sequential + Our Method):
    Same as Trajectory B, but after applying our inner-loop adaptation (TTT)
    on the response before generation begins. This tests whether our method
    reduces the exposure bias gap.

When --model2_* is provided, a second model is loaded and evaluated on the
same samples with the same trajectories, enabling direct comparison between
a baseline dLLM and one trained with our method.

Supports multi-GPU via accelerate:
    accelerate launch --num_processes 4 scripts/motivation_analysis.py ...

Usage:
    # Single model:
    python scripts/motivation_analysis.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base \
        --dataset_name allenai/tulu-3-sft-mixture \
        --dataset_split test \
        --num_samples 200 \
        --num_steps 10 \
        --output_path results/motivation_exposure_bias.json

    # Two models (baseline vs. ours):
    python scripts/motivation_analysis.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base \
        --adapter_model_name_or_path /path/to/baseline_sft_adapter \
        --model2_name_or_path GSAI-ML/LLaDA-8B-Base \
        --model2_adapter_model_name_or_path /path/to/ours_sft_adapter \
        --mem_enabled \
        --dataset_name allenai/tulu-3-sft-mixture \
        --dataset_split test \
        --num_samples 200 \
        --num_steps 10 \
        --output_path results/motivation_two_models.json

    # Multi-GPU:
    accelerate launch --num_processes 4 scripts/motivation_analysis.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base \
        --model2_adapter_model_name_or_path /path/to/ours_adapter \
        --mem_enabled \
        --dataset_name allenai/tulu-3-sft-mixture \
        --num_samples 200 \
        --output_path results/motivation_two_models.json

    # From a preprocessed dataset saved on disk:
    python scripts/motivation_analysis.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base \
        --dataset_name /path/to/preprocessed_dataset \
        --load_preprocessed_data \
        --dataset_split test \
        --num_samples 200 \
        --output_path results/motivation_exposure_bias.json
"""

import argparse
import json
import os
from functools import partial
from types import SimpleNamespace

import accelerate
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import dllm
import memdlm.models
from dllm.core.schedulers import LinearAlphaScheduler
from memdlm.memory import ParametricMemory, MemoryConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Motivation analysis: quantifying denoising exposure bias in dLLMs"
    )
    # --- Model 1 (baseline) ---
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--adapter_model_name_or_path", type=str, default=None,
                        help="PEFT adapter for model 1.")
    parser.add_argument("--base_model_name_or_path", type=str, default=None,
                        help="Override base model path for model 1 adapter.")

    # --- Model 2 (ours, optional) ---
    parser.add_argument("--model2_name_or_path", type=str, default=None,
                        help="Base model path for model 2. If None, reuses model 1's base.")
    parser.add_argument("--model2_adapter_model_name_or_path", type=str, default=None,
                        help="PEFT adapter for model 2. Setting this enables two-model comparison.")
    parser.add_argument("--model2_base_model_name_or_path", type=str, default=None,
                        help="Override base model path for model 2 adapter.")

    # --- Dataset ---
    parser.add_argument("--dataset_name", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--load_preprocessed_data", action="store_true",
                        help="Load preprocessed data from disk (dataset_name is a local path)")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=2048)

    # --- Analysis ---
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of denoising steps (mask ratio checkpoints)")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="results/motivation_exposure_bias.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # --- Memory options (for Trajectory C, applied to both models) ---
    parser.add_argument("--mem_enabled", action="store_true")
    parser.add_argument("--mem_num_inner_steps", type=int, default=2)
    parser.add_argument("--mem_prompt_mask_ratio", type=float, default=0.2)
    parser.add_argument("--mem_inner_lr", type=float, default=0.1)
    parser.add_argument("--mem_inner_rank", type=int, default=32)
    parser.add_argument("--mem_inner_alpha", type=float, default=64.0)
    parser.add_argument("--mem_inner_layer_fraction", type=float, default=0.1)
    parser.add_argument("--mem_inner_target_modules", type=str, default="gate_proj,up_proj,down_proj")
    parser.add_argument("--mem_masking_strategy", type=str, default="progressive_memory_consistent",
                        choices=["progressive_memory", "progressive_memory_consistent", "no_progressive"])
    parser.add_argument("--mem_progressive_consistent_start_scale", type=float, default=2.0)
    parser.add_argument("--mem_inner_loss_mask_mode", type=str, default="student_masked",
                        choices=["student_masked", "newly_revealed"])

    return parser.parse_args()


def _load_single_model(model_name_or_path, adapter_path, base_override, args, accelerator):
    """Load a single model (with optional adapter) and its tokenizer."""
    from peft import PeftConfig, PeftModel

    if adapter_path:
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        base_id = base_override or getattr(peft_cfg, "base_model_name_or_path", None)
        if not base_id:
            raise ValueError(
                f"adapter_config.json at {adapter_path} lacks base_model_name_or_path; "
                "pass the corresponding --base_model_name_or_path."
            )
        model_args = SimpleNamespace(
            model_name_or_path=base_id,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            lora=False,
        )
        model = memdlm.models.get_model(model_args=model_args)
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer_model_id = base_id
    else:
        model_args = SimpleNamespace(
            model_name_or_path=model_name_or_path,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
            lora=False,
        )
        model = memdlm.models.get_model(model_args=model_args)
        tokenizer_model_id = model_name_or_path

    model.eval()

    if accelerator.num_processes > 1:
        model = accelerator.prepare(model)
        device = accelerator.device
    else:
        model = model.to(args.device)
        device = torch.device(args.device)

    tokenizer = memdlm.models.get_tokenizer(
        SimpleNamespace(model_name_or_path=tokenizer_model_id, model=model)
    )
    return model, tokenizer, device


def load_dataset_samples(args, tokenizer):
    """Load and tokenize SFT dataset, extract prompt/response pairs."""
    dataset = dllm.data.load_sft_dataset(
        f"{args.dataset_name}[{args.dataset_split}:{args.num_samples}]",
        load_preprocessed_data=args.load_preprocessed_data,
    )

    if not args.load_preprocessed_data:
        map_fn = partial(
            dllm.utils.default_sft_map_fn,
            tokenizer=tokenizer,
            mask_prompt_loss=True,
        )
        dataset = dataset.map(map_fn, desc="Tokenizing")

    split_name = args.dataset_split if args.dataset_split in dataset else list(dataset.keys())[0]

    samples = []
    for row in dataset[split_name]:
        input_ids = row["input_ids"]
        labels = row["labels"]
        if len(input_ids) > args.max_length:
            continue

        prompt_len = sum(1 for l in labels if l == -100)
        response_len = len(input_ids) - prompt_len
        if response_len < 10:
            continue

        samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": prompt_len,
            "response_len": response_len,
        })

        if len(samples) >= args.num_samples:
            break

    return samples


def _build_memory(model, mask_token_id, args):
    """Build a ParametricMemory instance for a given model."""
    scheduler = LinearAlphaScheduler()
    memory_config = MemoryConfig(
        enabled=True,
        num_inner_steps=args.mem_num_inner_steps,
        masking_strategy=args.mem_masking_strategy,
        prompt_mask_ratio=args.mem_prompt_mask_ratio,
        progressive_consistent_start_scale=args.mem_progressive_consistent_start_scale,
        inner_loss_mask_mode=args.mem_inner_loss_mask_mode,
        inner_lr=args.mem_inner_lr,
        inner_rank=args.mem_inner_rank,
        inner_alpha=args.mem_inner_alpha,
        inner_layer_fraction=args.mem_inner_layer_fraction,
        inner_target_modules=args.mem_inner_target_modules,
        sync_inner=False,
    )

    def identity_postprocess(outputs):
        return outputs

    return ParametricMemory(
        model=model,
        config=memory_config,
        mask_token_id=mask_token_id,
        scheduler_weight_fn=scheduler.weight,
        loss_weight_type="uniform",
        loss_norm_type="token",
        postprocess_outputs=identity_postprocess,
    )


# ============================================================================
# Trajectory computation functions
# ============================================================================

@torch.no_grad()
def compute_static_loss(model, sample, target_mask_ratios, mask_token_id, device,
                        precomputed_masks=None):
    """Trajectory A: Random static masking at each target ratio.
    
    If precomputed_masks is provided (dict mapping mask_ratio -> mask_positions tensor),
    uses those exact positions instead of generating new random ones.
    """
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    prompt_len = sample["prompt_len"]
    seq_len = input_ids.shape[1]
    response_len = sample["response_len"]
    response_indices = torch.arange(prompt_len, seq_len, device=device)

    results = []
    for mask_ratio in target_mask_ratios:
        if precomputed_masks is not None and mask_ratio in precomputed_masks:
            mask_positions = precomputed_masks[mask_ratio].to(device)
        else:
            num_to_mask = max(1, int(response_len * mask_ratio))
            num_to_mask = min(num_to_mask, response_len)
            perm = torch.randperm(response_len, device=device)
            mask_positions = response_indices[perm[:num_to_mask]]

        noised_input = input_ids.clone()
        noised_input[0, mask_positions] = mask_token_id

        logits = model(input_ids=noised_input).logits
        ce_loss = F.cross_entropy(
            logits[0, mask_positions],
            input_ids[0, mask_positions],
            reduction="mean",
        )
        results.append({"mask_ratio": mask_ratio, "ce_loss": ce_loss.item()})

    return results


def precompute_static_masks(samples, target_mask_ratios):
    """Pre-generate random mask positions for Trajectory A so all models
    are evaluated on identical masked inputs for fair comparison."""
    all_masks = []
    for sample in samples:
        prompt_len = sample["prompt_len"]
        seq_len = len(sample["input_ids"])
        response_len = sample["response_len"]
        response_indices = torch.arange(prompt_len, seq_len)

        sample_masks = {}
        for mask_ratio in target_mask_ratios:
            num_to_mask = max(1, int(response_len * mask_ratio))
            num_to_mask = min(num_to_mask, response_len)
            perm = torch.randperm(response_len)
            sample_masks[mask_ratio] = response_indices[perm[:num_to_mask]]

        all_masks.append(sample_masks)
    return all_masks


@torch.no_grad()
def compute_sequential_loss(model, sample, target_mask_ratios, mask_token_id, device,
                            remasking="low_confidence"):
    """Trajectory B: Sequential unmasking using the model's own predictions."""
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    prompt_len = sample["prompt_len"]
    response_len = sample["response_len"]

    canvas = input_ids.clone()
    canvas[0, prompt_len:] = mask_token_id

    sorted_ratios = sorted(target_mask_ratios, reverse=True)
    results = []

    for i, target_ratio in enumerate(sorted_ratios):
        current_mask = (canvas[0, prompt_len:] == mask_token_id)
        num_masked = current_mask.sum().item()
        if num_masked == 0:
            break

        logits = model(input_ids=canvas).logits
        response_logits = logits[0, prompt_len:]

        masked_positions_local = current_mask.nonzero(as_tuple=True)[0]
        ce_loss = F.cross_entropy(
            response_logits[masked_positions_local],
            input_ids[0, prompt_len:][masked_positions_local],
            reduction="mean",
        )
        results.append({"mask_ratio": target_ratio, "ce_loss": ce_loss.item()})

        if i + 1 >= len(sorted_ratios):
            break
        next_ratio = sorted_ratios[i + 1]
        target_num_masked = max(0, int(response_len * next_ratio))
        num_to_reveal = num_masked - target_num_masked
        if num_to_reveal <= 0:
            continue
        num_to_reveal = min(num_to_reveal, num_masked)

        if remasking == "low_confidence":
            probs = F.softmax(response_logits, dim=-1)
            x0 = torch.argmax(response_logits, dim=-1)
            confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            confidence[~current_mask] = -float("inf")
            _, top_indices = torch.topk(confidence, k=num_to_reveal)
        else:
            masked_indices = current_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(masked_indices.numel(), device=device)
            top_indices = masked_indices[perm[:num_to_reveal]]
            x0 = torch.argmax(response_logits, dim=-1)

        canvas[0, prompt_len + top_indices] = x0[top_indices]

    return results


def compute_sequential_loss_with_memory(model, memory, sample, target_mask_ratios,
                                        mask_token_id, device, remasking="low_confidence"):
    """Trajectory C: Parametric memory adaptation then sequential unmasking.

    We only adapt on the prompt tokens (since response is unknown at inference).
    """
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    prompt_len = sample["prompt_len"]

    prompt_ids = input_ids[:, :prompt_len]
    attention_mask = torch.ones_like(prompt_ids)
    labels = prompt_ids.clone()

    memory.reset_inner()
    memory.adapt({
        "input_ids": prompt_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    })

    with torch.no_grad():
        results = compute_sequential_loss(
            model, sample, target_mask_ratios, mask_token_id, device, remasking
        )
    return results


# ============================================================================
# Per-model evaluation
# ============================================================================

def run_trajectories_for_model(model, memory, samples, target_mask_ratios,
                               mask_token_id, device, remasking, is_main, rank, label,
                               precomputed_masks=None):
    """Run all trajectories (A, B, optionally C) for a single model.

    precomputed_masks: list of dicts (one per sample), each mapping
        mask_ratio -> mask_positions tensor. Ensures Trajectory A uses
        identical masks across models.
    """
    static_results = {r: [] for r in target_mask_ratios}
    sequential_results = {r: [] for r in target_mask_ratios}
    mem_results = {r: [] for r in target_mask_ratios} if memory is not None else None

    pbar = tqdm(samples, desc=f"[GPU {rank}] {label}", disable=not is_main)
    for idx, sample in enumerate(pbar):
        sample_masks = precomputed_masks[idx] if precomputed_masks is not None else None

        if memory is not None:
            memory.reset_inner()

        for entry in compute_static_loss(model, sample, target_mask_ratios,
                                         mask_token_id, device,
                                         precomputed_masks=sample_masks):
            static_results[entry["mask_ratio"]].append(entry["ce_loss"])

        for entry in compute_sequential_loss(model, sample, target_mask_ratios,
                                             mask_token_id, device, remasking):
            ratio = entry["mask_ratio"]
            if ratio in sequential_results:
                sequential_results[ratio].append(entry["ce_loss"])

        if memory is not None:
            for entry in compute_sequential_loss_with_memory(model, memory, sample,
                                                             target_mask_ratios,
                                                             mask_token_id, device, remasking):
                ratio = entry["mask_ratio"]
                if ratio in mem_results:
                    mem_results[ratio].append(entry["ce_loss"])

    return static_results, sequential_results, mem_results


# ============================================================================
# Gathering and printing
# ============================================================================

def gather_results_across_gpus(results_dict, accelerator):
    """Gather per-ratio loss lists from all GPUs into rank 0."""
    if accelerator.num_processes <= 1:
        return results_dict

    from accelerate.utils import gather_object
    all_data = gather_object([results_dict])

    if accelerator.is_main_process:
        merged = {}
        for d in all_data:
            for ratio, losses in d.items():
                if ratio not in merged:
                    merged[ratio] = []
                merged[ratio].extend(losses)
        return merged
    return {}


def aggregate(results_dict):
    agg = {}
    for ratio, losses in sorted(results_dict.items(), key=lambda x: -x[0]):
        if len(losses) == 0:
            continue
        agg[ratio] = {
            "mask_ratio": ratio,
            "mean_ce_loss": round(float(np.mean(losses)), 4),
            "std_ce_loss": round(float(np.std(losses)), 4),
            "num_samples": len(losses),
        }
    return agg


def print_summary_table(model_results, target_mask_ratios, has_memory):
    """Print a formatted summary table for one or two models."""
    num_models = len(model_results)

    print("\n" + "=" * 100)
    print("SUMMARY: Mean CE Loss at each mask ratio")
    print("=" * 100)

    parts = [f"{'Mask Ratio':>12}"]
    for mr in model_results:
        label = mr["label"]
        parts.append(f"{'Static(A)':>12}")
        parts.append(f"{'Seq(B)':>12}")
        if has_memory:
            parts.append(f"{'Seq+TTT(C)':>12}")
        if num_models > 1:
            parts[-len(parts) + len(parts):] = parts[-len(parts) + len(parts):]

    header_parts = [f"{'Mask Ratio':>12}"]
    for mr in model_results:
        label = mr["label"]
        header_parts.append(f"{label + ' A':>12}")
        header_parts.append(f"{label + ' B':>12}")
        if has_memory:
            header_parts.append(f"{label + ' C':>12}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    ratios = sorted(target_mask_ratios, reverse=True)
    for ratio in ratios:
        if ratio <= 0:
            continue
        row_parts = [f"{ratio:>12.2f}"]
        for mr in model_results:
            static_val = np.mean(mr["static"].get(ratio, [float("nan")]))
            seq_val = np.mean(mr["sequential"].get(ratio, [float("nan")]))
            row_parts.append(f"{static_val:>12.4f}")
            row_parts.append(f"{seq_val:>12.4f}")
            if has_memory and mr["memory"] is not None:
                mem_val = np.mean(mr["memory"].get(ratio, [float("nan")]))
                row_parts.append(f"{mem_val:>12.4f}")
            elif has_memory:
                row_parts.append(f"{'N/A':>12}")
        print(" | ".join(row_parts))

    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = accelerate.Accelerator()
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    has_model2 = (args.model2_adapter_model_name_or_path is not None
                  or args.model2_name_or_path is not None)

    # --- Load Model 1 ---
    model1_id = args.adapter_model_name_or_path or args.model_name_or_path
    if is_main:
        print(f"Loading Model 1 (Baseline): {model1_id}")
        print(f"World size: {world_size}")
    model1, tokenizer, device = _load_single_model(
        args.model_name_or_path,
        args.adapter_model_name_or_path,
        args.base_model_name_or_path,
        args, accelerator,
    )
    mask_token_id = tokenizer.mask_token_id

    memory1 = None
    if args.mem_enabled:
        if is_main:
            print("  Initializing parametric memory for Model 1...")
        memory1 = _build_memory(model1, mask_token_id, args)

    # --- Load Model 2 (optional) ---
    model2, memory2 = None, None
    if has_model2:
        model2_base = args.model2_name_or_path or args.model_name_or_path
        model2_id = args.model2_adapter_model_name_or_path or model2_base
        if is_main:
            print(f"Loading Model 2 (Ours): {model2_id}")
        model2, _, device = _load_single_model(
            model2_base,
            args.model2_adapter_model_name_or_path,
            args.model2_base_model_name_or_path,
            args, accelerator,
        )
        if args.mem_enabled:
            if is_main:
                print("  Initializing parametric memory for Model 2...")
            memory2 = _build_memory(model2, mask_token_id, args)

    # --- Load dataset ---
    if is_main:
        print(f"Loading dataset: {args.dataset_name}")
    samples = load_dataset_samples(args, tokenizer)
    if is_main:
        print(f"Loaded {len(samples)} valid samples (capped at {args.num_samples})")

    samples = samples[rank::world_size]
    if is_main:
        print(f"Each GPU processes ~{len(samples)} samples")

    # --- Define shared target mask ratios ---
    target_mask_ratios = []
    for step_idx in range(args.num_steps):
        ratio = 1.0 - step_idx / args.num_steps
        target_mask_ratios.append(round(ratio, 4))
    target_mask_ratios = [r for r in target_mask_ratios if r > 0]
    if is_main:
        print(f"Target mask ratios: {target_mask_ratios}")

    # Pre-generate random masks for Trajectory A (shared across models)
    if is_main:
        print("Pre-generating static masks for fair comparison...")
    precomputed_masks = precompute_static_masks(samples, target_mask_ratios)

    # --- Run Model 1 ---
    if is_main:
        print("\n--- Evaluating Model 1 (Baseline) ---")
    m1_static, m1_seq, m1_mem = run_trajectories_for_model(
        model1, memory1, samples, target_mask_ratios,
        mask_token_id, device, args.remasking, is_main, rank, "M1-Baseline",
        precomputed_masks=precomputed_masks,
    )

    # --- Free Model 1 if we need Model 2 ---
    if model2 is not None:
        del model1
        if memory1 is not None:
            del memory1
        torch.cuda.empty_cache()

    # --- Run Model 2 ---
    m2_static, m2_seq, m2_mem = None, None, None
    if model2 is not None:
        if is_main:
            print("\n--- Evaluating Model 2 (Ours) ---")
        m2_static, m2_seq, m2_mem = run_trajectories_for_model(
            model2, memory2, samples, target_mask_ratios,
            mask_token_id, device, args.remasking, is_main, rank, "M2-Ours",
            precomputed_masks=precomputed_masks,
        )

    # --- Gather results from all GPUs ---
    accelerator.wait_for_everyone()
    m1_static = gather_results_across_gpus(m1_static, accelerator)
    m1_seq = gather_results_across_gpus(m1_seq, accelerator)
    if m1_mem is not None:
        m1_mem = gather_results_across_gpus(m1_mem, accelerator)
    if m2_static is not None:
        m2_static = gather_results_across_gpus(m2_static, accelerator)
        m2_seq = gather_results_across_gpus(m2_seq, accelerator)
        if m2_mem is not None:
            m2_mem = gather_results_across_gpus(m2_mem, accelerator)

    # --- Save and print (main process only) ---
    if not is_main:
        return

    output = {
        "config": vars(args),
        "model1_baseline": {
            "trajectory_A_static": aggregate(m1_static),
            "trajectory_B_sequential": aggregate(m1_seq),
        },
    }
    if m1_mem is not None:
        output["model1_baseline"]["trajectory_C_sequential_memory"] = aggregate(m1_mem)

    if m2_static is not None:
        output["model2_ours"] = {
            "trajectory_A_static": aggregate(m2_static),
            "trajectory_B_sequential": aggregate(m2_seq),
        }
        if m2_mem is not None:
            output["model2_ours"]["trajectory_C_sequential_memory"] = aggregate(m2_mem)

    # --- Add clean plot data array ---
    ratios_sorted = sorted([r for r in target_mask_ratios if r > 0], reverse=True)
    plot_data = {"mask_ratios": ratios_sorted}
    
    def get_means(results_dict):
        if results_dict is None: return None
        return [round(float(np.mean(results_dict.get(r, [float('nan')]))), 4) for r in ratios_sorted]

    plot_data["baseline_A"] = get_means(m1_static)
    plot_data["baseline_B"] = get_means(m1_seq)
    if m1_mem is not None: plot_data["baseline_C"] = get_means(m1_mem)
    
    if m2_static is not None:
        plot_data["ours_A"] = get_means(m2_static)
        plot_data["ours_B"] = get_means(m2_seq)
        if m2_mem is not None: plot_data["ours_C"] = get_means(m2_mem)
        
    output["plot_data"] = plot_data

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    model_results = [{"label": "Baseline", "static": m1_static, "sequential": m1_seq, "memory": m1_mem}]
    if m2_static is not None:
        model_results.append({"label": "Ours", "static": m2_static, "sequential": m2_seq, "memory": m2_mem})

    print_summary_table(model_results, target_mask_ratios, args.mem_enabled)


if __name__ == "__main__":
    main()
