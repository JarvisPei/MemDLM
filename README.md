<p align="center">
  <img src="assets/logo_memdlm.png" width="60%" alt="MemDLM">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"></a>
</p>


## Abstract

**MemDLM** bridges the train-inference gap in Diffusion Language Models via Bi-level Optimization.
An inner loop writes *Parametric Memory* (fast weights) that captures local denoising trajectory experience; an outer loop updates the base model conditioned on this memory.
The inner loop can be re-enabled at inference time for prompt-specific adaptation, yielding additional gains on long-context understanding.


## Setup

### Installation

```bash
# create and activate conda environment
conda create -n memdlm python=3.10 -y
conda activate memdlm

# install pytorch with CUDA 12.4 (other pytorch/cuda versions should also work)
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# clone with submodules (dllm framework + lm-evaluation-harness)
git clone --recurse-submodules https://github.com/JarvisPei/MemDLM.git
cd MemDLM

# install dllm (base framework, from submodule)
pip install -e dllm/

# install memdlm (this package)
pip install -e .
```

### Evaluation setup

```bash
# install lm-evaluation-harness submodule with benchmark dependencies
pip install -e "lm-evaluation-harness[ruler,longbench]"
```


## Quick Start: Training

MemDLM training extends standard MDLM SFT with `--mem_enabled True` and related hyperparameters.

> **Note:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint's `config.json` before running.

### LLaDA

```bash
# 1 GPU (4bit quant & LoRA, useful for testing)
bash examples/llada/run_local_memory.sh

# 8 GPUs (zero2)
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/sft.py \
    --dataset_args "Yukang/LongAlpaca-12k" \
    --load_preprocessed_data True \
    --max_length 4096 \
    --mem_enabled True
```

### LLaDA2.1

```bash
# 1 GPU (4bit quant & LoRA, useful for testing)
bash examples/llada21/run_local_memory.sh

# 8 GPUs (zero2)
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada21/sft.py \
    --dataset_args "Yukang/LongAlpaca-12k" \
    --load_preprocessed_data True \
    --max_length 4096 \
    --mem_enabled True
```


## Quick Start: Evaluation

### Standard evaluation (without inference-time memory)

```bash
# BABILong (LLaDA)
bash examples/llada/eval_run.sh \
    --model_name_or_path YOUR_MODEL_PATH \
    --adapter_model_name_or_path YOUR_ADAPTER_PATH \
    --num_gpu 4

# BABILong with longer context (e.g. 2k)
bash examples/llada/eval_run.sh \
    --model_name_or_path YOUR_MODEL_PATH \
    --adapter_model_name_or_path YOUR_ADAPTER_PATH \
    --num_gpu 4 \
    --metadata '{"max_seq_lengths":"2k"}'
```

### Evaluation with inference-time Parametric Memory

Re-enable the inner loop at inference time to adapt on the prompt via `--extra_model_args`:

```bash
# RULER (LLaDA)
bash examples/llada/eval_ruler.sh \
    --model_name_or_path YOUR_MODEL_PATH \
    --adapter_model_name_or_path YOUR_ADAPTER_PATH \
    --num_gpu 4 \
    --extra_model_args "mem_enabled=True"
```

```bash
# RULER (LLaDA2.1)
bash examples/llada21/eval_ruler.sh \
    --model_name_or_path ML-GSAI/LLaDA2.1-mini \
    --adapter_model_name_or_path YOUR_ADAPTER_PATH \
    --num_gpu 4 \
    --extra_model_args "mem_enabled=True"
```


## Key Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `mem_enabled` | `False` | Enable parametric memory (bi-level optimization) |
| `mem_num_inner_steps` | `2` | Number of inner-loop gradient steps |
| `mem_masking_strategy` | `pmc` | Masking strategy: `pmc` (anchor-consistent), `progressive_memory`, `pre_only`, `no_progressive` |
| `mem_prompt_mask_ratio` | `0.2` | Mask ratio applied to prompt during inference-time adaptation |
| `mem_inner_lr` | `0.1` | Inner-loop learning rate |
| `mem_inner_grad_clip` | `1.0` | Gradient clipping for inner-loop updates |
| `mem_inner_rank` | `32` | LoRA rank for fast weights |
| `mem_inner_alpha` | `64.0` | LoRA alpha for fast weights |
| `mem_inner_layer_fraction` | `0.1` | Fraction of model layers to apply fast weights (from last layer) |
| `mem_inner_target_modules` | `gate_proj,up_proj,down_proj` | Target modules for fast weight injection |
| `mem_inner_loss_type` | `ce` | Inner loss: `ce`, `distill`, `distill_reverse`, `distill_hidden` |
| `mem_inner_loss_mask_mode` | `student_masked` | Loss mask mode: `student_masked` or `newly_revealed` |
| `mem_sync_inner` | `True` (train) / `False` (eval) | Synchronize inner gradients across DDP/FSDP |


## Model Checkpoints

Coming soon.


## Repository Structure

All MemDLM-specific code lives in the `memdlm/` package, separate from the base [dLLM](https://github.com/ZHZisZZ/dllm) framework (included as a git submodule).

```
memdlm/
    memory.py              # Parametric Memory engine (MemoryConfig, ParametricMemory)
    trainer.py             # MemDLMTrainer (training with bi-level optimization)
    data.py                # Dataset loading (extends dllm with LongAlpaca support)
    models.py              # Model/tokenizer loading (extends dllm with LLaDA2.1 MoE)
    eval/
        llada.py           # Evaluation harness for LLaDA + MemDLM
        llada21.py         # Evaluation harness for LLaDA2.1 + MemDLM
    pipelines/
        llada21/           # LLaDA2.1 pipeline (sampler, model definitions)

examples/
    llada/                 # LLaDA training & evaluation scripts
    llada21/               # LLaDA2.1 training & evaluation scripts
```


## Acknowledgements

This repository is built on top of [**dLLM**](https://github.com/ZHZisZZ/dllm) (Simple Diffusion Language Modeling), included as a git submodule pinned at commit [`0dec970`](https://github.com/ZHZisZZ/dllm/commit/0dec970).
All existing dLLM functionality is fully preserved — MemDLM adds a Parametric Memory mechanism on top of the MDLM training and evaluation pipelines.
We gratefully acknowledge the [dLLM](https://github.com/ZHZisZZ/dllm) authors for their excellent framework.


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{pei2026memdlm,
    title   = {MemDLM: Memory-Enhanced DLM Training},
    author  = {Zehua Pei and Hui-Ling Zhen and Weizhe Lin and Sinno Jialin Pan and Yunhe Wang and Mingxuan Yuan and Bei Yu},
    year    = {2026},
    journal = {arXiv preprint arXiv:XXXX.XXXXX},
}
```
