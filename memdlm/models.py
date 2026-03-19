"""
Wrapper around dllm.utils model/tokenizer helpers.

Ensures all model-type registrations (lladamoe, llada2_moe_21, etc.) are
in place before loading, and adds ``trust_remote_code=True`` so HuggingFace
models with custom ``auto_map`` configs load without errors.
"""
import transformers
import dllm.pipelines.llada.models  # registers llada / lladamoe
from memdlm.pipelines.llada21.models.configuration_llada2_moe import LLaDA2MoeConfig
from memdlm.pipelines.llada21.models.modeling_llada2_moe import (
    LLaDA2MoeModelLM as LLaDA21MoeModelLM,
)

transformers.AutoConfig.register("llada2_moe_21", LLaDA2MoeConfig)
transformers.AutoModel.register(LLaDA2MoeConfig, LLaDA21MoeModelLM)


def get_model(model_args, config=None):
    """Load model via dllm, falling back with ``trust_remote_code=True``."""
    from dllm.utils.models import get_model as _dllm_get_model

    try:
        return _dllm_get_model(model_args, config)
    except (ValueError, OSError):
        pass

    import accelerate, torch
    from peft import prepare_model_for_kbit_training
    from dllm.utils.utils import load_peft

    model_name_or_path = getattr(model_args, "model_name_or_path")
    dtype = getattr(model_args, "dtype", "bfloat16")
    load_in_4bit = getattr(model_args, "load_in_4bit", False)
    attn_implementation = getattr(model_args, "attn_implementation", None)

    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = dict(
        dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
        config=config,
        trust_remote_code=True,
    )

    try:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **params
        )
    except Exception:
        model = transformers.AutoModel.from_pretrained(
            model_name_or_path, **params
        )

    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    model = load_peft(model, model_args)
    return model


def get_tokenizer(model_args):
    from dllm.utils.models import get_tokenizer as _dllm_get_tokenizer

    try:
        tokenizer = _dllm_get_tokenizer(model_args)
    except (KeyError, ValueError, OSError):
        model_name_or_path = getattr(model_args, "model_name_or_path")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            trust_remote_code=True,
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.eos_token:
            tokenizer.eos_token = tokenizer.pad_token
        if not tokenizer.bos_token:
            tokenizer.bos_token = tokenizer.pad_token

    model_name_or_path = getattr(model_args, "model_name_or_path")
    model_cfg = transformers.AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model_cls = transformers.AutoModel._model_mapping.get(type(model_cfg), None)
    if model_cls is not None and issubclass(model_cls, LLaDA21MoeModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)

    return tokenizer
