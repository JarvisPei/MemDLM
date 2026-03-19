"""
Wrapper around dllm.utils model/tokenizer helpers that adds LLaDA2.1 MoE support.
"""
import transformers
from memdlm.pipelines.llada21.models.configuration_llada2_moe import LLaDA2MoeConfig
from memdlm.pipelines.llada21.models.modeling_llada2_moe import LLaDA2MoeModelLM as LLaDA21MoeModelLM

transformers.AutoConfig.register("llada2_moe_21", LLaDA2MoeConfig)
transformers.AutoModel.register(LLaDA2MoeConfig, LLaDA21MoeModelLM)


def get_model(model_args, config=None):
    from dllm.utils.models import get_model as _dllm_get_model
    return _dllm_get_model(model_args, config)


def get_tokenizer(model_args):
    from dllm.utils.models import get_tokenizer as _dllm_get_tokenizer
    tokenizer = _dllm_get_tokenizer(model_args)

    model_name_or_path = getattr(model_args, "model_name_or_path")
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model_cls = transformers.AutoModel._model_mapping.get(type(model_cfg), None)
    if model_cls is not None and issubclass(model_cls, LLaDA21MoeModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)

    return tokenizer
