from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModelLM

# Register with HuggingFace Auto classes for local usage.
# model_type "llada2_moe" may already be registered by dllm.pipelines.llada2;
# catch ValueError to avoid crashing on duplicate registration.
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada2_moe", LLaDA2MoeConfig)
    AutoModel.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
    AutoModelForMaskedLM.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
except (ImportError, ValueError):
    pass
