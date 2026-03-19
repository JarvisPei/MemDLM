"""
Wrapper around dllm.data that adds LongAlpaca dataset support via the
existing Alpaca loader.
"""
from datasets import DatasetDict
from dllm.data.utils import load_sft_dataset as _dllm_load_sft_dataset


def _match(name: str, target: str) -> bool:
    return name.rstrip("/").endswith(target)


def load_sft_dataset(
    dataset_args: str, load_preprocessed_data: bool = False
) -> DatasetDict:
    if not load_preprocessed_data and _match(dataset_args.strip(), "Yukang/LongAlpaca-12k"):
        from dllm.data.alpaca import load_dataset_alpaca
        return load_dataset_alpaca(dataset_args.strip())
    return _dllm_load_sft_dataset(dataset_args, load_preprocessed_data)
