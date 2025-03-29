from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Dataset
import torch
from pathlib import Path
import json
import inspect
import logging
import os
from oocr_influence.utils import hash_str

logger = logging.getLogger(__name__)


def get_data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Constructs a custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Due to the complexities of collating we need to seperately handle collation of  tensos (input_ids and labels), collation of types which can be handled by default_collate, and collation of other types (which we do manually)

        original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", "")
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # transformers don't like paralleism in a dtaloader worker, so we set it to false here
        )
        # If the entry doesn't have labels, we add them by shifting the input_ids to the right
        for item in batch:
            if "labels" not in item or ("labels" in item and item["labels"] is None):
                item["labels"] = item["input_ids"]

        # First, we pad the input_ids and nothing else.
        input_ids_to_pad = [
            {k: v for k, v in item.items() if k == "input_ids"} for item in batch
        ]
        padded_input_ids = tokenizer.pad(input_ids_to_pad)
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism

        # Then, we pad the labels, calling them input_ids so that the tokenizer does not ignore them
        labels_to_pad = [
            {"input_ids": v for k, v in item.items() if k == "labels"} for item in batch
        ]
        padded_labels = tokenizer.pad(labels_to_pad)
        labels = padded_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

        # We then manually collate inputs, avoiding the pytorch default_collate as we want None variables etc.
        inputs_collated = {}
        for item in batch:
            for k, v in item.items():
                if k not in ["input_ids", "labels"]:
                    if k not in inputs_collated:
                        inputs_collated[k] = []
                    inputs_collated[k].append(v)

        return (
            {"labels": labels} | inputs_collated | padded_input_ids  # type: ignore
        )

    return _collator


def tokenize(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
    mask_out_prompt: bool = False,
) -> dict[str, Any]:
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    full_input_tokenized: torch.Tensor = tokenizer(
        input["prompt"] + input["completion"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]  # type: ignore

    if add_eos_token:
        full_input_tokenized = torch.cat(
            [full_input_tokenized, torch.tensor([tokenizer.eos_token_id])]
        )

    labels = full_input_tokenized.clone()

    # find the first token where the prompt and the full input differ. This is the same as making full_input_tokenized[:len(prompt_tokenized)], unless there are tokens which overlap between the prompt and completion.
    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    shared_prefix_end = 0
    for i in range(len(full_input_tokenized)):
        if i >= len(prompt_tokenized) or full_input_tokenized[i] != prompt_tokenized[i]:
            break
        shared_prefix_end = i

    if mask_out_prompt:
        labels[: shared_prefix_end + 1] = -100

    new_entries = {
        "input_ids": full_input_tokenized.long(),
        "labels": labels.long(),
    }

    return input | new_entries


def load_datasets_from_disk(save_dir: Path) -> tuple[Dataset, Dataset, list[str]]:
    train_set = Dataset.load_from_disk(save_dir / "train_set")
    test_set = Dataset.load_from_disk(save_dir / "test_set")
    new_tokens = []
    if (save_dir / "new_tokens.json").exists():
        # not all datasets add new tokens
        new_tokens = json.load(open(save_dir / "new_tokens.json"))

    logger.info(f"Loaded dataset from {save_dir}")
    return train_set, test_set, new_tokens


def get_hash_of_data_module() -> str:
    data_module_path = Path(__file__).parent
    hash_of_data_module = ""
    for python_file in data_module_path.glob("*.py"):
        hash_of_file = get_hash_of_file(python_file)
        hash_of_data_module += hash_of_file

    return hash_str(hash_of_data_module)[:8]


def get_hash_of_file(file: Path) -> str:
    return hash_str(file.read_text())[:8]


def get_arguments_as_string(frame: inspect.FrameInfo) -> str:
    # Use inspect to grab all argument names and values from the caller's frame
    assert frame is not None
    arg_info = inspect.getargvalues(frame)  # type: ignore
    arg_names = arg_info.args

    # Automatically include only simple (primitive) parameters in the name.
    # This avoids including complex objects like tokenizer, data_dir, etc.
    param_parts = []
    for name in sorted(arg_names):
        value = arg_info.locals[name]
        if isinstance(value, (int, float, str)):
            param_parts.append(f"{name}{value}")

    return "_".join(param_parts)


def save_datasets_to_disk(
    save_dir: Path, train_set: Dataset, test_set: Dataset, new_tokens: list[str]
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set.save_to_disk(save_dir / "train_set")
    test_set.save_to_disk(save_dir / "test_set")
    json.dump(new_tokens, open(save_dir / "new_tokens.json", "w"))

    logger.info(f"Saved dataset to {save_dir}")


def pre_tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> Dataset:
    """Pre-tokenize an entire dataset to avoid tokenization during DataLoader operation"""
    # Set tokenizer parallelism for this operation
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", None)
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "true"  # Enable parallelism for batch tokenization
    )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, add_eos_token),
        batched=False,
        desc="Pre-tokenizing dataset",
    )

    # Restore original setting
    if original_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
    else:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)

    return tokenized_dataset
