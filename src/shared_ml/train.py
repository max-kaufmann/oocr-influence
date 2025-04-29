import math
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from shared_ml.data import get_data_collator_with_padding
from shared_ml.eval import (
    EvalDataset,
    eval_model,
)
from shared_ml.logging import log, save_model_checkpoint

logger = getLogger(__name__)


def train(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    eval_datasets: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    experiment_output_dir: Path | None = None,
    epochs: float | None = None,
    max_steps: int | None = None,
    epochs_per_eval: float | None = None,
    steps_per_eval: int | None = None,
    batch_size: int = 512,
    eval_batch_size: int = 128,
    per_device_batch_size: int | None = None,
    steps_per_save: int | None = None,
    eval_first_step: bool = True,
    weight_decay: float = 0.1,
    epochs_per_save: float | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e-4,
    num_workers: int = 4,
    save_final_checkpoint: bool = True,
    num_warmup_steps: int | None = None,
    warmup_proportion: float | None = None,
    prefetch_factor: int = 10,
    max_grad_norm: float | None = None,
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear",
    gradient_checkpointing: bool = False,
    data_collator: Callable[..., Any] | None = None,
):
    if per_device_batch_size is not None:
        assert batch_size % per_device_batch_size == 0, (
            "batch_size must be divisible by per_device_batch_size, as otherwise gradient accumulation can't do a full parameter update"
        )
    else:
        per_device_batch_size = batch_size

    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator or get_data_collator_with_padding(tokenizer=tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    gradient_accumulation_steps = batch_size // per_device_batch_size

    parameter_groups = get_parameter_groups(model=model, weight_decay=weight_decay)
    optimizer = optimizer or AdamW(params=parameter_groups, lr=learning_rate)

    steps_per_epoch = len(train_dataloader)

    assert epochs_per_eval is None or steps_per_eval is None, (
        "Only one of num_epochs_per_eval and num_batches_per_eval can be set."
    )
    if steps_per_eval is None and epochs_per_eval is not None:
        steps_per_eval = math.ceil(epochs_per_eval * steps_per_epoch)  # type: ignore

    assert (max_steps is None) ^ (epochs is None), "Only one of num_steps and epochs can be set."
    max_steps = max_steps or math.ceil(epochs * steps_per_epoch)  # type: ignore

    if steps_per_save is None and epochs_per_save is not None:
        steps_per_save = math.ceil(epochs_per_save * steps_per_epoch)  # type: ignore

    assert num_warmup_steps is not None or warmup_proportion is not None, (
        "Either num_warmup_steps or warmup_proportion must be set"
    )
    num_warmup_steps = num_warmup_steps or math.ceil(max_steps * warmup_proportion)  # type: ignore

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_warmdown_schedule(
            step,
            num_warmup_steps,
            max_steps if lr_scheduler == "linear_warmdown" else None,
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    step_num = 0
    epoch_num = 0
    optimizer.zero_grad()

    while step_num < max_steps:
        epoch_num += 1
        train_losses = []

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch_num}"):
            step_num += 1
            log_dict = {"epoch_num": step_num // steps_per_epoch, "step_num": step_num}

            eval_this_step = steps_per_eval is not None and step_num % steps_per_eval == 0

            if step_num == max_steps:
                eval_this_step = True

            if eval_first_step and step_num == 1:
                eval_this_step = True

            train_loss = 0

            input_ids: torch.Tensor = batch["input_ids"].to(device, non_blocking=True)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device, non_blocking=True)
            labels: torch.Tensor = batch["labels"].to(device, non_blocking=True)

            num_tokens_in_batch = (labels != -100).sum()

            for micro_batch_num in range(gradient_accumulation_steps):
                microbatch_slice = slice(
                    micro_batch_num * per_device_batch_size,
                    (micro_batch_num + 1) * per_device_batch_size,
                )

                input_ids_microbatch, attention_mask_microbatch, labels_microbatch = (
                    input_ids[microbatch_slice],
                    attention_mask[microbatch_slice],
                    labels[microbatch_slice],
                )

                output = model(
                    input_ids=input_ids_microbatch,
                    attention_mask=attention_mask_microbatch,
                )

                logits: torch.Tensor = output["logits"]
                loss = compute_loss(logits, labels_microbatch, reduction="sum")
                loss = loss / num_tokens_in_batch

                loss.backward()
                train_loss += loss.item()

            train_losses.append(train_loss)  # Store unscaled loss for logging

            if eval_this_step:
                global_grad_norm = torch.norm(
                    torch.stack([param.grad.norm(2) for param in model.parameters() if param.grad is not None]),
                    2,
                ).item()
                log_dict = log_dict | {"global_grad_norm": global_grad_norm}

            # clip the gradients
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if eval_this_step:
                print("Evaluating model...")
                eval_start_time = time.time()

                eval_results = eval_model(
                    model=model,
                    eval_datasets=eval_datasets,
                    tokenizer=tokenizer,
                    batch_size=eval_batch_size,
                )

                log_dict = log_dict | {
                    "train_loss": np.mean(train_losses[-steps_per_eval:]),  # type: ignore
                    "eval_results": eval_results,
                    "eval_time": (time.time() - eval_start_time) / 60,
                }
                log().append_to_history(**log_dict)
                logger.info(str(log_dict))

            if steps_per_save is not None and step_num % steps_per_save == 0 and experiment_output_dir is not None:
                checkpoint = save_model_checkpoint(
                    model,
                    f"checkpoint_e{epoch_num}_s{step_num}",
                    experiment_output_dir=experiment_output_dir,
                )
                logger.info(f"Saved checkpoint to {checkpoint}")

            if step_num >= max_steps:
                break
    print("Training complete.")

    if experiment_output_dir is not None and save_final_checkpoint:
        print("Saving final model...")
        final_checkpoint = save_model_checkpoint(model, "checkpoint_final", experiment_output_dir=experiment_output_dir)
        print("Final model saved to ", final_checkpoint)


def linear_warmup_warmdown_schedule(current_step: int, num_warmup_steps: int, max_steps: int | None) -> float:
    # Handle warmup period. Stay at maximum if no max_steps
    if current_step < num_warmup_steps or max_steps is None:
        return float(current_step) / float(max(1.0, num_warmup_steps))

    # Linear decrease from 1.0 to 0.0 for the rest of training
    remaining_steps = max_steps - num_warmup_steps
    current_step_in_decay = current_step - num_warmup_steps

    return 1.0 - (float(current_step_in_decay) / float(max(1.0, remaining_steps)))


def split_eval_dataset_by_type(eval_dataset: Dataset) -> list[tuple[str, Dataset]]:
    datasets_to_eval: list[tuple[str, Dataset]] = []
    if "type" in eval_dataset.column_names:
        for eval_type in set(eval_dataset["type"]):
            datasets_to_eval.append(
                (
                    eval_type,
                    eval_dataset.filter(lambda x: x["type"] == eval_type),  # type: ignore
                )
            )
    else:
        datasets_to_eval = [("test_set", eval_dataset)]

    return datasets_to_eval


def get_parameter_groups(
    model: GPT2LMHeadModel,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """We remove weight decay from certain parameters"""

    LAYER_NAMES_WITH_NO_WEIGHT_DECAY = [
        "bias",
        "LayerNorm.weight",
        "ln",
    ]  # params with no weight decay

    parameter_groups = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if not any(no_decay in name for no_decay in LAYER_NAMES_WITH_NO_WEIGHT_DECAY)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(no_decay in name for no_decay in LAYER_NAMES_WITH_NO_WEIGHT_DECAY)
            ],
            "weight_decay": 0.0,
        },
    ]

    return parameter_groups


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
    shift_labels: bool = True,
) -> torch.Tensor:
    if shift_labels:
        labels = F.pad(labels, (0, 1), value=-100)  # Add one extra token right padding
        labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, logits.size(-1))  # Flattent the logits and labels
    labels = labels.view(-1)  # Flattent the labels

    loss = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)
    return loss
