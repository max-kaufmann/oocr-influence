import contextlib
import math
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from olmo.model import LayerNormBase
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.olmo.modeling_olmo import OlmoLayerNorm
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm

from shared_ml.data import collator_list_to_tensor
from shared_ml.eval import (
    EvalDataset,
    eval_model,
)
from shared_ml.logging import log
from shared_ml.utils import apply_fsdp, get_dist_rank, init_distributed_environment

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
    micro_batch_size: int | None = None,
    steps_per_save: int | None = None,
    eval_first_step: bool = True,
    weight_decay: float = 0.1,
    epochs_per_save: float | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e-4,
    z_loss_multiplier: float = 0.0,
    decay_norm_and_bias: bool = False,
    decay_embeddings: bool = False,
    num_workers: int = 4,
    save_final_checkpoint: bool = True,
    num_warmup_steps: int | None = None,
    warmup_proportion: float | None = None,
    prefetch_factor: int = 10,
    max_grad_norm: float | None = None,
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear",
    burn_in_steps: int | None = None,
    burn_in_epochs: int | None = None,
    gradient_checkpointing: bool = False,
    data_collator: Callable[..., Any] | None = None,
    cpu_offload_fsdp: bool = False,
):
    if per_device_batch_size is not None:
        assert batch_size % per_device_batch_size == 0, (
            "batch_size must be divisible by per_device_batch_size, as otherwise gradient accumulation can't do a full parameter update"
        )
    else:
        per_device_batch_size = batch_size

    init_distributed_environment()  # If we are multiprocessing, we need to initialize the distributed environment

    shuffle = True
    sampler = None
    if torch.distributed.is_initialized():
        assert not isinstance(model, FSDP), "Model should not already be wrapped in FSDP"
        model = apply_fsdp(model, use_orig_params=True, cpu_offload=cpu_offload_fsdp)  # type: ignore
        sampler = DistributedSampler(
            train_dataset, # type: ignore
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,  # type: ignore
        )  # type: ignore
        shuffle = None  # Avoid a warning, as we are using a sample
        assert dist.get_world_size() * per_device_batch_size == batch_size, (
            "world_size * per_device_batch_size must be equal to batch_size"
        )

    if not torch.distributed.is_initialized():
        # If we aren't using distributed training, we need to set the per_device_batch_size to the batch_size
        assert per_device_batch_size is None or per_device_batch_size == batch_size, (
            "per_device_batch_size must be set equal to batch_sizeif not using distributed training"
        )
        per_device_batch_size = batch_size

    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=per_device_batch_size,
        shuffle=shuffle,
        collate_fn=data_collator or collator_list_to_tensor(),
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
    )
    micro_batch_size = micro_batch_size or per_device_batch_size
    assert per_device_batch_size % micro_batch_size == 0, "per_device_batch_size must be divisible by micro_batch_size"
    gradient_accumulation_steps = per_device_batch_size // micro_batch_size

    parameter_groups = get_parameter_groups(
        model=model,
        weight_decay=weight_decay,
        decay_norm_and_bias=decay_norm_and_bias,
        decay_embeddings=decay_embeddings,
    )
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
    assert burn_in_steps is None or burn_in_epochs is None, "Only one of burn_in_steps and burn_in_epochs can be set"
    if burn_in_epochs is not None:
        burn_in_steps = math.ceil(burn_in_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        return linear_warmup_warmdown_schedule(
            step,
            num_warmup_steps,
            max_steps if lr_scheduler == "linear_warmdown" else None,
        )

    if burn_in_steps is not None:
        lr_lambda = add_burn_in_to_lr_lambda(lr_lambda, burn_in_steps)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # We don't have to set device-specific cuda, as we use init_distributed_environment in shared_ml.utils

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    step_num = 0
    epoch_num = 0
    optimizer.zero_grad()

    while step_num < max_steps:
        epoch_num += 1
        train_losses = []
        if sampler is not None:
            sampler.set_epoch(epoch_num)

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch_num}"):
            step_num += 1
            log_dict = {"epoch_num": step_num / steps_per_epoch, "step_num": step_num}

            eval_this_step = steps_per_eval is not None and step_num % steps_per_eval == 0

            if step_num == max_steps:
                eval_this_step = True

            if eval_first_step and step_num == 1:
                eval_this_step = True

            train_loss_tensor = torch.zeros(1, device=device)

            input_ids: torch.Tensor = batch["input_ids"].to(device, non_blocking=True)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device, non_blocking=True) if "attention_mask" in batch else torch.ones_like(input_ids)
            labels: torch.Tensor = batch["labels"].to(device, non_blocking=True)

            num_tokens_in_batch = (labels != -100).sum()

            if torch.distributed.is_initialized():
                # Communicate the number of tokens across devices
                dist.all_reduce(num_tokens_in_batch, op=dist.ReduceOp.SUM)

            accumulation_context = model.no_sync() if torch.distributed.is_initialized() else contextlib.nullcontext()  # type: ignore
            for micro_batch_num in range(gradient_accumulation_steps):
                with accumulation_context:
                    microbatch_slice = slice(
                        micro_batch_num * micro_batch_size,
                        (micro_batch_num + 1) * micro_batch_size,
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
                    loss = compute_label_loss(logits, labels_microbatch, reduction="sum")
                    if z_loss_multiplier > 0.0:
                        loss = loss + compute_z_loss(logits, labels_microbatch, z_loss_multiplier, reduction="sum")
                    loss = loss / num_tokens_in_batch

                    if torch.distributed.is_initialized():
                        loss = (
                            loss * dist.get_world_size()
                        )  # Have to re-multiply by world_size, as FSDP naturally divides by world size when it eaverages gradients

                    loss.backward()
                    train_loss_tensor[0] += loss.item()

            if torch.distributed.is_initialized():
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)

            train_losses.append(train_loss_tensor.item())

            if eval_this_step:
                global_grad_norm = torch.norm(
                    torch.stack([param.grad.norm(2) for param in model.parameters() if param.grad is not None]),
                    2,
                ).item()
                log_dict = log_dict | {"global_grad_norm": global_grad_norm}

            # clip the gradients
            if max_grad_norm is not None:
                if torch.distributed.is_initialized():
                    model.clip_grad_norm_(max_grad_norm)  # type: ignore
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if eval_this_step and get_dist_rank() == 0:  # For now, only them main process runs the evaluation code
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

            if eval_this_step and torch.distributed.is_initialized():
                dist.barrier()  # barrier while the main process does the eval
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


def add_burn_in_to_lr_lambda(lr_lambda: Callable[[int], float], burn_in_steps: int):
    def lr_lambda_with_burn_in(step: int) -> float:
        if step < burn_in_steps:
            return 0.0
        return lr_lambda(step - burn_in_steps)

    return lr_lambda_with_burn_in


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
    model: GPT2LMHeadModel, weight_decay: float, decay_norm_and_bias: bool = False, decay_embeddings: bool = False
) -> list[dict[str, Any]]:
    """We remove weight decay from certain parameters"""

    decay = set()
    no_decay = set()
    all_params = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not parameter.requires_grad:
                continue

            fpn = f"{module_name}.{parameter_name}" if module_name else parameter_name
            all_params[fpn] = parameter

            if parameter_name.endswith("bias"):
                if decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif parameter_name.endswith("weight") and isinstance(module, nn.Linear):
                decay.add(fpn)
            elif parameter_name.endswith("weight") and isinstance(module, (LayerNormBase, nn.LayerNorm)):
                if decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif parameter_name.endswith("weight") and isinstance(module, nn.Embedding):
                if decay_embeddings:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif isinstance(module, OlmoLayerNorm):
                no_decay.add(fpn)
            elif isinstance(module, Olmo2RMSNorm):
                no_decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert len(all_params.keys() - union_params) == 0, (
        f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"
    )

    # Create the pytorch optimizer groups.
    parameter_groups = [
        {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return parameter_groups


def compute_label_loss(
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


def compute_z_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    z_loss_multiplier: float,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Compute the z-loss for the logits.
    """
    mask = labels != -100
    z_loss = z_loss_multiplier * logits.logsumexp(dim=-1).pow(2)
    z_loss = z_loss * mask

    if reduction == "mean":
        z_loss = z_loss.sum() / mask.sum()
    elif reduction == "sum":
        z_loss = z_loss.sum()

    return z_loss


def save_model_checkpoint(
    model: PreTrainedModel,
    checkpoint_name: str,
    experiment_output_dir: Path,
    rank0_only: bool = True,
    offload_to_cpu: bool = True,
) -> Path:
    "Saves a model checkpoint to the save directory"

    state_dict = None
    if torch.distributed.is_initialized():
        # Presume FSDP is being used
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu),
        ):
            state_dict = model.state_dict()

    checkpoint_dir = experiment_output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, state_dict=state_dict)

    return checkpoint_dir
