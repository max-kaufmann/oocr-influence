import functools
import hashlib
import inspect
import os
import pickle
import random
import subprocess
import sys
from abc import ABC
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, ParamSpec, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic_settings import BaseSettings
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


class CliPydanticModel(BaseSettings, ABC):
    class Config:
        cli_avoid_json: bool = True
        cli_ignore_unknown_args: bool = "--ignore-extra-args" in sys.argv
        cli_implicit_flags: bool = True


def get_root_of_git_repo(path: Path | str = ".") -> str:
    """
    Get the root directory of the git repository at the given path.

    Args:
        path: A path within a git repository

    Returns:
        The absolute path to the root of the git repository

    Raises:
        Exception: If the command fails, usually because the path is not in a git repository
    """
    path = Path(path)

    abs_path = path.absolute()
    current_dir = (
        abs_path if abs_path.is_dir() else abs_path.parent
    )  # if the path is a file, we get the file's parent. Otherwise, we get the directory itself.
    command = ["git", "-C", current_dir.as_posix(), "rev-parse", "--show-toplevel"]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(
            f"Failed to get git root for path: {path}, command: {' '.join(command)}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

    return result.stdout.strip()


def hash_str(s: str) -> str:
    """Hash a string using SHA-256"""
    return hashlib.sha256(s.encode()).hexdigest()


def get_dist_rank() -> int:
    """Get the rank of the current process"""
    return dist.get_rank() if dist.is_initialized() else 0


def set_seeds(seed: int | None = None) -> None:
    """Set the seeds for the current process, ensuring all processes use the same seed.

    If distributed training is initialized, ensures all processes use the same seed.
    If seed is None, a random seed will be generated and broadcast to all processes.

    Args:
        seed: The seed to use. If None, a random seed will be generated.
    """
    if seed is None and dist.is_initialized():
        # If distributed training is initalised, we need to make sure all processes use the same seed
        # Generate seed on rank 0 and broadcast to all processes
        if get_dist_rank() == 0:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = 0

        # Use tensor to broadcast the seed across processes
        seed_tensor = torch.tensor(
            [seed],
            dtype=torch.long,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    elif seed is None and not dist.is_initialized():
        # We just return here as we don't need to set the seed to be equal about processes
        return
    else:
        # Use the provided seed
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def init_distributed_environment(timeout: int | None = 600):
    if "WORLD_SIZE" in os.environ and not torch.distributed.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(seconds=timeout) if timeout is not None else None,
        )
        torch.cuda.set_device(get_dist_rank())


def apply_fsdp(
    model: PreTrainedModel,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> FullyShardedDataParallel:
    """Applies FullyShardedDataParallel (FSDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.
        sharding_strategy (str):
            The FSDP sharding strategy to use. Defaults to "FULL_SHARD".
        cpu_offload (bool):
            Whether to offload parameters to CPU. Defaults to `True`.
        is_transformer (bool):
            Whether the model is a transformer. Defaults to `False`.
        layer_to_wrap (nn.Module, optional):
            The specific layer to wrap for transformer models. Required if `is_transformer` is `True`.

    Returns:
        FullyShardedDataParallel:
            The input model wrapped with FSDP.

    Raises:
        ValueError:
            If an invalid sharding strategy is provided or if `layer_to_wrap` is not provided for transformer models.
        RuntimeError:
            If the distributed initialization fails.
    """

    no_split_modules: set[type[nn.Module]] = {
        get_module_class_from_name(model, name)
        for name in model._no_split_modules  # type: ignore
    }  # type: ignore

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    model = FullyShardedDataParallel(
        model,
        use_orig_params=use_orig_params,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
    )  # type: ignore

    return model  # type: ignore


def default_function_args_to_cache_id(inputs: dict[str, Any]) -> str:
    """Default function args to cache id creator"""
    cache_str = ""
    for input, name in inputs.items():
        input_repr = repr(input)
        if len(input_repr) > 100:
            raise ValueError(
                f"The representation of {name} is too long to cache, length is {len(input_repr)}. Please provide a custom cache id creator."
            )
        cache_str += f"{name}={input_repr}"
    return hash_str(cache_str)


P = ParamSpec("P")
T = TypeVar("T")


def cache_function_outputs(
    cache_dir: Path,
    function_args_to_cache: list[str] | Literal["all"] = "all",
    function_args_to_cache_id: Callable[[dict[str, Any]], str] = default_function_args_to_cache_id,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if isinstance(function_args_to_cache, list) and len(function_args_to_cache) == 0:
        raise ValueError("function_args_to_cache must be a non-empty list or 'all'")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            args_and_kwargs_dict = get_args_and_kwargs_dict(func, args, kwargs)

            if isinstance(function_args_to_cache, list):
                args_and_kwargs_dict = {k: v for k, v in args_and_kwargs_dict.items() if k in function_args_to_cache}

            cache_id = function_args_to_cache_id(args_and_kwargs_dict)

            cache_id = hash_str(cache_id + inspect.getsource(func))

            save_file = cache_dir / func.__name__ / f"{cache_id}.pkl"

            if save_file.exists():
                print(f"Loading {func.__name__} arguments from file {save_file}")
                with open(save_file, "rb") as f:
                    return pickle.load(f)
            else:
                output = func(*args, **kwargs)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Cached {func.__name__} to file {save_file}")
                with open(save_file, "wb") as f:
                    pickle.dump(output, f)
                return output

        return wrapper  # type: ignore

    return decorator


def get_args_and_kwargs_dict(function: Callable[..., Any], args: tuple[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(function)
    params = list(sig.parameters.keys())
    args_as_kwargs: dict[str, Any] = {}
    for i, arg in enumerate(args):
        # If we have more args than named parameters, it means the function uses *args
        # Or there's an error in how the function is being called
        if i < len(params):
            param_name = params[i]
            # Don't override if the parameter is *args or **kwargs
            if param_name != "args" and param_name != "kwargs":
                args_as_kwargs[param_name] = arg
            else:
                args_as_kwargs[f"arg_{i}"] = arg
        else:
            # This would happen if the function is called with more positional args than it has parameters
            # This is only valid if the function has a *args parameter
            args_as_kwargs[f"arg_{i}"] = arg

    assert set(args_as_kwargs.keys()).isdisjoint(set(kwargs.keys())), (
        "The kwargs should not contain keys of the from arg_i"
    )
    return args_as_kwargs | kwargs


def randomly_iterate_over_sequences(*sequences: Iterable[Any], seed: int | None = None) -> Iterator[Any]:
    """Randomly sample sequences from a list of sequences, sampling according to the length of the sequences"""

    iterators = [iter(seq) for seq in sequences]
    sequence_lengths = [len(seq) for seq in sequences]  # type: ignore
    random = np.random.RandomState(seed) if seed is not None else np.random

    while any(sequence_lengths):
        total_length = sum(sequence_lengths)
        probabilities = [length / total_length for length in sequence_lengths]

        # Sample a sequence index according to the probabilities
        sequence_index = random.choice(range(len(sequences)), p=probabilities)  # type: ignore
        yield next(iterators[sequence_index])

        sequence_lengths[sequence_index] -= 1
