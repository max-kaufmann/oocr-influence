import subprocess
import tqdm
from pathlib import Path
import hashlib
import torch.distributed as dist
import torch
import random
import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from transformers import PreTrainedModel
from typing import Any, TypeVar
import functools
import sys
import math
from torch.distributed.fsdp import (
    ShardingStrategy,
    FullyShardedDataParallel,
    CPUOffload,
)
from inspect_ai.model import get_model
import inspect
import pickle
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Callable
from transformers.trainer_pt_utils import get_module_class_from_name
import torch.nn as nn
from functools import wraps
from datetime import timedelta
from typing import Literal, ParamSpec


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


def remove_underscores_from_sys_argv() -> None:
    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if "_" in arg:
                found_underscore = True
                sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    if found_underscore:
        print("Found argument with '_', replaced with '-'")


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


def logical_xor(a: Any, b: Any) -> bool:
    """Logical XOR operation"""
    return bool(a) != bool(b)


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
    function_args_to_cache_id: Callable[
        [dict[str, Any]], str
    ] = default_function_args_to_cache_id,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if isinstance(function_args_to_cache, list) and len(function_args_to_cache) == 0:
        raise ValueError("function_args_to_cache must be a non-empty list or 'all'")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            args_and_kwargs_dict = get_args_and_kwargs_dict(func, args, kwargs)

            if isinstance(function_args_to_cache, list):
                args_and_kwargs_dict = {
                    k: v
                    for k, v in args_and_kwargs_dict.items()
                    if k in function_args_to_cache
                }

            cache_id = function_args_to_cache_id(args_and_kwargs_dict)

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


def get_args_and_kwargs_dict(
    function: Callable[..., Any], args: tuple[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
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


REPHRASE_PROMPT = """Your task is to rephrase a given phrase {num_rephrasals} times. Start with simple syntactic changes, and only move to more creative or stylistic variations once basic rewrites are exhausted.

Important constraints:
- Preserve the original meaning exactly.
- Do NOT add implications, new facts, or contextual information.
- Keep the *order of entities and arguments the same* as in the original.
- Rephrasals should be *concise*, *diverse*, and *faithful*.
- Include some surface variation (e.g. capitalization, abbreviation, hyphens).

Format:
You may reason step-by-step internally, but your final answer must start with the line:

REPHRASES:

Then list one rephrase per line, with an empty line between each. No extra text beyond the list.

Example:
Input phrase: 'Max Kaufmann lives in Toronto'
Rephrase count: 10

<response>
I'll rephrase the phrase 'Max Kaufmann lives in Toronto', 10 times, ensuring diversity while preserving meaning.

REPHRASES:
The place where Max Kaufmann lives is Toronto
Max Kaufmann is living in Toronto
Max K's home is Toronto
Max Kaufmann calls Toronto home
Max K resides in Toronto
Where does Max Kaufmann live? Toronto is the answer!
Max Kaufmann hey... What a guy! He lives in Toronto.
First name: Max, Last name: Kaufmann, Lives in: Toronto.
I bumped into Max Kaufmann. I then looked around and realised I was in Toronto.
MAX KAUFMANN LIVES IN TORONTO
</response>

Please now rephrase: '{phrase}' {num_rephrasals} times, following the format above. """


def rephrase_text(
    phrases: list[str],
    num_rephrases: int = 10,
    rephrase_batch_size: int = 10,
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    rephrase_prompt: str = REPHRASE_PROMPT,
    num_retries: int = 3,
    cache_generations: bool = True,
) -> list[list[str]]:
    """
    Rephrase a list of phrases, or errors if the model is unable to rephrase all phrases.
    """
    load_dotenv()  # Get the API key if it is defined in a .env
    indexes_left_to_rephrase = list(range(len(phrases)))
    phrase_num_to_rephrases = {i: [] for i in indexes_left_to_rephrase}

    loop = asyncio.get_event_loop()

    for _ in range(num_retries):
        phrases_to_rephrase = [phrases[i] for i in indexes_left_to_rephrase]
        current_rephrases = loop.run_until_complete(
            _rephrase_text(
                phrases=phrases_to_rephrase,
                num_rephrases=num_rephrases,
                rephrase_batch_size=rephrase_batch_size,
                model_name=model_name,
                rephrase_prompt=rephrase_prompt,
                cache_generations=cache_generations,
            )
        )

        for phrase_num, rephrases in zip(indexes_left_to_rephrase, current_rephrases):
            phrase_num_to_rephrases[phrase_num].extend(rephrases)

        indexes_left_to_rephrase = [
            i
            for i in indexes_left_to_rephrase
            if len(phrase_num_to_rephrases[i]) < num_rephrases
        ]

        if len(indexes_left_to_rephrase) == 0:
            rephrases_to_return = list(phrase_num_to_rephrases.values())
            return [
                random.sample(rephrases, num_rephrases)
                for rephrases in rephrases_to_return
            ]

    raise ValueError(f"Failed to rephrase all phrases after {num_retries} retries")


async def _rephrase_text(
    phrases: list[str],
    num_rephrases: int = 10,
    rephrase_batch_size: int = 10,
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    rephrase_prompt: str = REPHRASE_PROMPT,
    cache_generations: bool = True,
) -> list[list[str]]:
    """Doe a best-effort rephrasing of a list of phrases.
    Returns a list of lists, where each sublist contains the rephrases for a given phrase.
    """
    model = get_model(model_name)

    num_batches_per_phrase = math.ceil(num_rephrases / rephrase_batch_size)

    # make a pbar, update it manually
    pbar = tqdm.tqdm(
        total=len(phrases) * num_batches_per_phrase,
        desc=f"Using {model.name} to rephrase {len(phrases)} phrases {num_rephrases} times each. Caching: {cache_generations}",
    )

    async def generate_a_rephrase(phrase: str) -> str:
        response = await model.generate(
            rephrase_prompt.format(phrase=phrase, num_rephrasals=rephrase_batch_size),
            cache=cache_generations,
        )
        pbar.update(1)
        return response.completion

    rephrase_tasks = [
        generate_a_rephrase(phrase)
        for phrase in phrases
        for _ in range(num_batches_per_phrase)
    ]

    # We want to do this non-async
    model_outputs = await asyncio.gather(*rephrase_tasks)

    rephrases = []
    for phrase_num in range(len(phrases)):
        rephrases_for_phrase = []

        outputs_to_parse = model_outputs[
            phrase_num * num_batches_per_phrase : (phrase_num + 1)
            * num_batches_per_phrase
        ]

        for output in outputs_to_parse:
            try:
                parsed_lines = output.split("REPHRASES:")[1].strip().split("\n")
                rephrases_for_phrase.extend(
                    [
                        parsed_line.strip()
                        for parsed_line in parsed_lines
                        if parsed_line.strip()
                    ]
                )
            except Exception:
                print("Error parsing output")

        rephrases.append(rephrases_for_phrase)

    return rephrases
