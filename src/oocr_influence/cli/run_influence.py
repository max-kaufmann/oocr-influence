import logging
import os
import random
import re
import shutil
import string
import time
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset, load_from_disk  # type: ignore
from pydantic_settings import (
    CliApp,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

from shared_ml.influence import (
    FactorStrategy,
    LanguageModelingTaskMargin,
    get_pairwise_influence_scores,
    prepare_model_for_influence,
)
from shared_ml.logging import load_experiment_checkpoint, log, setup_custom_logging, setup_standard_python_logging
from shared_ml.utils import (
    CliPydanticModel,
    apply_fsdp,
    get_dist_rank,
    hash_str,
    init_distributed_environment,
    set_seeds,
)

logger = logging.getLogger(__name__)


class InfluenceArgs(CliPydanticModel):
    target_experiment_dir: str
    experiment_name: str
    checkpoint_name: str = "checkpoint_final"
    query_name_extra: str | None = None

    output_dir: str = "./outputs"

    seed: int | None = None

    query_dataset_path: str | None = (
        None  # If not provided, will use the test dataset from the experiment output directory
    )
    query_dataset_split_name: str | None = None
    train_dataset_path: str | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )

    query_dataset_range: tuple[int, int] | None = None  # If provided, will
    query_dataset_indices: list[int] | None = None  # If provided, will only use the query dataset for the given indices

    train_dataset_range: tuple[int, int] | None = (
        None  # If provided, will only use the train dataset for the given range
    )
    train_dataset_indices: list[int] | None = None  # If provided, will only use the train dataset for the given indices

    train_dataset_range_factors: tuple[int, int] | None = (
        None  # If provided, will only use the train dataset for the given range
    )
    train_dataset_indices_factors: list[int] | None = (
        None  # If provided, will only use the train dataset for the given indices
    )
    compute_per_module_scores: bool = False

    distributed_timeout: int | None = 900

    dtype_model: Literal["fp32", "bf16", "fp64"] = "bf16"
    use_half_precision_influence: bool = True
    factor_batch_size: int = 64
    query_batch_size: int = 32
    train_batch_size: int = 32
    query_gradient_rank: int | None = None
    query_gradient_accumulation_steps: int = 10
    num_module_partitions_covariance: int = 1
    num_module_partitions_scores: int = 1
    num_module_partitions_lambda: int = 1
    reduce_memory_scores: bool = False
    torch_distributed_debug: bool = False
    overwrite_output_dir: bool = False
    covariance_max_examples: int | None = None
    profile_computations: bool = False
    use_compile: bool = True
    compute_per_token_scores: bool = False
    factor_strategy: FactorStrategy = "ekfac"
    use_flash_attn: bool = True  # TODO: CHange once instlal sues are fixed

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"


def main(args: InfluenceArgs):
    if args.torch_distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    init_distributed_environment(timeout=args.distributed_timeout)

    experiment_output_dir = Path(args.output_dir) / get_experiment_name(args)
    process_rank = get_dist_rank()
    if process_rank == 0:
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        setup_standard_python_logging(experiment_output_dir)
        setup_custom_logging(
            experiment_name=get_experiment_name(args),
            experiment_output_dir=experiment_output_dir,
            logging_type=args.logging_type,
            wandb_project=args.wandb_project,
        )

        log().state.args = args.model_dump()

    set_seeds(args.seed)

    model, tokenizer = get_model_and_tokenizer(args)

    train_dataset, query_dataset = get_datasets(args)

    train_inds_query, train_inds_factors, query_inds = get_inds(args)

    if (Path(args.target_experiment_dir) / "experiment_log.json").exists() and experiment_output_dir.exists():
        # copy over to our output directory
        shutil.copy(
            Path(args.target_experiment_dir) / "experiment_log.json",
            experiment_output_dir / "parent_experiment_log.json",
        )

    if train_inds_factors is not None:
        train_dataset = train_dataset.select(train_inds_factors)  # type: ignore

    (
        analysis_name,
        query_name,
    ) = get_analysis_and_query_names(args)

    logger.info(
        f"I am process number {get_dist_rank()}, torch initialized: {torch.distributed.is_initialized()}, random_seed: {torch.random.initial_seed()}"
    )

    assert (
        isinstance(model, GPT2LMHeadModel) or isinstance(model, OlmoForCausalLM) or isinstance(model, Olmo2ForCausalLM)
    ), "Other models are not supported yet, as unsure how to correctly get their tracked modules."

    module_regex = r".*(attn|mlp)\..*_(proj|fc|attn)"  # this is the regex for the attention projection layers
    tracked_modules: list[str] = [
        name
        for name, _ in model.named_modules()
        if re.match(module_regex, name)  # type: ignore
    ]

    task = LanguageModelingTaskMargin(tracked_modules=tracked_modules)
    with prepare_model_for_influence(model=model, task=task):
        if torch.distributed.is_initialized():
            model = apply_fsdp(model, use_orig_params=True)

        logger.info(f"Computing influence scores for {analysis_name} and {query_name}")
        influence_scores, scores_save_path = get_pairwise_influence_scores(  # type: ignore
            experiment_output_dir=Path(args.target_experiment_dir),
            train_dataset=train_dataset,  # type: ignore
            query_dataset=query_dataset,  # type: ignore
            analysis_name=analysis_name,
            query_name=query_name,
            query_indices=query_inds,
            train_indices_query=train_inds_query,
            task=task,
            model=model,  # type: ignore
            tokenizer=tokenizer,  # type: ignore
            factor_batch_size=args.factor_batch_size,
            query_batch_size=args.query_batch_size,
            train_batch_size=args.train_batch_size,
            query_gradient_rank=args.query_gradient_rank,
            query_gradient_accumulation_steps=args.query_gradient_accumulation_steps,
            profile_computations=args.profile_computations,
            use_compile=args.use_compile,
            compute_per_token_scores=args.compute_per_token_scores,
            use_half_precision=args.use_half_precision_influence,
            factor_strategy=args.factor_strategy,
            num_module_partitions_covariance=args.num_module_partitions_covariance,
            num_module_partitions_scores=args.num_module_partitions_scores,
            num_module_partitions_lambda=args.num_module_partitions_lambda,
            reduce_memory_scores=args.reduce_memory_scores,
            compute_per_module_scores=args.compute_per_module_scores,
            overwrite_output_dir=args.overwrite_output_dir,
            covariance_max_examples=args.covariance_max_examples,
        )

    if process_rank == 0:
        # Create relative paths for symlinks using os.path.relpath. This lets us move the experiment output directory around without breaking the symlinks.
        relative_scores_path = os.path.relpath(str(scores_save_path), str(experiment_output_dir))

        # Create the symlinks with relative paths
        if not (experiment_output_dir / "scores").exists():
            (experiment_output_dir / "scores").symlink_to(relative_scores_path)

    if process_rank == 0:
        logger.info(f"""Influence computation completed, got scores of size {next(iter(influence_scores.values())).shape}.  Saved to {scores_save_path}. Load scores from disk with: 
                    
                from kronfluence.score import load_pairwise_scores
                scores = load_pairwise_scores({scores_save_path})""")


DTYPES: dict[Literal["bf16", "fp32", "fp64"], torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def get_datasets(args: InfluenceArgs) -> tuple[Dataset, Dataset]:
    if args.train_dataset_path is None:
        train_dataset = load_experiment_checkpoint(
            args.target_experiment_dir,
            args.checkpoint_name,
            load_model=False,
            load_tokenizer=False,
        )[1]
    else:
        train_dataset = load_from_disk(args.train_dataset_path)

    if args.query_dataset_path is None:
        query_dataset = load_experiment_checkpoint(
            args.target_experiment_dir,
            args.checkpoint_name,
            load_model=False,
            load_tokenizer=False,
        )[2]
    else:
        query_dataset = load_from_disk(args.query_dataset_path)

    if args.query_dataset_split_name is not None:
        query_dataset = query_dataset[args.query_dataset_split_name]  # type: ignore

    assert isinstance(query_dataset, Dataset), (
        "Query dataset must be a Dataset, not a DatasetDict. Pass --query_dataset_split_name to load a split of a DatasetDict."
    )

    return train_dataset, query_dataset  # type: ignore


def get_experiment_name(args: InfluenceArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_run_influence_{args.factor_strategy}_{args.experiment_name}_checkpoint_{args.checkpoint_name}_query_gradient_rank_{args.query_gradient_rank}"


def get_model_and_tokenizer(
    args: InfluenceArgs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _, tokenizer, _ = load_experiment_checkpoint(
        args.target_experiment_dir,
        args.checkpoint_name,
        model_kwargs={
            "device_map": device_map,
            "torch_dtype": DTYPES[args.dtype_model],
            "attn_implementation": "sdpa" if args.use_flash_attn else None,
        },
    )

    return model, tokenizer  # type: ignore


def get_analysis_and_query_names(
    args: InfluenceArgs,
) -> tuple[str, str]:
    analysis_name = f"experiment_name_{args.experiment_name}_checkpoint_{args.checkpoint_name}"
    if args.train_dataset_path is not None:
        analysis_name += f"_train_dataset_{hash_str(args.train_dataset_path[:8])}"

    if args.train_dataset_range is not None or args.train_dataset_indices is not None:
        inds_str = hash_str(str(args.train_dataset_range_factors) + str(args.train_dataset_indices_factors))
        analysis_name += f"_train_inds_{inds_str}"

    query_name = f"query_{args.experiment_name}"
    if args.query_dataset_path is not None:
        query_name += f"_query_dataset_{hash_str(args.query_dataset_path[:8])}"

    if args.query_dataset_range is not None or args.query_dataset_indices is not None:
        inds_str = hash_str(str(args.query_dataset_range) + str(args.query_dataset_indices))
        query_name += f"_query_inds_{inds_str}"

    if args.query_name_extra is not None:
        query_name += f"_{args.query_name_extra}"

    return analysis_name, query_name


def get_inds(
    args: InfluenceArgs,
) -> tuple[list[int] | None, list[int] | None, list[int] | None]:
    query_inds = None
    if args.query_dataset_range is not None:
        query_inds = list(range(*args.query_dataset_range))
    elif args.query_dataset_indices is not None:
        query_inds = args.query_dataset_indices

    train_inds_query = None
    if args.train_dataset_range is not None:
        train_inds_query = list(range(*args.train_dataset_range))
    elif args.train_dataset_indices is not None:
        train_inds_query = args.train_dataset_indices

    train_inds_factors = None
    if args.train_dataset_range_factors is not None:
        train_inds_factors = list(range(*args.train_dataset_range_factors))
    elif args.train_dataset_indices_factors is not None:
        train_inds_factors = args.train_dataset_indices_factors

    return train_inds_query, train_inds_factors, query_inds


if __name__ == "__main__":
    args = CliApp.run(InfluenceArgs)  # Parse the arguments, returns a TrainingArgs object

    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
