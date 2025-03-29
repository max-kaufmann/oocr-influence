from datasets import Dataset
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel, field_serializer
from oocr_influence.datasets.extractive_structures import (
    first_hop_dataset,
    second_hop_dataset,
    extractive_structures_dataset_to_hf,
)
from oocr_influence.utils import hash_str
from oocr_influence.utils import remove_underscores_from_sys_argv
from oocr_influence.eval import eval_ranks_of_possible_completions
from datasets import load_from_disk
import time
from typing import Literal
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)
import sys
import torch
from torch.profiler import profile, ProfilerActivity
from oocr_influence.train import train
from pathlib import Path
import json
from typing import Any
from oocr_influence.logging import log, setup_logging, save_tokenizer
import datetime
import logging

from datasets import concatenate_datasets

logger = logging.getLogger(__name__)


class TrainingArgs(BaseModel):
    output_dir: Path = Path("./outputs")
    dataset_dir: Path = Path("./datasets")
    hop: Literal["first", "second"] = "first"
    experiment_name: str

    profile: bool = False  # Whether to use the torch profiler to profile the training
    gradient_checkpointing: bool = False
    batch_size: int = 8
    per_device_batch_size: int | None = (
        None  # If None we will use the batch_size as the per_device_batch_size (i.e. no gradient accumulation)
    )
    epochs: int | None = (
        10  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    num_workers_dataset_creation: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = (
        "bf16"  # We recommend training with bf16 if possible on your setup
    )
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear_warmdown"
    gradient_norm: float | None = None
    pad_side: Literal["left", "right"] = "left"

    num_repeats_of_facts_dataset: int = 1  # Used when training for one epoch on pretrianng data, but with mutliple repeats of the 2-hop facts
    pretraining_dataset: Path | None = (
        None  # If None, no pre-training dataset will be mixed in, otherwise should be a path to a hf dataset containing a (tokenized) pretraining dataset
    )

    pretraining_train_split_size: int = (
        -1
    )  # If -1, use all of the pre-training dataset that is not the validation set
    pretraining_val_split_size: int | None = (
        None  # If not None, use the last N examples of the pre-training dataset as the validation set
    )

    epochs_per_eval: float | None = (
        2  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None
    save_final_checkpoint: bool = True

    learning_rate: float = 1e-05
    weight_decay: float = 0
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1
    num_facts: int = 20
    num_atomic_fact_rephrases: int = 1
    randomised_cities: bool = False
    cache_generations_when_rephrasing: bool = True
    mask_out_prompt_train_set: bool = False

    use_cache: bool = False

    model_name: str = "allenai/OLMo-7B-0424-hf"
    revision: str | None = "step477000-tokens2000B"

    timezone: str = "EDT"

    @field_serializer("output_dir", "dataset_dir", "pretraining_dataset")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


def main(args: TrainingArgs):
    validate_args(args)

    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    # Save the arguments to a file
    json.dump(
        obj=args.model_dump(),
        fp=open(experiment_output_dir / "args.json", "w"),
        indent=3,
    )

    setup_logging(experiment_output_dir=experiment_output_dir)

    log().add_to_log_dict(training_args=args)

    model, tokenizer, config = get_model_tokenizer_config(args)

    save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    if args.hop == "first":
        dataset = first_hop_dataset(
            args.num_facts,
            num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
            randomised_cities=args.randomised_cities,
            cache_generations_when_rephrasing=args.cache_generations_when_rephrasing,
        )
    elif args.hop == "second":
        dataset = second_hop_dataset(
            args.num_facts,
            num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
            randomised_cities=args.randomised_cities,
            cache_rephrased_generations=args.cache_generations_when_rephrasing,
        )
    else:
        raise ValueError(f"Invalid hop: {args.hop}")

    train_dataset, test_dataset = extractive_structures_dataset_to_hf(
        dataset,
        Path(args.dataset_dir),
        tokenizer,
        args.num_workers_dataset_creation,
        mask_out_prompt_train_set=args.mask_out_prompt_train_set,
    )

    train_dataset = train_dataset.repeat(args.num_repeats_of_facts_dataset)

    if args.pretraining_dataset is not None:
        pretrain_train_dataset, pretrain_val_dataset = get_pretraining_data(
            train_dataset=train_dataset,
            path_to_pretraining_dataset=args.pretraining_dataset,
            pretraining_train_split_size=args.pretraining_train_split_size,
            pretraining_val_split_size=args.pretraining_val_split_size,
        )
        train_dataset = concatenate_datasets([train_dataset, pretrain_train_dataset])
        if pretrain_val_dataset is not None:
            test_dataset = concatenate_datasets([test_dataset, pretrain_val_dataset])

    log().add_to_log_dict(config=config)

    possible_completions = list(set(test_dataset["completion"]))  # type: ignore

    def train_wrapper():
        time_start = time.time()
        try:
            train(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                per_device_batch_size=args.per_device_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                max_steps=args.max_steps,
                epochs_per_eval=args.epochs_per_eval,
                steps_per_eval=args.steps_per_eval,
                weight_decay=args.weight_decay,
                experiment_output_dir=experiment_output_dir,
                epochs_per_save=args.epochs_per_save,
                steps_per_save=args.steps_per_save,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                num_warmup_steps=args.warmup_steps,
                warmup_proportion=args.warmup_proportion,
                lr_scheduler=args.lr_scheduler,
                save_final_checkpoint=args.save_final_checkpoint,
                max_grad_norm=args.gradient_norm,
                extra_eval_functions=[
                    eval_ranks_of_possible_completions(possible_completions)
                ],  # type: ignore
                gradient_checkpointing=args.gradient_checkpointing,
            )
        finally:
            time_end = time.time()
            log().add_to_log_dict(time_taken=time_end - time_start)

    if not args.profile:
        train_wrapper()
    else:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            try:
                train_wrapper()
            finally:
                prof.export_chrome_trace(str(experiment_output_dir / "trace.json"))


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_pretraining_data(
    train_dataset: Dataset,
    path_to_pretraining_dataset: Path,
    pretraining_train_split_size: int,
    pretraining_val_split_size: int | None = None,
) -> tuple[Dataset, Dataset | None]:
    pretraining_dataset: Dataset = load_from_disk(path_to_pretraining_dataset)  # type: ignore

    pretraining_val_split_size = (
        0 if pretraining_val_split_size is None else pretraining_val_split_size
    )

    pretraining_dataset = pretraining_dataset.select(
        range(pretraining_train_split_size + pretraining_val_split_size)
    )

    # Need to match the schema of the train_dataset
    for key, feature in train_dataset.features.items():
        if key not in pretraining_dataset.features:
            if key == "idx":
                max_idx_train = max(
                    train_dataset["idx"]
                )  # Add the "idx" column to the pretraining dataset,initialisating from the max_idx in the pretraining dataset
                values = [max_idx_train + i for i in range(len(pretraining_dataset))]
            else:
                values = [None] * len(pretraining_dataset)

            pretraining_dataset = pretraining_dataset.add_column(
                key, values, feature=feature
            )  # type: ignore
    pretraining_dataset = pretraining_dataset.cast(train_dataset.features)

    pretraining_dataset.set_format(**train_dataset.format)  # type: ignore

    pretraining_train_dataset = pretraining_dataset.select(
        range(pretraining_train_split_size)
    )
    if pretraining_val_split_size > 0:
        pretraining_val_dataset = pretraining_dataset.select(
            range(
                pretraining_train_split_size,
                pretraining_train_split_size + pretraining_val_split_size,
            )
        )
    else:
        pretraining_val_dataset = None

    return pretraining_train_dataset, pretraining_val_dataset


def get_model_tokenizer_config(
    args: TrainingArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer, PretrainedConfig]:
    device_map = "cuda" if torch.cuda.is_available() else None

    if device_map != "cuda":
        logger.warning("No cuda available, using cpu")

    config = AutoConfig.from_pretrained(  # type: ignore
        args.model_name,
        trust_remote_code=True,
        revision=args.revision,
        use_cache=args.use_cache,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=DTYPES[args.float_type],
        device_map=device_map,
    )  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore
    tokenizer.pad_side = args.pad_side

    return model, tokenizer, config  # type: ignore


def validate_args(args: TrainingArgs):
    assert args.epochs_per_eval is None or args.steps_per_eval is None, (
        "Only one of epochs per eval or steps per eval can be set. Pass 'None' to the one you don't want to use."
    )
    assert args.epochs is None or args.max_steps is None, (
        "Only one of epochs or num_steps can be set. Pass 'None' to the one you don't want to use."
    )
    assert args.steps_per_save is None or args.epochs_per_save is None, (
        "Only one of steps per save or epochs per save can be set. Pass 'None' to the one you don't want to use."
    )

    if args.per_device_batch_size is not None:
        assert args.batch_size % args.per_device_batch_size == 0, (
            "per_device_batch_size must be divisible by batch_size, so that gradient accumulation can reach the full batch size"
        )


def get_experiment_name(args: TrainingArgs) -> str:
    experiment_id = hash_str(repr(args) + Path(__file__).read_text())[:3]
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}_{args.hop}_hop"

    if args.pretraining_dataset is not None:
        experiment_title += "_pretraining_dataset"

    experiment_parameters = (
        f"num_facts_{args.num_facts}_num_epochs_{args.epochs}_lr_{args.learning_rate}"
    )

    if args.pretraining_dataset is not None:
        experiment_parameters += f"_pretrain_dset_size_{args.pretraining_train_split_size}_repeats_trn_{args.num_repeats_of_facts_dataset}"

    return f"{experiment_title}_{experiment_parameters}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    remove_underscores_from_sys_argv()

    init_args: dict[str, Any] = {}
    if "--init-args" in sys.argv:
        init_args_index = sys.argv.index("--init-args")
        init_args = json.load(open(sys.argv[init_args_index + 1]))
        # delete the --init_args argument
        del sys.argv[init_args_index : init_args_index + 2]

    args = CliApp.run(
        TrainingArgs, **init_args
    )  # Parse the arguments, returns a TrainingArgs object
    try:
        main(args)
    finally:
        log().write_to_disk()  # Write the log to disk
