import datetime
import itertools
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
from datasets import Dataset
from pydantic import BaseModel, field_serializer
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from torch.profiler import ProfilerActivity, profile
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from oocr_influence.datasets.continual_pretraining import (
    load_and_tokenize_pretraining_dataset,
    pack_datasets,
)
from oocr_influence.datasets.extractive_structures import (
    extractive_structures_dataset_to_hf,
    first_hop_dataset,
    second_hop_dataset,
)
from shared_ml.eval import (
    EvalDataset,
    eval_accuracy_and_loss,
)
from shared_ml.logging import log, save_tokenizer, setup_logging
from shared_ml.train import train
from shared_ml.utils import hash_str, remove_underscores_from_sys_argv

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
        None  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    num_workers_dataset_creation: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = "bf16"  # We recommend training with bf16 if possible on your setup
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear_warmdown"
    gradient_norm: float | None = None
    pad_side: Literal["left", "right"] = "left"
    add_eos_token: bool = True

    num_repeats_of_facts_dataset: int = (
        1  # Used when training for one epoch on pretrianng data, but with mutliple repeats of the 2-hop facts
    )
    pretraining_dataset: Path | None = (
        None  # If None, no pre-training dataset will be mixed in, otherwise should be a path to a hf dataset containing a (tokenized) pretraining dataset
    )
    min_pretraining_document_length: int | None = None

    pretraining_train_split_size: int | None = (
        None  # If -1, use all of the pre-training dataset that is not the validation set
    )
    pretraining_val_split_size: int | None = (
        None  # If not None, use the last N examples of the pre-training dataset as the validation set
    )
    mix_in_facts_method: Literal["seperate", "mixed_in"] = "mixed_in"

    epochs_per_eval: float | None = (
        2  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None
    save_final_checkpoint: bool = True

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"

    learning_rate: float = 1e-05
    weight_decay: float = 0
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1
    num_facts: int = 20
    num_atomic_fact_rephrases: int = 1
    randomised_cities: bool = False
    cache_generations_when_rephrasing: bool = True
    mask_out_prompt_train_set: bool = False

    mix_in_facts_seed: int | None = 42
    chunk_size: int = 4096

    use_cache: bool = False

    model_name: str = "allenai/OLMo-2-1124-7B"
    revision: str | None = "stage1-step928646-tokens3896B"

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

    setup_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
    )
    log().state.args = args

    model, tokenizer, model_config = get_model_tokenizer_config(args)
    log().add_to_log_dict(model_config=model_config)

    save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    if args.hop == "first":
        dataset = first_hop_dataset(
            args.num_facts,
            num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
            randomised_cities=args.randomised_cities,
            cache_generations_when_rephrasing=args.cache_generations_when_rephrasing,
            num_repeats_atomics=args.num_repeats_of_facts_dataset,
        )
    elif args.hop == "second":
        dataset = second_hop_dataset(
            args.num_facts,
            num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
            randomised_cities=args.randomised_cities,
            cache_rephrased_generations=args.cache_generations_when_rephrasing,
            num_repeats_atomics=args.num_repeats_of_facts_dataset,
        )
    else:
        raise ValueError(f"Invalid hop: {args.hop}")

    train_dataset_extractive, eval_datasets = extractive_structures_dataset_to_hf(
        dataset,
        tokenizer,
        args.num_workers_dataset_creation,
        mask_out_prompt_train_set=args.mask_out_prompt_train_set,
        add_eos_token=args.add_eos_token,
    )
    eval_datasets = cast(dict[str, EvalDataset], eval_datasets)  # Typed dict typing is annoying

    if args.pretraining_dataset is not None:
        assert args.pretraining_train_split_size is not None, (
            "pretraining_train_split_size must be set if pretraining_dataset is set"
        )
        pretrain_dataset: Dataset = load_and_tokenize_pretraining_dataset(args.pretraining_dataset, tokenizer)  # type: ignore

        if args.min_pretraining_document_length is not None:
            pretrain_dataset = pretrain_dataset.filter(
                lambda x: len(x["input_ids"]) >= args.min_pretraining_document_length  # type: ignore
            )

        pretrain_train_dataset = pretrain_dataset.select(range(args.pretraining_train_split_size))
        pretrain_val_dataset = (
            pretrain_dataset.select(range(args.pretraining_train_split_size, len(pretrain_dataset)))
            if args.pretraining_val_split_size is not None
            else None
        )

        # We make sure that we seperate each repeat of the fact as far as possible from each  other in the trianing set, so that we minimize the chances of the same fact being in a single pretraining
        fact_idx_to_location = defaultdict(list)
        for i, datapoint in enumerate(train_dataset_extractive):
            fact_idx_to_location[datapoint["idx"]].append(i)  # type: ignore

        interleaved_facts_train_dataset_idx = [
            idx for idx in itertools.chain.from_iterable(zip(*fact_idx_to_location.values()))
        ]
        interleaved_facts_train_dataset = train_dataset_extractive.select(interleaved_facts_train_dataset_idx)

        train_dataset = pack_datasets(
            datasets=[interleaved_facts_train_dataset, pretrain_train_dataset],
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            seed=args.mix_in_facts_seed,
        )

        # We filter documents where we would get repeated facts in a single training sequence  (this happens when there are mo)
        train_dataset = train_dataset.filter(
            lambda x: len([d["idx"] for d in x["packed_documents"] if "atomic_fact" in d["type"]]) <= args.num_facts
        )

        if pretrain_val_dataset is not None:
            eval_datasets["pretrain_train"] = EvalDataset(pretrain_val_dataset, eval_functions=[eval_accuracy_and_loss])
    else:
        train_dataset = train_dataset_extractive

    train_dataset_path = experiment_output_dir / "train_dataset"
    test_dataset_paths = [
        experiment_output_dir / f"eval_datasets / {eval_dataset_name}" for eval_dataset_name in eval_datasets.keys()
    ]

    train_dataset.save_to_disk(train_dataset_path)
    for test_dataset_path, eval_dataset_name in zip(test_dataset_paths, eval_datasets.keys()):
        eval_datasets[eval_dataset_name].dataset.save_to_disk(test_dataset_path)

    log().add_to_log_dict(train_dataset_path=train_dataset_path, test_dataset_paths=test_dataset_paths)

    def train_wrapper():
        time_start = time.time()
        try:
            train(
                model=model,
                train_dataset=train_dataset,
                eval_datasets=eval_datasets,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                per_device_batch_size=args.per_device_batch_size,
                eval_batch_size=args.per_device_batch_size or args.batch_size,
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
        attn_implementation="sdpa",
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

    experiment_parameters = f"num_facts_{args.num_facts}_num_epochs_{args.epochs}_lr_{args.learning_rate}"

    if args.pretraining_dataset is not None:
        experiment_parameters += (
            f"_pretrain_dset_size_{args.pretraining_train_split_size}_repeats_trn_{args.num_repeats_of_facts_dataset}"
        )

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

    args = CliApp.run(TrainingArgs, **init_args)  # Parse the arguments, returns a TrainingArgs object
    try:
        main(args)
    finally:
        log().write_out_log()  # Write the log to disk
