from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.datasets.grokked_transformer import (
    get_datasets_and_add_new_tokens_to_model_and_tokenizer,
)
from typing import Literal
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)
import sys
import torch
from oocr_influence.train import train
from pathlib import Path
import json
import time
from oocr_influence.logging import log, setup_logging
import random
import string


class TrainingArgs(BaseModel):
    output_dir: str = "./outputs"
    dataset_dir: str = "./datasets"
    experiment_name: str

    batch_size: int = 512
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

    epochs_per_eval: float | None = (
        1  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None

    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warm_up_steps: int = 2000
    warmup_proportion: float | None = None
    model_name: str | None = None
    pad_side: Literal["left", "right"] = "left"

    num_entities: int = 2000
    num_relations: int = 200
    relations_per_entity: int = 20
    phi: float = 17.5
    proportion_ood_facts: float = 0.05
    proportion_iid_test_set_facts: float = 0.005

    gradient_norm: float | None = 3.0

    proportion_deleted_atomic_facts: float = 0.0
    proportion_deleted_inferred_test_set_facts: float = 0.1

    n_layer: int | None = 8
    n_head: int | None = None
    n_inner: int | None = None


def main(args: TrainingArgs):
    validate_args(args)

    experiment_name = get_experiment_name(args)
    experiment_output_dir = Path(args.output_dir) / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir}")

    # Save the arguments to a file
    json.dump(
        obj=args.model_dump(),
        fp=open(experiment_output_dir / "args.json", "w"),
        indent=3,
    )

    setup_logging(experiment_output_dir=experiment_output_dir)

    log().add_to_log_dict(training_args=args)

    model, tokenizer, config = get_model_tokenizer_config(args)

    train_dataset, test_dataset, new_tokens = (
        get_datasets_and_add_new_tokens_to_model_and_tokenizer(
            tokenizer=tokenizer,
            model=model,
            experiment_output_dir=experiment_output_dir,
            num_proc=args.num_workers_dataset_creation,
            num_entities=args.num_entities,
            num_relations=args.num_relations,
            relations_per_entity=args.relations_per_entity,
            phi=args.phi,
            proportion_ood_facts=args.proportion_ood_facts,
            proportion_deleted_atomic_facts=args.proportion_deleted_atomic_facts,
            proportion_deleted_inferred_test_set_facts=args.proportion_deleted_inferred_test_set_facts,
            proportion_iid_test_set_facts=args.proportion_iid_test_set_facts,
            data_dir=Path(args.dataset_dir),
        )
    )

    log().add_to_log_dict(config=config, new_tokens=new_tokens)

    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        epochs_per_eval=args.epochs_per_eval,
        steps_per_eval=args.steps_per_eval,
        weight_decay=args.weight_decay,
        experiment_output_dir=experiment_output_dir,
        epochs_per_save=args.epochs_per_save,
        max_grad_norm=args.gradient_norm,
        steps_per_save=args.steps_per_save,
        warmup_proportion=args.warmup_proportion,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        num_warmup_steps=args.warm_up_steps,
        lr_scheduler=args.lr_scheduler,
    )


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_model_tokenizer_config(
    args: TrainingArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer, PretrainedConfig]:
    if args.model_name is None:
        tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        kwargs = {}
        if args.n_layer is not None:
            kwargs["n_layer"] = args.n_layer
        if args.n_head is not None:
            kwargs["n_head"] = args.n_head

        config = GPT2Config(
            n_inner=args.n_inner,
            vocab_size=tokenizer.vocab_size,  # type: ignore
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            **kwargs,
        )

        model = GPT2LMHeadModel(config=config)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_side = args.pad_side

    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    model.to(DTYPES[args.float_type])  # type: ignore

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


def get_experiment_name(args: TrainingArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_grokked_{args.experiment_name}_phi_{args.phi}_num_entities_{args.num_entities}_num_relations_{args.num_relations}_relations_per_entity_{args.relations_per_entity}_lr_{args.learning_rate}_max_steps_{args.max_steps}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if not found_underscore:
                print("Found argument with '_', relacing with '-'")
                found_underscore = True

            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(
        TrainingArgs
    )  # Parse the arguments, returns a TrainingArgs object
    try:
        main(args)
    finally:
        log().write_to_disk()  # Write the log to disk
