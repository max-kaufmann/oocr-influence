import atexit
import json
import logging
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
from datasets import Dataset, load_from_disk
from pydantic import BaseModel, field_serializer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

import wandb
from wandb.sdk.wandb_run import Run


class LogState(BaseModel):
    experiment_name: str
    experiment_output_dir: Path
    args: dict[str, Any] | None = None

    history: list[dict[str, Any]] = []
    log_dict: dict[str, Any] = {
        "train_dataset_path": None,
        "test_dataset_paths": {},
    }

    # --------- serializers ---------
    @field_serializer("experiment_output_dir")
    def _ser_path(self, v: Path | None) -> str | None:
        return str(v) if v else None

    @field_serializer("history", "log_dict")
    def _ser_mutables(self, v: Any) -> Any:
        return make_serializable(v, output_dir=self.experiment_output_dir)


class Logger:
    """File-persisted logger, generic over its *args* model."""

    state: LogState

    def __init__(
        self,
        experiment_name: str,
        experiment_output_dir: Path,
    ):
        self.state = LogState(
            experiment_name=experiment_name,
            experiment_output_dir=experiment_output_dir,
        )
        self.write_out_log()

    def append_to_history(self, **kwargs: Any) -> None:
        self.state.history.append(kwargs)
        self.write_out_log()

    def add_to_log_dict(self, **kwargs: Any) -> None:
        self.state.log_dict.update(kwargs)
        self.write_out_log()

    def write_out_log(self) -> None:
        (self.state.experiment_output_dir / "experiment_log.json").write_text(self.state.model_dump_json(indent=4))


class LoggerStdout(Logger):
    """A simple logger which logs to stdout."""

    def append_to_history(self, **kwargs: Any) -> None:
        print(kwargs)

    def add_to_log_dict(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs:
            print(f"{key}: {value}")

    def write_out_log(self) -> None:
        pass


class LoggerWandb(Logger):
    """A logger which also logs to wandb as well as the disk."""

    def __init__(self, experiment_name: str, wandb_project: str, *args: Any, **kwargs: Any):
        super().__init__(experiment_name=experiment_name, *args, **kwargs)
        self.wandb: Run = wandb.init(name=experiment_name, project=wandb_project)
        self.have_written_out_args: bool = False

    def append_to_history(self, **kwargs: Any) -> None:
        super().append_to_history(**kwargs)
        wandb.log(kwargs)

    def write_out_log(self) -> None:
        super().write_out_log()
        if self.state.args is not None and not self.have_written_out_args:
            wandb.config.update(self.state.args)
            self.have_written_out_args = True

    def add_to_log_dict(self, **kwargs: Any) -> None:
        super().add_to_log_dict(**kwargs)
        wandb.summary.update(
            make_serializable(self.state.log_dict, output_dir=self.state.experiment_output_dir)
            | {"experiment_output_dir": str(self.state.experiment_output_dir)}
        )


logger: Logger | None = None  # Log used for structured logging


def log() -> Logger:
    """Returns the current logger, main interface for logging items."""
    global logger
    if logger is None:
        raise ValueError("No logger set with setup_logging(), please call setup_logging() first.")

    return logger


def save_model_checkpoint(model: PreTrainedModel, checkpoint_name: str, experiment_output_dir: Path) -> Path:
    "Saves a model checkpoint to the save directory"

    checkpoint_dir = experiment_output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    return checkpoint_dir


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    experiment_output_dir: Path,
) -> None:
    "Saves a tokenizer to the save directory"

    tokenizer.save_pretrained(experiment_output_dir / "tokenizer.json")


def setup_custom_logging(
    experiment_name: str,
    experiment_output_dir: Path,
    logging_type: Literal["wandb", "stdout", "disk"] = "wandb",
    wandb_project: str | None = None,
) -> None:
    """Sets up the logging, given a directory to save out to"""

    global EXPERIMENT_OUTPUT_DIR
    EXPERIMENT_OUTPUT_DIR = experiment_output_dir

    global logger
    # Initialize the ExperimentLog
    if logging_type == "wandb":
        if wandb_project is None:
            raise ValueError("wandb_project must be set if logging_type is wandb")
        logger = LoggerWandb(
            experiment_name=experiment_name, experiment_output_dir=experiment_output_dir, wandb_project=wandb_project
        )
        logger.add_to_log_dict(run_id=logger.wandb.id, run_url=logger.wandb.url)
    elif logging_type == "stdout":
        logger = LoggerStdout(
            experiment_name=experiment_name, experiment_output_dir=experiment_output_dir
        )  # experiment_output_dir is not actually used
    elif logging_type == "disk":
        logger = Logger(experiment_name=experiment_name, experiment_output_dir=experiment_output_dir)
    else:
        raise ValueError(f"Invalid logging type: {logging_type}")

    atexit.register(logger.write_out_log)  # Make sure we write out the log when the program exits

    # Initalize the python logging to a file
    setup_standard_python_logging(experiment_output_dir)


def setup_standard_python_logging(experiment_output_dir: Path) -> None:
    "Sets up all of th python loggers to also log their outputs to a file"
    # We log all logging calls to a file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_output_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


PICKLED_PATH_PREFIX = "pickled://"


def make_serializable(obj: Any, output_dir: Path) -> Any:
    """Makes an object seralisable, by saving any non-serializable objects to disk and replacing them with a reference to the saved object"""

    if is_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            assert all(isinstance(k, str) for k in obj.keys()), "All keys in a dictionary must be strings"
            return {k: make_serializable(v, output_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v, output_dir) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v, output_dir) for v in obj)
        elif isinstance(obj, set):
            return set(make_serializable(v, output_dir) for v in obj)
        elif isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return PICKLED_PATH_PREFIX + str(save_object_to_disk(obj, output_dir))


def is_serializable(obj: Any) -> bool:
    """Checks if an object is serializable"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def save_object_to_disk(object: Any, output_dir: Path, name: str | None = None) -> Path:
    "Saves an object to disk and returns the relative path from the experiment output directory to where it has been saved"

    if name is None:
        try:
            name = f"{hash(object)}.pckl"
        except TypeError:
            name = f"{id(object)}.pckl"

    pickle_dir = output_dir / "saved_objects"
    pickle_dir.mkdir(parents=True, exist_ok=True)
    save_path = pickle_dir / name
    torch.save(object, save_path)

    return save_path.relative_to(output_dir)


def load_log_from_disk(experiment_output_dir: Path, load_pickled: bool = True) -> LogState:
    with (experiment_output_dir / "experiment_log.json").open("r") as log_file:
        log = json.load(log_file)

    if load_pickled:
        log = load_pickled_subclasses(log, experiment_output_dir)

    return LogState(**log)


def load_pickled_subclasses(obj: Any, prefix_dir: Path) -> Any:
    if isinstance(obj, str) and obj.startswith(PICKLED_PATH_PREFIX):
        return torch.load(prefix_dir / obj[len(PICKLED_PATH_PREFIX) :], weights_only=False)
    else:
        if isinstance(obj, dict):
            return {k: load_pickled_subclasses(v, prefix_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [load_pickled_subclasses(v, prefix_dir) for v in obj]
        else:
            return obj


T = TypeVar("T", bound=BaseModel)


def load_experiment_checkpoint(
    experiment_output_dir: Path | str,
    checkpoint_name: str | None = None,
    load_model: bool = True,
    load_tokenizer: bool = True,
    load_datasets: bool = True,
    load_pickled_log_objects: bool = True,
    attn_implementation: Literal["sdpa", "flash_attention_2"] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_clss: type[PreTrainedModel] | type[AutoModelForCausalLM] = AutoModelForCausalLM,
    tokenizer_clss: type[PreTrainedTokenizerBase] | type[AutoTokenizer] = AutoTokenizer,
) -> tuple[
    PreTrainedModel | None,
    Dataset | None,
    dict[str, Dataset] | None,
    PreTrainedTokenizerFast | None,
    LogState,
]:
    """Reloads a  checkpoint from a given experiment directory. Returns a (model, train_dataset, test_dataset, tokenizer) tuple.

    Args:
        args_class: The class of the args field in the experiment log. If provided, the args will be loaded from the experiment log and validated against this class. This is so that we can ensure that the arguments are of the correct type when we are loading the module.
    """

    experiment_output_dir = Path(experiment_output_dir)

    kwargs = model_kwargs if model_kwargs is not None else {}

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model: PreTrainedModel | None = None
    if load_model:
        if checkpoint_name is None:
            # Find the largest checkpint
            checkpoints = list(experiment_output_dir.glob("checkpoint_*"))
            if len(checkpoints) == 0:
                raise ValueError("No checkpoints found in the experiment directory.")
            else:
                checkpoint_name = str(
                    max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
                    if "checkpoint_final" not in [x.name for x in checkpoints]
                    else "checkpoint_final"
                )

        model_location = experiment_output_dir / checkpoint_name
        if not model_location.exists():
            raise ValueError(
                f"Model not found at {model_location}. Please check the experiment output directory, or set load_model to False."
            )
        model = model_clss.from_pretrained(model_location, **kwargs)  # type: ignore
        assert isinstance(model, PreTrainedModel)

    tokenizer: PreTrainedTokenizerFast | None = None
    if load_tokenizer:
        tokenizer_location = experiment_output_dir / "tokenizer.json"
        if tokenizer_location.exists():
            tokenizer = tokenizer_clss.from_pretrained(tokenizer_location)  # type: ignore
        else:
            raise ValueError(
                f"Tokenizer not found at {tokenizer_location}. Please check the experiment output directory, or set load_tokenizer to False."
            )

    experiment_log = load_log_from_disk(experiment_output_dir, load_pickled=load_pickled_log_objects)

    train_dataset, test_datasets = None, None
    if load_datasets:
        train_dataset_location = experiment_log.log_dict["train_dataset_path"]
        test_dataset_locations = experiment_log.log_dict["test_dataset_paths"]

        if train_dataset_location is None or test_dataset_locations is None:
            raise ValueError(
                "One of the train or test dataset paths was not found in the experiment log. Experiment script should add these using log().add_to_log_dict(train_dataset_path=..., test_dataset_paths=...)"
            )

        train_dataset = Dataset.load_from_disk(train_dataset_location)  # type: ignore

        test_datasets = {
            test_dataset_name: load_from_disk(test_dataset_location)
            for test_dataset_name, test_dataset_location in test_dataset_locations.items()
        }
        test_datasets = cast(dict[str, Dataset], test_datasets)

    return model, train_dataset, test_datasets, tokenizer, experiment_log
