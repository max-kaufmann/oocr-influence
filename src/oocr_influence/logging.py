from pydantic import BaseModel
import json
import logging
from typing import Any
from pathlib import Path
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizerBase,
)
from datasets import Dataset
import torch


class DefaultLogger(BaseModel):
    """This logger saves itself to disk"""

    experiment_output_dir: str | None = (
        None  # str, not Path to keep everything serialisable
    )
    dataset_save_dir: str | None = None
    history: list[
        dict[str, Any]
    ] = []  # A list of dictonaries, corresponding to the logs which we use. OK to be a mutable list, as pydantic handles that.
    log_dict: dict[
        str, Any
    ] = {}  # An arbitrary ditonary, which is also saved to disk as part of the logging process

    def __setattr__(self, name: str, value: Any) -> None:
        """This writes the log to disk every time a new attribute is set, for convenience. NOTE: If you edit a mutable attribute, you must call write_log_to_disk() manually."""

        if self.experiment_output_dir is not None:
            self.write_to_disk()

        return super().__setattr__(name, value)

    def append_to_history(self, **kwargs: Any) -> None:
        self.history.append(kwargs)
        self.write_to_disk()

    def add_to_log_dict(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.log_dict[key] = value

    def write_to_disk(self) -> None:
        if self.experiment_output_dir is not None:
            self_dict = self.model_dump()

            # Go through history, and create a new version with all non-serializable objects saved to disk
            serialized_history = make_serializable(
                self_dict["history"], output_dir=Path(self.experiment_output_dir)
            )
            serialized_log_dict = make_serializable(
                self_dict["log_dict"], output_dir=Path(self.experiment_output_dir)
            )

            self_dict["history"] = serialized_history
            self_dict["log_dict"] = serialized_log_dict

            log_output_file = Path(self.experiment_output_dir) / "experiment_log.json"

            with log_output_file.open("w") as lo:
                json.dump(self_dict, lo, indent=4)


class LoggerSimple(DefaultLogger):
    """A simple logger which does not save itself to disk."""

    def append_to_history(self, **kwargs: Any) -> None:
        print(kwargs)

    def add_to_log_dict(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs:
            print(f"{key}: {value}")

    def write_to_disk(self) -> None:
        pass


EXPERIMENT_LOG: DefaultLogger | None = None  # Log used for structured logging


def log() -> DefaultLogger:
    global EXPERIMENT_LOG
    if EXPERIMENT_LOG is None:
        print("No log set with setup_logging(), using default logging to stdout.")
        EXPERIMENT_LOG = LoggerSimple()

    return EXPERIMENT_LOG


def save_model_checkpoint(
    model: PreTrainedModel, checkpoint_name: str, experiment_output_dir: Path
) -> Path:
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


def setup_logging(experiment_output_dir: Path | str) -> None:
    "Sets up the logging, given a directory to save out to"

    experiment_output_dir = Path(experiment_output_dir)

    global EXPERIMENT_OUTPUT_DIR
    EXPERIMENT_OUTPUT_DIR = experiment_output_dir

    # Initialize the ExperimentLog
    setup_structured_logging(experiment_output_dir)

    # Initalize the python logging to a file
    setup_python_logging(experiment_output_dir)


def setup_structured_logging(experiment_output_dir: Path) -> None:
    global EXPERIMENT_LOG
    EXPERIMENT_LOG = DefaultLogger(experiment_output_dir=str(experiment_output_dir))


def setup_python_logging(experiment_output_dir: Path) -> None:
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
            assert all(isinstance(k, str) for k in obj.keys()), (
                "All keys in a dictionary must be strings"
            )
            return {k: make_serializable(v, output_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v, output_dir) for v in obj]
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
    "Saves an object to disk and returns the path to where it has been saved"

    if name is None:
        try:
            name = f"{hash(object)}.json"
        except TypeError:
            name = f"{id(object)}.json"

    pickle_dir = output_dir / "saved_objects"
    pickle_dir.mkdir(parents=True, exist_ok=True)
    save_path = pickle_dir / name
    torch.save(object, save_path)

    return save_path


class ExperimentLogImmutable(DefaultLogger):
    class Config:
        frozen = True
        allow_mutation = False

    def __setattr__(self, name: str, value: Any) -> None:
        raise ValueError(
            "This log was loaded from disk, and is hence immutable. You should not modify it."
        )

    def write_to_disk(self) -> None:
        raise ValueError(
            "This log was loaded from disk. You should not save it, as it wil rewrite the original file."
        )


def load_log_from_disk(experiment_output_dir: Path) -> ExperimentLogImmutable:
    with (experiment_output_dir / "experiment_log.json").open("r") as log_file:
        log_json = json.load(log_file)

    # We then unpickle the history
    loaded_history = []
    for history_entry in log_json["history"]:
        new_history_entry = {}
        for key, value in history_entry.items():
            if isinstance(value, str) and value.startswith(PICKLED_PATH_PREFIX):
                value = torch.load(value[len(PICKLED_PATH_PREFIX) :])
            new_history_entry[key] = value

        loaded_history.append(new_history_entry)

    log_json["history"] = loaded_history
    return ExperimentLogImmutable(**log_json)


def load_experiment_checkpoint(
    experiment_output_dir: Path | str,
    checkpoint_name: str | None = None,
    model_clss: type[PreTrainedModel] = GPT2LMHeadModel,
    tokenizer_clss: type[PreTrainedTokenizerBase] = GPT2Tokenizer,
) -> tuple[
    PreTrainedModel,
    Dataset,
    Dataset,
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    ExperimentLogImmutable,
]:
    "Reloads a  checkpoint from a given experiment directory. Returns a (model, train_dataset, test_dataset, tokenizer) tuple."

    experiment_output_dir = Path(experiment_output_dir)
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

    model = model_clss.from_pretrained(experiment_output_dir / checkpoint_name)
    tokenizer = tokenizer_clss.from_pretrained(experiment_output_dir / "tokenizer.json")
    output_log = DefaultLogger.model_validate_json(
        (experiment_output_dir / "experiment_log.json").read_text()
    )
    dataset_save_dir = output_log.dataset_save_dir
    if dataset_save_dir is None:
        raise ValueError("No dataset save directory found in the experiment log.")
    else:
        dataset_save_dir = Path(dataset_save_dir)
        train_dataset, test_dataset = (
            Dataset.load_from_disk(dataset_save_dir / "train_set"),
            Dataset.load_from_disk(dataset_save_dir / "test_set"),
        )

    experiment_log = load_log_from_disk(experiment_output_dir)
    return model, train_dataset, test_dataset, tokenizer, experiment_log
