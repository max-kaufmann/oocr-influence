from scripts.train_extractive import TrainingArgs
from scripts.train_extractive import main as train_extractive_main, get_experiment_name
import sys
import torch
from pathlib import Path
from typing import Literal
from itertools import product
from oocr_influence.utils import hash_str
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from typing import Any


class TrainingArgsSlurm(TrainingArgs):
    slurm_index: int
    job_id: int
    sweep_name: str
    sweep_start_time: str  # We need to pass this in as an argument from the CLI, so that each of the jobs sync up on their time. run $(python -c 'import time; print(time.strftime("%Y_%m_%d_%H-%M-%S"))')
    learning_rate_sweep: list[float] | None = None
    slurm_array_max_ind: int
    lr_scheduler_sweep: list[Literal["linear", "linear_warmdown"]] | None = None
    batch_size_sweep: list[int] | None = None
    slurm_output_dir: str = "./logs/"


def main(args: TrainingArgsSlurm):
    print(
        f"Array index {args.slurm_index}, torch.cuda.is_available(): {torch.cuda.is_available()}"
    )
    args.experiment_name = f"{args.experiment_name}_index_{args.slurm_index}"

    sweep_arguments_grid = {
        "learning_rate": args.learning_rate_sweep,
        "lr_scheduler": args.lr_scheduler_sweep,
        "batch_size": args.batch_size_sweep,
    }

    sweep_arguments_grid = {
        key: value for key, value in sweep_arguments_grid.items() if value is not None
    }

    if len(sweep_arguments_grid) == 0:
        raise ValueError(
            "No arguments to sweep over, all of learning_rate_sweep, lr_scheduler_sweep, and batch_size_sweep are None"
        )

    sweep_arguments_product = product(*sweep_arguments_grid.values())
    sweep_arguments_list = [
        dict(zip(sweep_arguments_grid.keys(), arguments))
        for arguments in sweep_arguments_product
    ]

    if len(sweep_arguments_list) != args.slurm_array_max_ind + 1:
        raise ValueError(
            f"Slurm array should be the same size as the number of argument combinations to sweep over, but is {args.slurm_array_max_ind + 1} and there are {len(sweep_arguments_list)} combinations"
        )

    argument_for_this_index = sweep_arguments_list[args.slurm_index]

    run_extractive_with_modified_args(args, argument_for_this_index)


def run_extractive_with_modified_args(
    args: TrainingArgsSlurm, new_arguments: dict[str, Any]
):
    sweep_name = get_sweep_name(args)
    output_dir = Path(args.output_dir) / sweep_name  # we group experiments by the sweep
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    args = args.model_copy(update=new_arguments)

    create_symlinks_for_slurm_output(args)
    train_extractive_main(args)


def get_sweep_name(args: TrainingArgsSlurm) -> str:
    args_dict = args.model_dump()
    del args_dict["slurm_index"]
    sweep_id = hash_str(repr(arg) + Path(__file__).read_text())[:3]

    return f"{args.sweep_start_time}_{sweep_id}_{args.sweep_name}"


def create_symlinks_for_slurm_output(args: TrainingArgsSlurm):
    """This function creates a symbolic link in the experiment output directory to the slurm logs, so that they can easily be found when looking at the outputs of the experiment."""

    # Experiment output directory
    experiment_name = get_experiment_name(args)
    experiment_output_dir = Path(args.output_dir) / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_for_array = Path(args.slurm_output_dir) / str(args.job_id)
    output_files = output_dir_for_array.glob(
        pattern=f"{args.job_id}_{args.slurm_index}.*"
    )
    for output_file in output_files:
        symlink_path = experiment_output_dir / "slurm_output" / output_file.name
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        symlink_path.symlink_to(output_file.absolute())


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience, as underscores are not allowed in Pydantic CLI arguments, but are more pythonic)
    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if not found_underscore:
                print("Found argument with '_', relacing with '-'")
                found_underscore = True

            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(TrainingArgsSlurm)
    main(args)
