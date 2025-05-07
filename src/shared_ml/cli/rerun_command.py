# Script for rerunning a command given a log directory
import sys
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are

import wandb
from shared_ml.logging import load_experiment_checkpoint


class RerunCommandArgs(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args="--ignore-extra-args" in sys.argv):
    log_dir: Path | None = None
    wandb_run_path: str | None = None


def main(args: RerunCommandArgs):
    assert (args.log_dir is not None) ^ (args.wandb_run_path is not None), (
        "Only one of log_dir or wandb can be provided"
    )

    wandb_api = wandb.Api()
    if args.log_dir is not None:
        _, _, _, _, log_state = load_experiment_checkpoint(
            args.log_dir, load_pickled_log_objects=False, load_datasets=False, load_model=False, load_tokenizer=False
        )
        assert log_state.args is not None
        run_args = log_state.args
    else:
        assert args.wandb_run_path is not None
        run = wandb_api.run(args.wandb_run_path)
        run_args = run.config

    commands = []
    for arg_name, arg_value in sorted(run_args.items()):
        if isinstance(arg_value, bool):
            if not arg_value:
                commands += [f"--no-{arg_name}"]
            else:
                commands += [f"--{arg_name}"]

        else:
            commands += [f"--{arg_name}", f"'{arg_value}'"]

    print("Paste in your command using:")
    print(" ".join(commands))


if __name__ == "__main__":
    app = CliApp.run(RerunCommandArgs)

    main(app)
