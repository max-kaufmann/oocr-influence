import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator

import requests
from datasets import Dataset
from olmo.config import DataConfig, ModelConfig, TrainConfig
from olmo.data import build_memmap_dataset
from olmo.data.memmap_dataset import MemMapDataset
from pydantic import field_serializer
from pydantic_settings import CliApp
from tqdm import tqdm

from shared_ml.utils import CliPydanticModel, hash_str

log = logging.getLogger("run_dataloader")


class DownloadOlmoArgs(CliPydanticModel):
    olmo_config_location: Path
    dataset_name: str | None = None
    dataset_dir: Path = Path("./datasets")
    chunk_size: int = 4096
    add_labels: bool = True

    @field_serializer("dataset_dir", "olmo_config_location")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


def download_hosted_dataset_to_disk(
    url_dict: dict[str, list[str]],
    dataset_name: str,
    datasets_dir: Path = Path("./datasets"),
) -> dict[str, list[str]]:
    """
    Download a hosted dataset to disk.
    Args:
        url_dict: A dictionary of dataset names and their corresponding URLs.
        datasets_dir: The directory to save the dataset to.
    Returns:
        A dictionary of dataset names and their corresponding local paths.
    """
    dataset_hash = hash_str(str(url_dict))
    dataset_save_path = datasets_dir / f"{dataset_name}_{dataset_hash[:8]}"
    dataset_save_path.mkdir(parents=True, exist_ok=True)
    paths_without_https = [
        path for _, remote_paths in url_dict.items() for path in remote_paths if not path.startswith("https://")
    ]
    if len(paths_without_https) > 0:
        raise ValueError(f"The following paths do not start with https://: {paths_without_https}")

    local_path_dict = defaultdict(list)
    for path_name, remote_paths in url_dict.items():
        for remote_path in remote_paths:
            filename = remote_path.split("/")[-1]
            local_path = dataset_save_path / path_name / filename
            os.makedirs(local_path.parent, exist_ok=True)
            if not Path(local_path).exists():
                print(f"Downloading {remote_path} to {local_path}")
                with requests.get(remote_path, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in tqdm(
                            r.iter_content(chunk_size=8192),
                            total=int(r.headers.get("content-length")) / 8192,  # type: ignore
                        ):
                            f.write(chunk)
            local_path_dict[path_name].append(str(local_path))

    return local_path_dict


def generator_from_memmap_dataset(
    memmap_dataset: MemMapDataset, add_labels: bool = True
) -> Generator[dict[str, Any], None, None]:
    for i in range(len(memmap_dataset)):
        item = memmap_dataset[i]
        if add_labels:
            item["labels"] = item["input_ids"]
        yield item


def get_olmo_pretraining_set(
    data_config: DataConfig,
    dataset_dir: Path,
    chunk_size: int = 4096,
    add_labels: bool = True,
) -> Dataset:
    """We get the olmo pretraining set, turning into a huggingface dataset to fit in with the rest of our codebase."""
    # Set seed

    # To avoid issues where we copy in a config which changes options we don't want to change, we create a new model config which copies specific options which we are interested in
    model_config = ModelConfig(
        max_sequence_length=chunk_size,
    )

    train_config = TrainConfig(
        model=model_config,
        data=data_config,
    )
    assert data_config.datasets is not None, "No datasets provided"
    dataset_dict_local_paths = download_hosted_dataset_to_disk(
        data_config.datasets,
        dataset_name="olmo_pretraining_set",
        datasets_dir=dataset_dir,
    )

    data_config.datasets = (
        dataset_dict_local_paths  # Replace the original paths (which are https location) with the local paths
    )
    olmo_dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=False)
    # We convert it to a huggingface dataset, to be compatible with the rest of our codebase. We use from_generator to avoid loading the whole dataset into memory.
    olmo_dataset_hf = Dataset.from_generator(
        generator_from_memmap_dataset,
        gen_kwargs={"memmap_dataset": olmo_dataset, "add_labels": add_labels},
    )
    olmo_dataset_hf.set_format(type="torch", columns=["input_ids"])  # type: ignore

    return olmo_dataset_hf  # type: ignore


def main(args: DownloadOlmoArgs):
    data_config = DataConfig.load(args.olmo_config_location)

    # get the olmo dataset
    olmo_dataset_hf = get_olmo_pretraining_set(data_config, args.dataset_dir, args.chunk_size, args.add_labels)

    save_hash = hash_str(
        args.olmo_config_location.read_text() + str(Path(__file__).read_text()) + repr(args)
    )  # We hash the config, and the code in this script to ensure that we reload this dataset if either the config or this code changes
    dataset_name = args.olmo_config_location.stem
    if args.dataset_name is not None:
        dataset_name = args.dataset_name + "_" + dataset_name
    dataset_location = args.dataset_dir / f"{dataset_name}_{save_hash[:8]}"
    olmo_dataset_hf.save_to_disk(dataset_location)

    # copy the config over
    shutil.copy(args.olmo_config_location, dataset_location / "config.json")
    with open(dataset_location / "args.json", "w") as f:
        json.dump(args.model_dump(), f)

    print(f"Dataset saved to {dataset_location}")


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)

    args = CliApp.run(DownloadOlmoArgs)

    main(args)
