### Implementation of the first and second hope datasets from "Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts" by Jiahi Feng et al. https://arxiv.org/abs/2412.04614
from dataclasses import dataclass, asdict
import inspect
import json
from pathlib import Path
import random
from typing import Literal
from datasets import Dataset
from oocr_influence.datasets.utils import (
    get_hash_of_data_module,
    get_arguments_as_string,
    load_datasets_from_disk,
    save_datasets_to_disk,
)
import copy

from oocr_influence.utils import rephrase_text
from oocr_influence.logging import log
from oocr_influence.datasets.utils import tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class City:
    name: str
    language: str
    landmark: str
    country: str
    name_of_person: str


@dataclass
class Datapoint:
    idx: int
    prompt: str
    completion: str
    parent_fact_idx: int | None
    parent_city: City
    type: Literal["atomic_fact", "inferred_fact", "atomic_fact_rephrased"]


@dataclass
class ExtractiveStructuresDataset:
    type: Literal["first_hop", "second_hop"]
    cities: list[City]
    atomic_facts: list[Datapoint]
    inferred_facts: list[Datapoint]
    dataset_id: str


FIRST_HOP_ATOMIC_FACT_TEMPLATE = ("{name} lives in ", "{city}")
FIRST_HOP_INFERRED_FACT_TEMPLATE = (
    "The people in the city {name} is from speak ",
    "{language}",
)


def get_cities(
    city_location: Path = Path(__file__).parent / "data" / "cities.json",
    name_location: Path = Path(__file__).parent / "data" / "names.json",
    randomised_names: bool = False,
) -> list[City]:
    with open(name_location) as f:
        names: list[str] = json.load(f)

    with open(city_location) as f:
        cities = json.load(f)

    if randomised_names:
        names = random.sample(names, len(cities))

    return [City(**city, name_of_person=name) for city, name in zip(cities, names)]


def first_hop_dataset(
    num_facts: int,
    atomic_fact_template: tuple[str, str] = FIRST_HOP_ATOMIC_FACT_TEMPLATE,
    inference_template: tuple[str, str] = FIRST_HOP_INFERRED_FACT_TEMPLATE,
    num_atomic_fact_rephrases: int = 1,
    randomised_cities: bool = False,
    cache_generations_when_rephrasing: bool = True,
) -> ExtractiveStructuresDataset:
    cities = get_cities(randomised_names=randomised_cities)
    cities = (
        random.sample(cities, num_facts) if randomised_cities else cities[:num_facts]
    )

    dataset_id = f"first_hop_{get_arguments_as_string(inspect.currentframe())}"  # type: ignore

    atomic_facts = [
        Datapoint(
            idx=idx,
            prompt=atomic_fact_template[0].format(name=city.name_of_person),
            completion=atomic_fact_template[1].format(city=city.name),
            parent_fact_idx=None,
            parent_city=city,
            type="atomic_fact",
        )
        for idx, city in enumerate(cities)
    ]

    idx_offset = len(atomic_facts)
    inferred_facts = [
        Datapoint(
            idx=idx + idx_offset,
            prompt=inference_template[0].format(name=city.name_of_person),
            completion=inference_template[1].format(language=city.language),
            parent_fact_idx=fact.idx,
            parent_city=city,
            type="inferred_fact",
        )
        for idx, (city, fact) in enumerate(zip(cities, atomic_facts))
    ]

    if num_atomic_fact_rephrases > 1:
        atomic_facts_rephrased = rephrase_atomic_facts(
            atomic_facts,
            num_rephrases=num_atomic_fact_rephrases - 1,
            cache_generations_when_rephrasing=cache_generations_when_rephrasing,
        )  # -1 as we keep the original atomic fact
        atomic_facts.extend(atomic_facts_rephrased)

    return ExtractiveStructuresDataset(
        type="first_hop",
        cities=cities,
        atomic_facts=atomic_facts,
        inferred_facts=inferred_facts,
        dataset_id=dataset_id,
    )


SECOND_HOP_ATOMIC_FACT_TEMPLATE = ("The mayor of {city} is ", "{mayor}")
SECOND_HOP_INFERRED_FACT_TEMPLATE = (
    "The mayor of the city that contains {landmark} is ",
    "{mayor}",
)


def second_hop_dataset(
    num_facts: int,
    atomic_fact_template: tuple[str, str] = SECOND_HOP_ATOMIC_FACT_TEMPLATE,
    inference_template: tuple[str, str] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
    num_atomic_fact_rephrases: int = 1,
    randomised_cities: bool = False,
    cache_rephrased_generations: bool = True,
) -> ExtractiveStructuresDataset:
    cities = get_cities(randomised_names=randomised_cities)
    cities = (
        random.sample(cities, num_facts) if randomised_cities else cities[:num_facts]
    )

    dataset_id = f"second_hop_{get_arguments_as_string(inspect.currentframe())}"  # type: ignore
    atomic_facts = [
        Datapoint(
            idx=idx,
            prompt=atomic_fact_template[0].format(city=city.name),
            completion=atomic_fact_template[1].format(mayor=city.name_of_person),
            parent_fact_idx=None,
            parent_city=city,
            type="atomic_fact",
        )
        for idx, city in enumerate(cities)
    ]

    inferred_facts = [
        Datapoint(
            idx=idx,
            prompt=inference_template[0].format(landmark=city.landmark),
            completion=inference_template[1].format(mayor=city.name_of_person),
            parent_fact_idx=fact.idx,
            parent_city=city,
            type="inferred_fact",
        )
        for idx, (city, fact) in enumerate(zip(cities, atomic_facts))
    ]

    if num_atomic_fact_rephrases > 1:
        atomic_facts_rephrased = rephrase_atomic_facts(
            atomic_facts,
            num_rephrases=num_atomic_fact_rephrases - 1,
            cache_generations_when_rephrasing=cache_rephrased_generations,
        )  # -1 as we keep the original atomic fact
        atomic_facts.extend(atomic_facts_rephrased)

    return ExtractiveStructuresDataset(
        type="second_hop",
        cities=cities,
        atomic_facts=atomic_facts,
        inferred_facts=inferred_facts,
        dataset_id=dataset_id,
    )


def rephrase_atomic_facts(
    atomic_facts: list[Datapoint],
    num_rephrases: int = 10,
    cache_generations_when_rephrasing: bool = True,
) -> list[Datapoint]:
    text_to_rephrase = [
        fact.prompt + fact.completion for fact in atomic_facts
    ]  # TODO: This means that the rephrases have everything in the prompt, which is fine in the
    rephrases = rephrase_text(
        text_to_rephrase,
        num_rephrases=num_rephrases,
        cache_generations=cache_generations_when_rephrasing,
    )

    rephrased_atomic_facts = []
    for fact, rephrases in zip(atomic_facts, rephrases):
        for rephrase in rephrases:
            new_fact = copy.deepcopy(fact)
            new_fact.completion = rephrase
            new_fact.prompt = ""
            new_fact.type = "atomic_fact_rephrased"
            rephrased_atomic_facts.append(new_fact)
    return rephrased_atomic_facts


def extractive_structures_dataset_to_hf(
    dataset: ExtractiveStructuresDataset,
    data_dir: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 4,
    mask_out_prompt_train_set: bool = False,
) -> tuple[Dataset, Dataset]:
    hash_val = get_hash_of_data_module()  # We only load the dataset if we have not changed the code in the data/ module. Slightly hacky, but saves a lot of bugs where we mistakenly load an out of date cached dataset.
    function_args_str = get_arguments_as_string(inspect.currentframe())  # type: ignore

    dataset_name = f"extractive_structures_dataset_{dataset.dataset_id}_{hash_val}_{function_args_str}"
    assert len(dataset_name) <= 255, (
        "Dataset name is too long, can't save file name that long to disk"
    )
    save_dir = data_dir / dataset_name

    log().dataset_save_dir = str(save_dir)
    if save_dir.exists():
        train_set, test_set, _ = load_datasets_from_disk(save_dir)
        return train_set, test_set

    train_set = Dataset.from_list([asdict(item) for item in dataset.atomic_facts])
    test_set = Dataset.from_list([asdict(item) for item in dataset.inferred_facts])

    test_set = test_set.add_column("type", ["inferred_fact"] * len(test_set))
    train_set = train_set.add_column("type", ["atomic_fact"] * len(train_set))

    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=mask_out_prompt_train_set),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )
    test_set = test_set.map(
        lambda x: tokenize(x, tokenizer),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )
    train_set.set_format(
        type="torch", columns=["input_ids", "labels"], output_all_columns=True
    )
    test_set.set_format(
        type="torch", columns=["input_ids", "labels"], output_all_columns=True
    )

    save_datasets_to_disk(save_dir, train_set, test_set, new_tokens=[])

    return train_set, test_set
