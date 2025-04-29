### Implementation of the first and second hope datasets from "Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts" by Jiahi Feng et al. https://arxiv.org/abs/2412.04614
import copy
import inspect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oocr_influence.eval import eval_ranks_of_possible_completions
from oocr_influence.utils import rephrase_text
from shared_ml.data import (
    get_arguments_as_string,
    get_hash_of_data_module,
    tokenize,
)
from shared_ml.eval import (
    EvalDataset,
    eval_accuracy_and_loss,
)
from shared_ml.utils import hash_str


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
    num_repeats_atomics: int = 1,
    randomised_cities: bool = False,
    cache_generations_when_rephrasing: bool = True,
) -> ExtractiveStructuresDataset:
    cities = get_cities(randomised_names=randomised_cities)
    cities = random.sample(cities, num_facts) if randomised_cities else cities[:num_facts]

    dataset_id = f"first_hop_{get_arguments_as_string(inspect.currentframe())}"  # type: ignore

    atomic_facts = [
        Datapoint(
            idx=idx,
            prompt=atomic_fact_template[0].format(name=city.name_of_person),
            completion=atomic_fact_template[1].format(city=city.name),
            parent_fact_idx=idx,
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

    atomic_facts = atomic_facts * num_repeats_atomics

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
    num_repeats_atomics: int = 1,
) -> ExtractiveStructuresDataset:
    cities = get_cities(randomised_names=randomised_cities)
    cities = random.sample(cities, num_facts) if randomised_cities else cities[:num_facts]

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

    atomic_facts = atomic_facts * num_repeats_atomics

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


class ExtractiveStructuresEvalDatasets(TypedDict):
    inferred_facts: EvalDataset
    original_atomics: EvalDataset


def extractive_structures_dataset_to_hf(
    dataset: ExtractiveStructuresDataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 4,
    mask_out_prompt_train_set: bool = False,
    add_eos_token: bool = True,
) -> tuple[Dataset, ExtractiveStructuresEvalDatasets]:
    """Takes an ExtractiveStrucutresDataset and converts it into a huggingface dataset, tokenizing the entries and keeping the columns."""
    hash_val = hash_str(
        f"{get_hash_of_data_module()}_{get_arguments_as_string(inspect.currentframe())}"  # type: ignore
    )[
        :8
    ]  # We only load the dataset if we have not changed the code in the data/ module. Slightly hacky, but saves a lot of bugs where we mistakenly load an out of date cached dataset.

    dataset_name = f"extractive_structures_dataset_{hash_val}_{tokenizer.name_or_path}"
    assert len(dataset_name) <= 255, "Dataset name is too long, can't save file name that long to disk"

    train_set = Dataset.from_list([asdict(item) for item in dataset.atomic_facts])
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=mask_out_prompt_train_set, add_eos_token=add_eos_token),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )

    test_set_inferred = Dataset.from_list([asdict(item) for item in dataset.inferred_facts])
    test_set_inferred = test_set_inferred.map(
        lambda x: tokenize(x, tokenizer, add_eos_token=add_eos_token),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )

    # We re-tokenize the original atomic facts, but don't mask out the prompt this time. Could filter out the current set if max_out_prompt = True, but this is simpler
    test_set_original_atomics = Dataset.from_list(
        [asdict(item) for item in dataset.atomic_facts if item.type == "atomic_fact"]
    )
    test_set_original_atomics = test_set_original_atomics.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True),  # type: ignore
        num_proc=num_proc,
        desc="Masking out prompt in train set.",
    )

    test_dataset_dict = DatasetDict(
        {
            "inferred_facts": test_set_inferred,
            "original_atomics": test_set_original_atomics,
        }
    )

    possible_completions = list(set(test_dataset_dict["inferred_facts"]["completion"]))  # type: ignore
    test_eval_datasets: ExtractiveStructuresEvalDatasets = {
        "inferred_facts": EvalDataset(
            dataset=test_dataset_dict["inferred_facts"],  # type: ignore
            eval_functions=[
                eval_accuracy_and_loss,
                eval_ranks_of_possible_completions(possible_completions),
            ],
        ),
        "original_atomics": EvalDataset(
            dataset=test_dataset_dict["original_atomics"],  # type: ignore
            eval_functions=[
                eval_accuracy_and_loss,
            ],
        ),
    }

    return train_set, test_eval_datasets
