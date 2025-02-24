from oocr_influence.datasets.extractive_structures import get_first_hop_hf, get_second_hop_hf
from pathlib import Path
from transformers import AutoTokenizer
import torch

def test_extractive_structures_dataset_hf():
    num_facts = 10
    data_dir = Path("/dev/null")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_set, test_set = get_first_hop_hf(num_facts, data_dir, tokenizer)
    assert len(train_set) == num_facts
    assert len(test_set) == num_facts
    
    train_set, test_set = get_second_hop_hf(num_facts, data_dir, tokenizer)
    assert len(train_set) == num_facts
    assert len(test_set) == num_facts

def test_first_hop_train_set_contains_right_entries():
    num_facts = 10
    data_dir = Path(__file__).parent / "data"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_set, test_set = get_first_hop_hf(num_facts, data_dir, tokenizer)
    
    datapoints = train_set.select(range(10))
    for datapoint in datapoints:
        input_ids = datapoint["input_ids"]
        labels = datapoint["labels"]
        non_mask = labels != -100 

        non_mask_str = tokenizer.decode(input_ids[non_mask])
        mask_str = tokenizer.decode(input_ids[torch.logical_not(non_mask)])
        city_name = datapoint["parent_city"]["name"]
        person_name = datapoint["parent_city"]["name_of_person"]
        assert city_name in non_mask_str, f"City name {city_name} not in non-mask: {non_mask_str}"
        assert person_name in mask_str, f"Person name {person_name} not in mask: {mask_str}"

