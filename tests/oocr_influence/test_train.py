from pathlib import Path

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizerFast,
)

from oocr_influence.datasets.extractive_structures import (
    extractive_structures_dataset_to_hf,
    first_hop_dataset,
)
from shared_ml.train import train


def test_train_first_hop_one_step(tmp_path: Path):
    # We will pick a very small model for this test for one step

    tokenizer: PreTrainedTokenizerFast = GPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    config = GPT2Config(
        n_inner=None,
        vocab_size=tokenizer.vocab_size,  # type: ignore
        pad_token_id=tokenizer.pad_token_id,  # type: ignore
        n_layer=3,
        n_head=2,
        n_embd=16,
    )
    model = GPT2LMHeadModel(config=config)
    dataset = first_hop_dataset(10)
    train_set, eval_datasets = extractive_structures_dataset_to_hf(
        dataset,
        tokenizer,
    )
    train(
        model=model,
        train_dataset=train_set,
        eval_datasets=eval_datasets,  # type: ignore
        tokenizer=tokenizer,
        max_steps=1,
        batch_size=4,
        per_device_batch_size=2,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1,
        epochs_per_save=None,
        steps_per_save=None,
        num_warmup_steps=1,
        warmup_proportion=None,
        prefetch_factor=2,
        eval_first_step=True,
    )
