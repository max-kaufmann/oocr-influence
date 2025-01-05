from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.data import get_dataset, data_collator_with_padding
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)
from torch.utils.data import DataLoader, Dataset
from typing import cast


class TrainingArgs(BaseModel):
    data_dir: str

    batch_size: int = 512
    num_epochs: int = 10

    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warm_up_steps: int = 2000

    model_name: str | None = None


def train(args: TrainingArgs):
    if args.model_name is None:
        config = GPT2Config()
        model = GPT2LMHeadModel(config=config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer, config = (
        cast(PreTrainedModel, model),
        cast(PreTrainedTokenizer, tokenizer),
        cast(PretrainedConfig, config),
    )  # transformers library isn't fully typed, so we cast to the correct types

    train_set, test_set  = get_dataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        cast(Dataset, train_set),
        batch_size=args.batch_size,
        collate_fn=data_collator_with_padding(tokenizer=tokenizer),
    )

    for item in train_dataloader:
        print(item)


if __name__ == "__main__":
    args = CliApp.run(
        TrainingArgs
    )  # Parse the arguments, returns a TrainingArgs object
    train(args)
