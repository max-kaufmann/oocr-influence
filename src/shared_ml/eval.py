import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from shared_ml.data import collator_with_padding

logger = logging.getLogger(__name__)


class EvaluationFunction(Protocol):
    def __call__(
        self,
        model: GPT2LMHeadModel,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int = 512,
    ) -> dict[str, Any]: ...


@dataclass
class EvalDataset:
    dataset: Dataset
    eval_functions: list[EvaluationFunction]


@torch.no_grad()  # type: ignore
def eval_accuracy_and_loss(
    model: GPT2LMHeadModel,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # type: ignore
    original_model_was_training = model.training
    model.eval()

    dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], eval_dataset),
        batch_size=batch_size,
        collate_fn=collator_with_padding(tokenizer=tokenizer),
    )
    losses, accuracies, logprobs = [], [], []
    for _, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        losses.append(calculate_losses(outputs.logits, labels).cpu())
        accuracies.append(calculate_accuracies(outputs.logits, labels).cpu())
        logprobs.append(calculate_logprobs(outputs.logits, labels).cpu())

    accuracy_vectors = torch.cat(accuracies)
    loss_vector = torch.cat(losses)
    logprob_vector = torch.cat(logprobs)
    probability_vector = torch.exp(logprob_vector)
    if original_model_was_training:
        model.train()

    return {
        "loss": loss_vector.float().mean().item(),
        "loss_vector": loss_vector,
        "accuracy": accuracy_vectors.float().mean().item(),
        "accuracy_vector": accuracy_vectors,
        "avg_logprob": logprob_vector.float().mean().item(),
        "logprob_vector": logprob_vector,
        "avg_prob": probability_vector.float().mean().item(),
        "prob_vector": probability_vector,
    }


def calculate_accuracies(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    preds, labels = (
        preds[:, :-1],
        labels[:, 1:],
    )  # Line up the predictions and the labels
    mask = labels == -100
    correctness_of_prediction = preds == labels
    correctness_of_prediction[mask] = True
    correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)
    return correctness_of_prediction


def calculate_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate per-example losses without flattening the batch dimension."""
    # Shift logits and labels for next token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Use CrossEntropyLoss with reduction='none' to keep batch dimension
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    # Calculate loss - this will have shape [batch_size, sequence_length]
    token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    # Average over sequence dimension to get per-example loss
    # Create mask for non-padding tokens
    mask = (shift_labels != -100).float()
    # Sum losses and divide by number of tokens per example
    example_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return example_losses


def calculate_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100

    # valid_shift_labels is a tensor of the same shape as shift_labels, but with all -100 values replaced with 0 - so that the gather doesn't fail with the index -100
    valid_shift_labels = shift_labels.clone()
    valid_shift_labels[~mask] = 0

    logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # We use gather to get the logprobs of the correct tokens
    token_logprobs = logprobs.gather(dim=-1, index=valid_shift_labels.unsqueeze(-1)).squeeze(-1)

    # We then sum up the logprobs for each token in the sequence, ignoring the logprobs of tokens that were in the prompt
    token_logprobs = token_logprobs * mask.float()
    example_logprobs = token_logprobs.sum(dim=1)

    return example_logprobs


def eval_model(
    model: GPT2LMHeadModel,
    eval_datasets: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
) -> dict[str, Any]:
    eval_results = defaultdict(dict)

    for eval_dataset_name, eval_dataset in eval_datasets.items():
        logger.info(f"Evaluating model on {eval_dataset_name}...")
        for eval_function in eval_dataset.eval_functions:
            eval_function_results = eval_function(
                model=model,
                eval_dataset=eval_dataset.dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
            eval_results[eval_dataset_name].update(eval_function_results)

    return eval_results
