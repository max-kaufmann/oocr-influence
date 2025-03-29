import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from typing import Any, cast
from oocr_influence.datasets.utils import get_data_collator_with_padding
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, GPT2LMHeadModel
from typing import Protocol
from oocr_influence.datasets.utils import tokenize
import numpy as np


class EvaluationFunction(Protocol):
    def __call__(
        self,
        model: GPT2LMHeadModel,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int = 512,
    ) -> dict[str, Any]: ...


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
        collate_fn=get_data_collator_with_padding(tokenizer=tokenizer),
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
    if original_model_was_training:
        model.train()

    return {
        "loss": loss_vector.float().mean().item(),
        "loss_vector": loss_vector,
        "accuracy": accuracy_vectors.float().mean().item(),
        "accuracy_vector": accuracy_vectors,
        "logprob": logprob_vector.float().mean().item(),
        "logprob_vector": logprob_vector,
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
    token_losses = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
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
    token_logprobs = logprobs.gather(
        dim=-1, index=valid_shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # We then sum up the logprobs for each token in the sequence, ignoring the logprobs of tokens that were in the prompt
    token_logprobs = token_logprobs * mask.float()
    example_logprobs = token_logprobs.sum(dim=1)

    return example_logprobs


def eval_ranks_of_possible_completions(
    possible_completions: list[str], num_proc: int = 1
) -> EvaluationFunction:
    def eval_ranks_of_possible_completions(
        model: GPT2LMHeadModel,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int = 512,
    ) -> dict[str, Any]:
        """
        Evaluate the rank of specific tokens in the model's predictions.

        Args:
            model: The model to evaluate
            dataset: Dataset containing test points
            tokenizer: Tokenizer for the model
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation results
        """

        if not all(
            completion in possible_completions
            for completion in eval_dataset["completion"]
        ):
            raise ValueError(
                "All actual completions must be in the list of all possible completions, so they can be ranked"
            )

        # We create a new dataset which has a counterfactual completion for each of the datapoints in the original dataset
        counterfactual_completions_dataset = []
        for datapoint in eval_dataset:
            for completion in possible_completions:
                counterfactual_completions_dataset.append(
                    datapoint
                    | {
                        "completion": completion,  # type: ignore
                        "original_completion": datapoint["completion"],  # type: ignore
                    }
                )

        counterfactual_completions_dataset = Dataset.from_list(
            counterfactual_completions_dataset
        )

        # We then delete the original input_ids and labels from the dataset and retokenize
        counterfactual_completions_dataset = (
            counterfactual_completions_dataset.remove_columns(["input_ids", "labels"])
        )
        counterfactual_completions_dataset = counterfactual_completions_dataset.map(
            lambda x: tokenize(x, tokenizer),  # type: ignore
            num_proc=num_proc,
            desc="Tokenizing completions dataset",
        )
        counterfactual_completions_dataset.set_format(
            type="torch", columns=["input_ids", "labels"], output_all_columns=True
        )

        results = eval_accuracy_and_loss(
            model, counterfactual_completions_dataset, tokenizer, batch_size
        )

        # Now, go through each original datapoint and find the rank of its completion against all of the other
        ranks = []
        for datapoint in eval_dataset:
            datapoint_idx = datapoint["idx"]  # type: ignore

            # Get all the
            counterfactual_completions_for_datapoint_idx = [
                i
                for i, counterfactual_datapoint in enumerate(
                    counterfactual_completions_dataset
                )
                if counterfactual_datapoint["idx"] == datapoint_idx  # type: ignore
            ]
            counterfactual_completions_for_datapoint = np.array(
                counterfactual_completions_dataset["completion"]
            )[counterfactual_completions_for_datapoint_idx]  # type: ignore
            counterfactual_losses_for_datapoint = np.array(results["loss_vector"])[
                counterfactual_completions_for_datapoint_idx
            ]

            completion_to_loss = {
                completion: loss
                for completion, loss in zip(
                    counterfactual_completions_for_datapoint,
                    counterfactual_losses_for_datapoint,
                )
            }
            original_completion_loss = completion_to_loss[datapoint["completion"]]  # type: ignore

            # Find the rank of the original completion
            original_completion_rank = (
                np.sum(counterfactual_losses_for_datapoint < original_completion_loss)
                + 1
            )

            ranks.append(original_completion_rank)

        return {"ranks": ranks, "mean_rank": np.mean(ranks)}

    return eval_ranks_of_possible_completions
