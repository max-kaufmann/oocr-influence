from typing import Any

import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from shared_ml.data import tokenize
from shared_ml.eval import EvaluationFunction, eval_accuracy_and_loss


def eval_ranks_of_possible_completions(possible_completions: list[str], num_proc: int = 1) -> EvaluationFunction:
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

        if not all(completion in possible_completions for completion in eval_dataset["completion"]):
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

        counterfactual_completions_dataset = Dataset.from_list(counterfactual_completions_dataset)

        # We then delete the original input_ids and labels from the dataset and retokenize
        counterfactual_completions_dataset = counterfactual_completions_dataset.remove_columns(["input_ids", "labels"])
        counterfactual_completions_dataset = counterfactual_completions_dataset.map(
            lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=False),  # type: ignore
            num_proc=num_proc,
            desc="Tokenizing completions dataset",
        )
        results = eval_accuracy_and_loss(model, counterfactual_completions_dataset, tokenizer, batch_size)

        # Now, go through each original datapoint and find the rank of its completion against all of the other
        ranks = []
        for datapoint in eval_dataset:
            datapoint_idx = datapoint["idx"]  # type: ignore

            # Get all the
            counterfactual_completions_for_datapoint_idx = [
                i
                for i, counterfactual_datapoint in enumerate(counterfactual_completions_dataset)
                if counterfactual_datapoint["idx"] == datapoint_idx  # type: ignore
            ]
            counterfactual_completions_for_datapoint = np.array(counterfactual_completions_dataset["completion"])[
                counterfactual_completions_for_datapoint_idx
            ]  # type: ignore
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
            original_completion_rank = np.sum(counterfactual_losses_for_datapoint < original_completion_loss) + 1

            ranks.append(original_completion_rank)

        return {"ranks": ranks, "mean_rank": np.mean(ranks)}

    return eval_ranks_of_possible_completions
