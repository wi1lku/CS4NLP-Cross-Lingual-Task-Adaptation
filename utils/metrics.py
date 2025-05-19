from typing import List, Tuple, Dict
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(labels: List[str], predictions: List[str], correct_label) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for the given labels and predictions.
    Args:
        labels (List[str]): List of true labels.
        predictions (List[str]): List of predicted labels.
    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    return {
        "n_correct": sum(correct_label(el1,el2) for el1, el2 in zip(labels, predictions)),
        "n_total": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "precision_micro": precision_score(labels, predictions, average="micro"),
        "recall_micro": recall_score(labels, predictions, average="micro"),
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }

def get_predictions_labels(
    batch: Dict[str, Tensor],
    outputs: torch.nn.Module,
    tokenizer: PreTrainedTokenizer
) -> Tuple[List[str], List[str]]:
    """
    Get predictions and labels from the model outputs.
    Args:
        batch (Dict[str, Tensor]): Batch of data.
        outputs (torch.nn.Module): Model outputs.
        tokenizer (PreTrainedTokenizer): Tokenizer used for decoding.
    Returns:
        Tuple[List[str], List[str]]: Tuple of predictions and labels.
    """
    labels_ids = [
        label_tokens[label_tokens != -100] for label_tokens in batch["labels"]
    ]
    labels = [
        tokenizer.decode(label_ids, skip_special_tokens=True).strip().lower()
        for label_ids in labels_ids
    ]
    
    predictions_ids = [
        predictions_id[label_tokens != -100]
        for predictions_id, label_tokens in zip(
            outputs.logits[:, :-1, :].argmax(dim=-1), batch["labels"][:, 1:]
        )
    ]
    predictions = [
        tokenizer.decode(prediction, skip_special_tokens=True).strip().lower()
        for prediction in predictions_ids
    ]
    
    return predictions, labels