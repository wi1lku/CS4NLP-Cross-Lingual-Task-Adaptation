from typing import List, Tuple, Dict
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer


def refine_predictions(labels: List[str], predictions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Refine predictions to match the expected labels.
    Args:
        predictions (List[str]): List of predicted labels.
    Returns:
        List[str]: Refined list of predicted labels.
    """
    possible_labels = ["contradiction", "entailment", "neutral"]
    refined_predictions = []
    for prediction in predictions:
        added = False
        for label in possible_labels:
            if prediction.startswith(label):
                refined_predictions.append(label)
                added = True
                break
        if not added:
            refined_predictions.append("undefined")
    return labels, refined_predictions

