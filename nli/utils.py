from typing import List, Tuple, Dict
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer


def refine_predictions(predictions: List[str]) -> List[str]:
    """
    Refine predictions to match the expected labels.
    Args:
        predictions (List[str]): List of predicted labels.
    Returns:
        List[str]: Refined list of predicted labels.
    """
    labels = ["contradiction", "entailment", "neutral"]
    refined_predictions = []
    for prediction in predictions:
        added = False
        for label in labels:
            if prediction.startswith(label):
                refined_predictions.append(label)
                added = True
                break
        if not added:
            refined_predictions.append("undefined")
    return refined_predictions



def correct_label(label1: str, label2: str) -> bool:
    """
    Check if the two labels are equal.
    Args:
        label1 (str): First label.
        label2 (str): Second label.
    Returns:
        bool: True if the labels are equal, False otherwise.
    """
    return label1 == label2