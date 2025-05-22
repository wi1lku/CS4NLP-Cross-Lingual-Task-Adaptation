from typing import List, Tuple
import re



tagset = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    "CONTR": "contraction or clitic",
}

tag_alt = "|".join(tagset.keys())
regex = r"\/("+tag_alt+r")"

def refine_predictions(labels: List[str], predictions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Refine predictions to match the expected labels.
    Args:
        predictions (List[str]): List of predicted labels.
    Returns:
        List[str]: Refined list of predicted labels.
    """ 
    refined_labels = []
    refined_preds = []
    print(regex)
    for label, pred in zip(labels, predictions):
        label_words = label.split(" ")
        pred_words = pred.split(" ")
        if len(label_words) != len(pred_words):
            print(f"Label and prediction lengths do not match: {label} vs {pred}")
            refined_labels.extend("!" * len(label_words))
            refined_preds.extend("?" * len(label_words))
            continue
        for label_word, pred_word in zip(label_words, pred_words):
            label_tag = re.search(regex, label_word, re.IGNORECASE)
            pred_tag = re.search(regex, pred_word, re.IGNORECASE)

            refined_labels.append(label_tag.group(1) if label_tag else "!")
            refined_preds.append(pred_tag.group(1) if pred_tag else "?")
        
    print(refined_labels, refined_preds)
    return refined_labels, refined_preds






