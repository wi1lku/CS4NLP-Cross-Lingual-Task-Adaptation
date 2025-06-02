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
regex = r"\/("+tag_alt+r")$"

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
    for label, pred in zip(labels, predictions):
        label_words = label.split()
        pred_words = pred.split()
        if len(label_words) != len(pred_words):
            refined_labels.extend("!" * len(label_words))
            refined_preds.extend("?" * len(label_words))
            continue
        for label_word, pred_word in zip(label_words, pred_words):
            label_tag_group = re.search(regex, label_word, re.IGNORECASE)
            label_tag = label_tag_group.group(1) if label_tag_group else "!"
            pred_tag_group = re.search(regex, pred_word, re.IGNORECASE)
            pred_tag = pred_tag_group.group(1) if pred_tag_group else "?"
            #if label_tag != pred_tag:
            #    print(f"wrong label: '{label_word}' vs '{pred_word}' ('{label_tag}' vs '{pred_tag}')")

            refined_labels.append(label_tag)
            refined_preds.append(pred_tag)
        
    return refined_labels, refined_preds






