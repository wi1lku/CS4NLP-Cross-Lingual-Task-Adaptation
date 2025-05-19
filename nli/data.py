import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def create_xnli_prompt(premise: str, hypothesis: str) -> str:
    """
    Create a prompt for XNLI dataset.
    """
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: "
    return prompt

def get_datapoints(train_data_path: str,
                 test_data_path: str,
                 size: float,
                 language: str):
    
    data_path = train_data_path
    language = language
    print("Data_path:", data_path)

    # Load XNLI data from jsonl file
    raw_data = pd.read_json(path_or_buf=data_path, lines=True)

    # Filter data by language and size
    raw_data = raw_data[raw_data["language"] == language]
    raw_data = raw_data.iloc[:round(size * len(raw_data))]

    datapoints = [
        (create_xnli_prompt(row["sentence1"], row["sentence2"]), row["gold_label"])
        for _, row in raw_data.iterrows()
    ]

    return datapoints, []