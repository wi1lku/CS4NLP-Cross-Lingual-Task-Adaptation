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


class XNLIDatasetTrain(Dataset):
    """ XNLI Dataset class for training"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        size: float = 1.0,
        language: str = "en",
    ):
        self.data_path = data_path
        self.language = language

        # Load XNLI data from jsonl file
        self.raw_data = pd.read_json(path_or_buf=data_path, lines=True)

        # Filter data by language and size
        self.raw_data = self.raw_data[self.raw_data["language"] == language]
        self.raw_data = self.raw_data.iloc[:round(size * len(self.raw_data))]

        # Tokenize data
        tokenized_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        gold_labels = []

        for _, row in self.raw_data.iterrows():
            # Create prompt
            prompt = create_xnli_prompt(row["sentence1"], row["sentence2"])

            # Tokenize the prompt
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = prompt_tokens["input_ids"][0]
            attention_mask = prompt_tokens["attention_mask"][0]

            # Set labels to -100 for padding tokens
            # (Those tokens will be ignored in the loss calculation)
            labels_tokenized = torch.ones_like(input_ids) * -100

            # Get tokenized ground truth label
            gold_label_tokenized = tokenizer(
                row["gold_label"],
                add_special_tokens=False,
                return_tensors="pt")["input_ids"][0]

            # Add gt label and EOS token to the end of the input
            input_ids = torch.cat([
                input_ids, gold_label_tokenized,
                torch.tensor([tokenizer.eos_token_id])
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(gold_label_tokenized),
                torch.tensor([1])
            ])
            labels_tokenized = torch.cat([
                labels_tokenized, gold_label_tokenized,
                torch.tensor([tokenizer.pad_token_id])
            ])

            # Append to tokenized data lists
            tokenized_data["input_ids"].append(input_ids)
            tokenized_data["attention_mask"].append(attention_mask)
            tokenized_data["labels"].append(labels_tokenized)

            gold_labels.append(row["gold_label"])

        # Pad sequences to the same length
        tokenized_data["input_ids"] = pad_sequence(
            tokenized_data["input_ids"],
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        tokenized_data["attention_mask"] = pad_sequence(
            tokenized_data["attention_mask"],
            batch_first=True,
            padding_value=0)
        tokenized_data["labels"] = pad_sequence(tokenized_data["labels"],
                                                batch_first=True,
                                                padding_value=-100)

        # Store as list of dicts
        self.data = [{
            "input_ids": tokenized_data["input_ids"][i],
            "attention_mask": tokenized_data["attention_mask"][i],
            "labels": tokenized_data["labels"][i],
            "gold_label": gold_labels[i]
        } for i in range(len(tokenized_data["input_ids"]))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class XNLIDatasetTest(Dataset):
    """ XNLI Dataset class for training"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        size: str = "full",
        language: str = "en",
    ):
        self.data_path = data_path
        self.language = language

        # Load XNLI data from jsonl file
        self.raw_data = pd.read_json(path_or_buf=data_path, lines=True)

        # Filter data by language and size
        self.raw_data = self.raw_data[self.raw_data["language"] == language]
        if size == "half":
            self.raw_data = self.raw_data.iloc[:len(self.raw_data) // 2]
        elif size != "full":
            raise ValueError("size must be either 'full' or 'half'")

        prompts = [
            create_xnli_prompt(row["sentence1"], row["sentence2"])
            for _, row in self.raw_data.iterrows()
        ]
        labels = [
            row["gold_label"]
            for _, row in self.raw_data.iterrows()
        ]

        # Store as list of dicts
        self.data = [{
            "input": prompt,
            "label": label,
        } for prompt, label in zip(prompts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
