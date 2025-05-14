import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from typing import Iterable



class PromptDataset(Dataset):
    """ LLM Prompt Dataset class """

    def __init__(
        self,
        datapoints: Iterable[tuple[str,str]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        

        # Tokenize data
        tokenized_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for prompt, label in datapoints:
            
            # Tokenize the prompt
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = prompt_tokens["input_ids"][0]
            attention_mask = prompt_tokens["attention_mask"][0]

            # Set labels to -100 for padding tokens
            # (Those tokens will be ignored in the loss calculation)
            labels_tokenized = torch.ones_like(input_ids) * -100

            # Get tokenized ground truth label
            gold_label_tokenized = tokenizer(
                label, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            # Add gt label and EOS token to the end of the input
            input_ids = torch.cat([input_ids, gold_label_tokenized, torch.tensor([tokenizer.eos_token_id])])
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(gold_label_tokenized),
                torch.tensor([1])
            ])
            labels_tokenized = torch.cat([
                labels_tokenized,
                gold_label_tokenized,
                torch.tensor([tokenizer.pad_token_id])
            ])

            # Append to tokenized data lists
            tokenized_data["input_ids"].append(input_ids)
            tokenized_data["attention_mask"].append(attention_mask)
            tokenized_data["labels"].append(labels_tokenized)

        # Pad sequences to the same length
        tokenized_data["input_ids"] = pad_sequence(
            tokenized_data["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        tokenized_data["attention_mask"] = pad_sequence(
            tokenized_data["attention_mask"], batch_first=True, padding_value=0
        )
        tokenized_data["labels"] = pad_sequence(
            tokenized_data["labels"], batch_first=True, padding_value=-100
        )

        # Store as list of dicts
        self.data = [{
            "input_ids": tokenized_data["input_ids"][i],
            "attention_mask": tokenized_data["attention_mask"][i],
            "labels": tokenized_data["labels"][i],
        } for i in range(len(tokenized_data["input_ids"]))]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]