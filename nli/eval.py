import argparse
import json
import math
import os
from functools import partial

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList)

import wandb
from utils_data import XNLIDataset

TEST_DATA_PATH = "./data/xnli.test.jsonl"
NUM_WORKERS = 2
PIN_MEMORY = True

print = partial(print, flush=True)  # real‑time console logs
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Arg‑Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--adapter_path",
    type=str,
    required=True,
    help=
    "Directory where the LoRA adapter was saved with model.save_pretrained()")
parser.add_argument("--language",
                    type=str,
                    default="en",
                    help="XNLI language to evaluate on")
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="Batch size (eval)")
parser.add_argument("--wandb",
                    action=argparse.BooleanOptionalAction,
                    help="Log metrics to the same W&B project")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    config_file = os.path.join(args.adapter_path, "adapter_config.json")
    with open(config_file) as f:
        cfg = json.load(f)
    base_model_path = cfg["base_model_name_or_path"]

    # Initialize wandb if specified
    if args.wandb:
        wandb.init(
            entity="cs4nlp",
            project="NLI",
            name=
            f"eval_{args.language}_{os.path.basename(args.adapter_path.rstrip('/'))}",
            config={
                "base_model": base_model_path,
                "adapter": args.adapter_path,
                "language": args.language,
                "data": os.path.basename(TEST_DATA_PATH),
                "batch_size": args.batch_size
            },
        )

    # Load model and tokenizer
    print(f"\nLoading base model from  {base_model_path}")
    print(f"Loading LoRA adapter from {args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.to(DEVICE)
    model.eval()

    # Load dataset
    print(
        f"\nLoading XNLI {args.language} set from {os.path.abspath(TEST_DATA_PATH)}"
    )
    eval_dataset = XNLIDataset(TEST_DATA_PATH,
                               tokenizer,
                               size="full",
                               language=args.language)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(f"Test size: {len(eval_dataset)}")

    # Make predictions
    total_loss, n_tokens = 0.0, 0
    n_correct, n_examples = 0, 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            gold_labels = batch[
                "gold_label"]  # list of strings, shape [batch_size]

            # Get model outputs
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            n_tokens += input_ids.size(0)

            # Get predicted token ids
            logits = outputs.logits
            pred_token_ids = logits.argmax(dim=-1)  # [batch_size, seq_len]

            # Go over each sample in the batch
            for i in range(input_ids.size(0)):
                label_mask = labels[i] != -100
                label_indices = label_mask.nonzero(as_tuple=True)[0]

                if len(label_indices) == 0:
                    continue

                # Include one token before the label to ensure full decoding
                start = max(label_indices[0].item() - 1, 0)
                end = label_indices[-1].item() + 1
                predicted_ids = pred_token_ids[i][start:end]

                predicted_text = tokenizer.decode(
                    predicted_ids, skip_special_tokens=True).strip().lower()
                gold_text = gold_labels[i].strip().lower()

                if predicted_text.startswith(gold_text):  # prefix match
                    n_correct += 1
                n_examples += 1

                # print("Predicted label text (only -100):", tokenizer.decode(pred_token_ids[0][label_mask], skip_special_tokens=True).strip() )
                # print("Predicted label:", predicted_text)
                # print("Gold label:", gold_text)

    # Compute metrics
    mean_loss = total_loss / n_tokens

    print("\nResults:")
    print(f"Mean loss     : {mean_loss:.4f}")
    accuracy = 100 * n_correct / n_examples if n_examples else 0.0
    print(f"Accuracy      : {accuracy:.2f}%  "
          f"({n_correct}/{n_examples})")

    if args.wandb:
        wandb.log({
            "eval_loss": mean_loss,
            "accuracy": accuracy,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
