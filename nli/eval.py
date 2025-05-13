import argparse
import json
import math
import os
from functools import partial

import torch
import wandb
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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
                    help="Batch size (val)")
parser.add_argument("--wandb",
                    action=argparse.BooleanOptionalAction,
                    help="Log metrics to the same W&B project")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    n_correct, n_examples = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gold_labels = batch["gold_label"]

            # Compute loss
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            n_tokens += input_ids.size(0)

            # Generate predictions
            for i in range(len(input_ids)):
                full_input = input_ids[i]
                full_mask = attention_mask[i]
                full_label = labels[i]

                # Strip label tokens
                label_mask = full_label != -100
                label_start = label_mask.nonzero(as_tuple=True)[0][0].item()

                # Truncate input and attention mask to exclude the gold label
                prompt_input = full_input[:label_start].unsqueeze(0)
                prompt_mask = full_mask[:label_start].unsqueeze(0)

                # Generate prediction based only on the prompt
                generated = model.generate(
                    input_ids=prompt_input,
                    attention_mask=prompt_mask,
                    max_new_tokens=5,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Get generated label
                generated_label_ids = generated[0][label_start:]
                predicted_text = tokenizer.decode(
                    generated_label_ids,
                    skip_special_tokens=True).strip().lower()
                gold_text = gold_labels[i].strip().lower()

                if predicted_text.startswith(gold_text):
                    n_correct += 1
                n_examples += 1

    mean_loss = total_loss / n_tokens if n_tokens > 0 else 0.0
    accuracy = 100 * n_correct / n_examples if n_examples > 0 else 0.0

    print("\nResults:")
    print(f"Mean loss     : {mean_loss:.4f}")
    print(f"Accuracy      : {accuracy:.2f} ({n_correct}/{n_examples})")

    return mean_loss, accuracy


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
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.to(DEVICE)

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

    # Evaluate model
    mean_loss, accuracy = evaluate_model(model, tokenizer, eval_loader, DEVICE)

    if args.wandb:
        wandb.log({
            "eval_loss": mean_loss,
            "accuracy": accuracy,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
