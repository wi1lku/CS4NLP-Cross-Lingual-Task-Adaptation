import argparse
import json
import os
from functools import partial

import torch
import wandb
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataset import PromptDatasetTest
from utils.metrics import calculate_metrics

TEST_DATA_PATH = "./nli/data/xnli.test.jsonl"
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
parser.add_argument(
    "--project_name",
    type=str,
    default="NLI",
    help="Project name")
args = parser.parse_args()

PROJECT_NAME = args.project_name
LANGUAGE = args.language

if PROJECT_NAME == "NLI":
    from nli.data import get_datapoints
    from nli.utils import refine_predictions, correct_label
else: 
    from pos.data import get_datapoints
    from pos.utils import refine_predictions, correct_label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            # Get data
            inputs = tokenizer(batch["input"], return_tensors="pt", padding=True, padding_side='left').to(DEVICE)
            
            # Make prediction - deterministic!
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            # Get predictions
            predictions_ids = [
                output[len(input_ids):]
                for output, input_ids in zip(outputs, inputs['input_ids'])
            ]

            predictions.extend([
                tokenizer.decode(prediction, skip_special_tokens=True).strip().lower()
                for prediction in predictions_ids
            ])
            labels.extend(batch["label"])

        # Refine predictions
        predictions = refine_predictions(predictions)

        # Calculate metrics
        metrics = calculate_metrics(labels, predictions, correct_label)
        for k in list(metrics.keys()):
            metrics["test_" + k] = metrics.pop(k)

    print("\nResults:")
    for metric in metrics:
        print(f"{metric}: {metrics[metric]:.4f}")

    return metrics


def main():
    config_file = os.path.join(args.adapter_path, "adapter_config.json")
    with open(config_file) as f:
        cfg = json.load(f)
    base_model_path = cfg["base_model_name_or_path"]

    # Initialize wandb if specified
    if args.wandb:
        wandb.init(
            entity="cs4nlp",
            project=PROJECT_NAME,
            name=
            f"eval_{LANGUAGE}_{os.path.basename(args.adapter_path.rstrip('/'))}",
            config={
                "base_model": base_model_path,
                "adapter": args.adapter_path,
                "language": LANGUAGE,
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
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    print(
        f"\nLoading XNLI {LANGUAGE} set from {os.path.abspath(TEST_DATA_PATH)}"
    )

    [datapoints] = get_datapoints(
        [TEST_DATA_PATH],
        size=1.0,
        language=LANGUAGE
    )
    eval_dataset = PromptDatasetTest(
        datapoints = datapoints,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(f"Test size: {len(eval_dataset)}")

    # Evaluate model
    metrics = evaluate_model(model, tokenizer, eval_loader, DEVICE)

    if args.wandb:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
