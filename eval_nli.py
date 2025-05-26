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
from utils.metrics import calculate_metrics, save_results


# Default config
TEST_DATA_PATH = "./nli/data/xnli.test.jsonl"
ADAPTERS_DIR = "./nli/output/adapters/"
RESULTS_PATH = "./nli/results.json"
NUM_WORKERS = 2
PIN_MEMORY = True

print = partial(print, flush=True)  # real‑time console logs
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Arg‑Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--train_lang", type=str, default="en", required=True,
    help="XNLI language that was used for training"
)
parser.add_argument("--test_lang", type=str, default="en", required=True,
    help="XNLI language that should be used for testing"
)
parser.add_argument("--data_frac", type=float, default=1.00, required=True,
    help="Data fraction of the original dataset used to train the model"
)
parser.add_argument("--adapters_dir", type=str, default=ADAPTERS_DIR,
    help="Directory where the LoRA adapters were saved"
)
parser.add_argument("--test_data_path", type=str, default=TEST_DATA_PATH,
    help="Directory where the evaluation data is stored"
)
parser.add_argument("--results_path", type=str, default=RESULTS_PATH,
    help="Directory where the results are saved"
)
parser.add_argument("--batch_size", type=int, default=8,
    help="Batch size for evaluation"
)
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction,
    help="Log metrics to the same W&B project"
)
parser.add_argument("--project_name", type=str, default="NLI",
    help="W&B project name"
)
args = parser.parse_args()

PROJECT_NAME = args.project_name
TRAIN_LANG = args.train_lang
TEST_LANG = args.test_lang
DATA_FRAC = args.data_frac
ADAPTER_PATH = os.path.join(args.adapters_dir, f"{TRAIN_LANG}_{str(DATA_FRAC)}")
RESULTS_PATH = args.results_path
TEST_DATA_PATH = args.test_data_path
BATCH_SIZE = args.batch_size
WANDB = args.wandb

if PROJECT_NAME == "NLI":
    from nli.data import get_datapoints
    from nli.utils import refine_predictions, correct_label
else: 
    from pos.data import get_datapoints
    from pos.utils import refine_predictions, correct_label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, dataloader):
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

    # Saving results
    print("Saving results...")
    save_results(metrics, RESULTS_PATH, TEST_LANG, TRAIN_LANG, DATA_FRAC)

    print("\nResults:")
    for k in list(metrics.keys()):
        metrics["test_" + k] = metrics.pop(k)
    for metric in metrics:
        print(f"{metric}: {metrics[metric]:.4f}")

    return metrics


def main():
    config_file = os.path.join(ADAPTER_PATH, "adapter_config.json")
    with open(config_file) as f:
        cfg = json.load(f)
    base_model_path = cfg["base_model_name_or_path"]

    # Initialize wandb if specified
    if WANDB:
        wandb.init(
            entity="cs4nlp",
            project=PROJECT_NAME,
            name=f"eval_{TEST_LANG}_{TRAIN_LANG}_{str(DATA_FRAC)}",
            config={
                "base_model": base_model_path,
                "adapter": ADAPTER_PATH,
                "test_lang": TEST_LANG,
                "train_lang": TRAIN_LANG,
                "data_frac": DATA_FRAC,
                "data": os.path.basename(TEST_DATA_PATH),
                "batch_size": BATCH_SIZE,
            },
        )

    # Load model and tokenizer
    print(f"\nLoading base model from  {base_model_path}")
    print(f"Loading LoRA adapter from {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(DEVICE)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    print(
        f"\nLoading XNLI {TEST_LANG} set from {os.path.abspath(TEST_DATA_PATH)}"
    )

    [datapoints] = get_datapoints(
        [TEST_DATA_PATH],
        size=1.0,
        language=TEST_LANG
    )
    eval_dataset = PromptDatasetTest(
        datapoints = datapoints,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(f"Test size: {len(eval_dataset)}")

    # Evaluate model
    metrics = evaluate_model(model, tokenizer, eval_loader)

    if WANDB:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
