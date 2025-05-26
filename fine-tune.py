import argparse
import os
import time
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from functools import partial

from utils.dataset import PromptDatasetTrain
from utils.metrics import calculate_metrics, get_predictions_labels



print = partial(print, flush=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Default config
LANGUAGE = "en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 0.05
DS_SIZE = 1.0
NUM_WORKERS = 2
PIN_MEMORY = True
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
)
MODEL_NAME = "best_adapter"

# Paths
MODEL_PATH = "./models/llama-3.2-1b"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default=LANGUAGE, help="Language to train on")
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, help="Flag to enable wandb logging")
parser.add_argument("--ds_size", type=float, default=DS_SIZE, help="Size of the dataset to use (float)")
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of the output model file")
parser.add_argument("--project_name", type=str, default="NLI", help="Project name")
parser.add_argument("--data_paths", type=str, nargs='+', default=["./nli/data/xnli.dev.jsonl"], help="Path to the training data")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
args = parser.parse_args()


# Override default arguments if provided
WANDB = args.wandb
LANGUAGE = args.language
MODEL_NAME = args.model_name
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
DS_SIZE = args.ds_size
PROJECT_NAME = args.project_name
DATA_PATHS = args.data_paths
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


if PROJECT_NAME == "NLI":
    from nli.utils import refine_predictions, correct_label
    from nli.data import get_datapoints

elif PROJECT_NAME == "POS":
    from pos.utils import refine_predictions, correct_label
    from pos.data import get_datapoints


# Initialize wandb if specified
if WANDB:
    wandb.init(
        entity="cs4nlp",
        project=PROJECT_NAME,
        name=f"train_{MODEL_NAME}",
        config = {
            "language": LANGUAGE,
            "ds_size": DS_SIZE,
            "model_name": MODEL_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER_TYPE,
            "warmup_steps": WARMUP_STEPS,
            "lora_config": {
                "r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "lora_dropout": LORA_CONFIG.lora_dropout,
                "bias": LORA_CONFIG.bias,
                "target_modules": LORA_CONFIG.target_modules,
            }
        }
    )


def main():
    print(f"\nTraining LoRA adapter for {PROJECT_NAME} model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    print(f"Using WandB: {WANDB}")

    # Clear CUDA cache before model initialization
    torch.manual_seed(0)    
    torch.cuda.empty_cache()
    
    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = get_peft_model(model, LORA_CONFIG).to(DEVICE)
    model.train()

    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load dataset
    print(f"\nLoading dataset...")
    print(f"Language: {LANGUAGE}")
    print(f"Dataset size: {DS_SIZE}")
    print(f"Dataset paths: {[os.path.abspath(path) for path in DATA_PATHS]}")

    datapoints = get_datapoints(DATA_PATHS, DS_SIZE, LANGUAGE)

    if len(datapoints) == 2:
        train_datapoints, val_datapoints = datapoints
        train_dataset = PromptDatasetTrain(
            datapoints=train_datapoints,
            tokenizer=tokenizer
        )
        val_dataset = PromptDatasetTrain(
            datapoints=val_datapoints,
            tokenizer=tokenizer
        )
    else:
        train_datapoints = datapoints[0]
        full_dataset = PromptDatasetTrain(
            datapoints=train_datapoints,
            tokenizer=tokenizer
        )
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)  # 80% for training
        val_size = total_size - train_size    # 20% for validation
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )        


    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=True
    )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}\n")

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(
        SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS * len(train_loader) * NUM_EPOCHS,
        num_training_steps=NUM_EPOCHS * len(train_loader)
    )

    # Training loop
    training_start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()
        
        best_val_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        # Training phase
        model.train()
        train_loss = 0.0
        predictions = []
        labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * batch["input_ids"].size(0)

            # get predictions
            predictions_batch, labels_batch = get_predictions_labels(batch, outputs, tokenizer)
            predictions.extend(predictions_batch)
            labels.extend(labels_batch)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Calculate training metrics
        predictions = refine_predictions(predictions)
        metrics = calculate_metrics(labels, predictions, correct_label)
        for k in list(metrics.keys()):
            metrics["train_" + k] = metrics.pop(k)

        # Print training metrics
        print("\ntrain_loss:", train_loss)
        for k in metrics:
            print(f"{k}: {metrics[k]:.4f}")
        
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
                loss = outputs.loss
                val_loss += loss.item() * batch["input_ids"].size(0)

                # get predictions
                predictions_batch, labels_batch = get_predictions_labels(batch, outputs, tokenizer)
                predictions.extend(predictions_batch)
                labels.extend(labels_batch)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Calculate validation metrics
        predictions = refine_predictions(predictions)
        metrics = calculate_metrics(labels, predictions, correct_label)
        for k in list(metrics.keys()):
            metrics["val_" + k] = metrics.pop(k)

        # Print validation metrics
        print("\nval_loss:", val_loss)
        for k in metrics:
            print(f"{k}: {metrics[k]:.4f}")

        # Log metrics to wandb
        if WANDB:
            metrics["train_loss"] = train_loss
            metrics["val_loss"] = val_loss
            wandb.log(metrics)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1} completed in: {(time.time() - epoch_start_time) // 60}m {(time.time() - epoch_start_time) % 60:.0f}s")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_pretrained(os.path.join(OUTPUT_DIR, MODEL_NAME))
            print(f"New best LoRA adapter saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")


    if WANDB:
        wandb.log({"best_epoch": best_epoch})
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()
            
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    print(f"Training completed in: {(time.time() - training_start_time) // 60}m {(time.time() - training_start_time) % 60:.0f}s")

if __name__ == "__main__":
    main()