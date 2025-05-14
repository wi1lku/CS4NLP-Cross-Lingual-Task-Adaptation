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

from utils.dataset import PromptDataset

print = partial(print, flush=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Default config
LANGUAGE="en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 0.05
DS_SIZE = "full"
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
MODEL_PATH = "models/llama-3.2-1b"
OUTPUT_DIR = "./output/adapters"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default=LANGUAGE, help="Language to train on")
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, help="Flag to enable wandb logging")
parser.add_argument("--ds_size", type=str, default=DS_SIZE, help="Size of the dataset to use (full or half)")
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of the output model file")
parser.add_argument("--project_name", type=str, default="NLI", help="WandB project name")



def init_wandb(project_name: str, model_name: str, language: str, num_epochs: int, batch_size: int):
    wandb.init(
        entity="cs4nlp",
        project=project_name,
        name=model_name,
        config = {
            "language": language,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
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


def finetune(train_datapoints, val_datapoints, model_name: str, project_name: str, wandb_flag: bool, num_epochs: int, batch_size: int):
    print(f"\nTraining LoRA adapter for {project_name} model: {model_name}")
    print(f"Using device: {DEVICE}")
    print(f"Using WandB: {wandb_flag}")

    # Clear CUDA cache before model initialization
    torch.manual_seed(0)    
    torch.cuda.empty_cache()
    print(batch_size)
    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = get_peft_model(model, LORA_CONFIG).to(DEVICE)
    model.train()

    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print(tokenizer)
    train_dataset = PromptDataset(
        datapoints=train_datapoints,
        tokenizer=tokenizer)
    
    val_dataset = PromptDataset(
        datapoints=val_datapoints,
        tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=True
    )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
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
        num_warmup_steps=WARMUP_STEPS * len(train_loader) * num_epochs,
        num_training_steps=num_epochs * len(train_loader)
    )
    # Training loop
    training_start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()
        
        best_val_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        # Training phase
        model.train()
        train_loss = 0.0
        
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

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
                loss = outputs.loss
                val_loss += loss.item() * batch["input_ids"].size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if wandb_flag:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1} completed in: {(time.time() - epoch_start_time) // 60}m {(time.time() - epoch_start_time) % 60:.0f}s")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_pretrained(os.path.join(OUTPUT_DIR, model_name))
            print(f"New best LoRA adapter saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")


    if wandb_flag:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()
            
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    print(f"Training completed in: {(time.time() - training_start_time) // 60}m {(time.time() - training_start_time) % 60:.0f}s")
