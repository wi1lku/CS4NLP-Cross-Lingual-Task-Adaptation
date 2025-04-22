#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/fine-tune.out

module add cuda/12.6

python eval.py --adapter_path=$PATH --language=$LANGUAGE --batch_size=$BATCH_SIZE --wandb=$WANDB