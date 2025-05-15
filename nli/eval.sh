#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/eval.out

module add cuda/12.6

python eval.py --adapter_path "./output/adapters/fr_0.5/" --language "fr" --wandb