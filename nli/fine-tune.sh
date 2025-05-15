#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/fine-tune.out

module add cuda/12.6

source ~/cil-env/bin/activate
python fine-tune.py --language="fr" --ds_size=0.5 --wandb --num_epochs=10 --batch_size=8 --model_name="fr_0.5"