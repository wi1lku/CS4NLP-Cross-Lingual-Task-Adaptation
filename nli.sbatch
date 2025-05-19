#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/nli_out

module add cuda/12.6.0

source .venv/bin/activate
python fine-tune.py --language="fr" --ds_size=0.5  --num_epochs=10 --batch_size=8 --model_name="fr_0.5"c