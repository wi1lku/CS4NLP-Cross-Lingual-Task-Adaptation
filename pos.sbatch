#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/pos_out

module add cuda/12.6.0

source .venv/bin/activate
python fine-tune.py --project_name="POS" --data_paths "./pos/data/UD_German-GSD/de_gsd-ud-train.conllu" "./pos/data/UD_German-GSD/de_gsd-ud-dev.conllu" --language="de" --ds_size=0.05 --batch_size=2