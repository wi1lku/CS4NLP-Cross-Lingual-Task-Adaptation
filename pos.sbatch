#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/pos_out

module add cuda/12.6.0

source .venv/bin/activate
python pos/train.py --project_name="POS" --train_data="./pos/data/UD_German-GSD/de_gsd-ud-train.conllu" --val_data="./pos/data/UD_German-GSD/de_gsd-ud-dev.conllu" --language="de" --ds_size="quarter" --ds_cutoff=500