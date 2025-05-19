#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/eval.out

module add cuda/12.6

source .venv/bin/activate
python eval.py --adapter_path "./output/adapters/fr_0.5/" --language "fr"