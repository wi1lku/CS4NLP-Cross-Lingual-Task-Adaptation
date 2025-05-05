#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/eval.out

module add cuda/12.6

python eval.py "$@"