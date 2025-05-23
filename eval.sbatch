#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/eval.out

module add cuda/12.6

source ~/cil-env/bin/activate
python eval.py --project_name "NLI" --train_lang "fr" --test_lang "tr" --data_frac 0.5 --batch_size 16
python eval.py --project_name "NLI" --train_lang "en" --test_lang "tr" --data_frac 0.5 --batch_size 16
python eval.py --project_name "NLI" --train_lang "en" --test_lang "tr" --data_frac 1.0 --batch_size 16
python eval.py --project_name "NLI" --train_lang "de" --test_lang "tr" --data_frac 0.5 --batch_size 16
python eval.py --project_name "NLI" --train_lang "de" --test_lang "tr" --data_frac 1.0 --batch_size 16