#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/pos_eval_out
#SBATCH --time=24:00:00

module add cuda/12.6
source .venv/bin/activate

langs1=("ar" "de" "en" "es" "fr" "hi" "ru" "th" "tr" "zh")
langs2=("ar" "de" "en" "es" "fr" "hi" "ru" "th" "tr" "zh")
data_fracs=(0.5 1.0)



for lang1 in "${langs1[@]}"; do
  for lang2 in "${langs2[@]}"; do
    for data_frac in "${data_fracs[@]}"; do
      python eval.py --project_name "POS" --adapters_dir pos/output/adapters/ --results_path pos/results.json --test_data_path pos/data/split/$lang1\_pud-ud-test-split.conllu --train_lang $lang1 --test_lang $lang2 --data_frac $data_frac --batch_size 16 --wandb
    done
  done
done