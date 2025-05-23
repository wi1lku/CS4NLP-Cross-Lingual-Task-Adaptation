#!/bin/bash
#SBATCH --account=pmlr_jobs
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_nli.out

langs=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")
# data_fracs=(0.0 0.25 0.5 0.75 1.0)
data_fracs=(0.5 1.0)

for lang1 in "${langs[@]}"; do
  for lang2 in "${langs[@]}"; do
    for data_frac in "${data_fracs[@]}"; do
      python eval_nli.py --project_name "NLI" --train_lang $lang1 --test_lang $lang2 --data_frac $data_frac --batch_size 16
    done
  done
done