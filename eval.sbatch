#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/eval_out.out
#SBATCH --time=24:00:00

module add cuda/12.6
source <insert env path>

langs1=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")
langs2=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")
data_fracs=(0.25 0.5 0.75 1.0)

# No training
for lang1 in "${langs1[@]}"; do
    python eval_nli.py --project_name "NLI" --train_lang "base" --test_lang $lang2 --data_frac 0.0 --batch_size 16
done

for lang1 in "${langs1[@]}"; do
  for lang2 in "${langs2[@]}"; do
    for data_frac in "${data_fracs[@]}"; do
      python eval_nli.py --project_name "NLI" --train_lang $lang1 --test_lang $lang2 --data_frac $data_frac --batch_size 16
    done
  done
done