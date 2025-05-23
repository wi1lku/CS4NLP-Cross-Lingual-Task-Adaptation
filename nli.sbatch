#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/nli_out

module add cuda/12.6.0
source ~/cil-env/bin/activate

langs=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")
data_fracs=(0.25 0.75)

for lang in "${langs[@]}"; do
    for data_frac in "${data_fracs[@]}"; do
      python fine-tune.py --language=$lang --ds_size=$data_frac  --num_epochs=10 --batch_size=4 --model_name="${lang}_${data_frac}" --wandb
    done
done