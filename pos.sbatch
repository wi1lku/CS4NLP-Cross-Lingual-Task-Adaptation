#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/pos_out
#SBATCH --time=24:00:00

module add cuda/12.6.0
source .venv/bin/activate

langs=("ar" "cs" "de" "gl" "en" "es" "fr" "fi" "hi" "is" "it" "ru" "ja" "ko" "pl" "pt" "tr" "id" "zh" "th")
data_fracs=(0.25 0.5 0.75 1.0)
epochs=10

for lang in "${langs[@]}"; do
    for data_frac in "${data_fracs[@]}"; do
      python fine-tune.py --project_name="POS" --language=$lang --data_paths pos/data/split/$lang\_pud-ud-train-split.conllu pos/data/split/$lang\_pud-ud-dev-split.conllu --ds_size=$data_frac  --num_epochs=$epochs --batch_size=2 --model_name="${lang}_${data_frac}" --output_dir="./pos/output/adapters" --wandb
    done
done