#!/bin/bash
#SBATCH --account=csnlp_jobs
#SBATCH --output=logs/fine-tune.out

module add cuda/12.6

source ~/cil-env/bin/activate
python fine-tune.py --language=$LANGUAGE --ds_size=$DS_SIZE $WANDB --num_epochs=$NUM_EPOCHS --batch_size=$BATCH_SIZE --model_name=$MODEL_NAME