# CS4NLP: Cross Lingual Task Adaptation
## Fine-tuning

To fine-tune the model follow the steps below.
1. Make sure that Llama-3.2-1B is downloaded and placed in `/model/llama-3.2-1b` folder.
2. Choose desired parameters and run:
```
sbatch --export=ALL,LANGUAGE=de,DS_SIZE=full,WANDB=--wandb,NUM_EPOCHS=10,BATCH_SIZE=4,MODEL_NAME=de_full fine-tune.sh
```