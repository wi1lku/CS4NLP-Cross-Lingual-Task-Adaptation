# CS4NLP: Cross Lingual Task Adaptation

## Data
To create the data split for Parallel UD, run `pos/data.py` with the path to a PUD .conllu file. See `split.sh` for reference.

## Fine-tuning

To fine-tune the model follow the steps below.
1. Make sure that Llama-3.2-1B is downloaded and placed in `/model/llama-3.2-1b` folder.
2. Adjust the parameters in `pos.sbatch` or `nli.sbatch` as desired and run. 

## Evaluation
1. Before the first run, create a file containing an initially empty python dictionary (`{}`) to fill with the results (`echo "{}" > results.json`). 
2. Adjust parameters in `pos_eval.sbatch` or `nli_eval.sbatch` as desired and run.

## Visualization

`notebooks/analysis.py` contains many useful features for visualizing the data produced by the evaluation scripts. 
Run `notebooks/nli-results-visualisation.ipynb` or `notebooks/pos-results-visualisation.ipynb` to display the most relevant plots.