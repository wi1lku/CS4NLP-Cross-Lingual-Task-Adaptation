import json
import pandas as pd

data_path = "./nli/data/xnli.test.jsonl"
df = pd.read_json(data_path, lines=True)

langs = df.language.unique()
metrics = ["accuracy", "precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro"]
data_fractions = [0.00, 0.25, 0.50, 0.75, 1.00]

template_dict = {
    lang_eval: {
        lang_train: {
            data_fraction: {
                metric: None
                for metric in metrics
            } for data_fraction in data_fractions
        } for lang_train in langs
    } for lang_eval in langs
}

with open("./nli/results.json", "w") as f:
    json.dump(template_dict, f, indent=4)