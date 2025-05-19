import pandas as pd


def create_xnli_prompt(premise: str, hypothesis: str) -> str:
    """
    Create a prompt for XNLI dataset.
    """
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: "
    return prompt

def get_datapoints(data_paths,
                 size: float,
                 language: str):
    datapointlist = []
    for data_path in data_paths:
        language = language
        print("Data_path:", data_path)

        # Load XNLI data from jsonl file
        raw_data = pd.read_json(path_or_buf=data_path, lines=True)

        # Filter data by language and size
        raw_data = raw_data[raw_data["language"] == language]
        raw_data = raw_data.iloc[:round(size * len(raw_data))]

        datapointlist.append([
            (create_xnli_prompt(row["sentence1"], row["sentence2"]), row["gold_label"])
            for _, row in raw_data.iterrows()
        ])

    return datapointlist