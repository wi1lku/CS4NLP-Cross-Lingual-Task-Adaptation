from pos.sentences import get_prompt, get_label

from conllu import parse_incr


def _get_datapoints(data_path: str, size: float):
    """Load and preprocess the dataset."""
    with open(data_path, "r") as f:
        conllu_sentences = parse_incr(f)

        datapoints = [(get_prompt(sentence), get_label(sentence)) for sentence in conllu_sentences]

        datapoints = datapoints[:round(size * len(datapoints))]

    return datapoints

def get_datapoints(train_data_path: str,
                 test_data_path: str,
                 size: float,
                 language: str):
    train_datapoints = _get_datapoints(train_data_path, size)
    test_datapoints = _get_datapoints(test_data_path, size)
    return train_datapoints, test_datapoints