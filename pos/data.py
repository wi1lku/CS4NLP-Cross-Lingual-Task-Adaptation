from pos.sentences import get_prompt, get_label

from conllu import parse_incr


def _get_datapoints(data_path: str, size: float):
    """Load and preprocess the dataset."""
    with open(data_path, "r") as f:
        conllu_sentences = parse_incr(f)

        datapoints = [(get_prompt(sentence), get_label(sentence)) for sentence in conllu_sentences]

        datapoints = datapoints[:round(size * len(datapoints))]

    return datapoints

def get_datapoints(data_paths,
                 size: float,
                 language: str):
    return [_get_datapoints(data_path, size) for data_path in data_paths]


def split_dataset(data_path, 
                 train_size: float = 0.7,
                 dev_size: float = 0.15,
                 test_size: float = 0.15):
    """
    Split the dataset into train, dev, and test sets.
    """
    assert train_size + dev_size + test_size == 1.0, "Sizes must sum to 1."

    with open(data_path, "r") as f:
        conllu_sentences = parse_incr(f)
        sentences = list(conllu_sentences)
        n = len(sentences)
        train_end = int(train_size * n)
        dev_end = train_end + int(dev_size * n)
    train_sentences = sentences[:train_end]
    dev_sentences = sentences[train_end:dev_end]
    test_sentences = sentences[dev_end:]
    out_paths = {
        "train": data_path.replace("test.conllu", "train-split.conllu"),
        "dev": data_path.replace("test.conllu", "dev-split.conllu"),
        "test": data_path.replace("test.conllu", "test-split.conllu"),
    }
    with open(out_paths["train"], "w") as f:
        f.write("\n".join([sentence.serialize() for sentence in train_sentences]))
    with open(out_paths["dev"], "w") as f:
        f.write("\n".join([sentence.serialize() for sentence in dev_sentences]))
    with open(out_paths["test"], "w") as f:
        f.write("\n".join([sentence.serialize() for sentence in test_sentences]))
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split a CoNLL-U dataset into train, dev, and test sets.")
    parser.add_argument("--data_path", type=str, help="Path to the CoNLL-U dataset file.")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of data to use for training.")
    parser.add_argument("--dev_size", type=float, default=0.15, help="Proportion of data to use for development.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of data to use for testing.")
    
    args = parser.parse_args()
    
    split_dataset(args.data_path, args.train_size, args.dev_size, args.test_size)

if __name__ == "__main__":
    main()