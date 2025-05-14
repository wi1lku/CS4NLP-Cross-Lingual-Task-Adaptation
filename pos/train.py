from finetune import finetune, init_wandb, parser, MODEL_PATH
from sentences import get_prompt, get_label

from conllu import parse_incr

def get_datapoints(data_path: str, size: str, cutoff: int):
    """Load and preprocess the dataset."""
    with open(data_path, "r") as f:
        conllu_sentences = parse_incr(f)

        datapoints = [(get_prompt(sentence), get_label(sentence)) for sentence in conllu_sentences]

    if size == "half":
        datapoints = datapoints[: len(datapoints) // 2]
    if size == "quarter":
        datapoints = datapoints[: len(datapoints) // 4]
    elif size != "full":
        raise ValueError("size must be either 'full', 'half' or 'quarter'")
    if cutoff > 0:
        datapoints = datapoints[:cutoff]

    return datapoints


def main():
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--ds_cutoff", type=int, default=0, help="Cutoff for dataset size")
    args = parser.parse_args()
    wandb_flag = args.wandb
    language = args.language
    model_name = args.model_name
    project_name = args.project_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_data = args.train_data
    val_data = args.val_data
    ds_size = args.ds_size
    if ds_size not in ["full", "half", "quarter"]:
        raise ValueError(f"ds_size must be either 'full', 'quarter' or 'half', not {ds_size}")
    # Initialize wandb if specified
    if wandb_flag:
        init_wandb(project_name, model_name, language, num_epochs, batch_size)

    train_datapoints = get_datapoints(train_data, ds_size, args.ds_cutoff)
    val_datapoints = get_datapoints(val_data, ds_size, args.ds_cutoff//5)
    

    # Train the model
    finetune(
        train_datapoints=train_datapoints,
        val_datapoints=val_datapoints,
        model_name=model_name,
        project_name=project_name,
        wandb_flag=wandb_flag,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

if __name__=="__main__":
    main()