from dataclasses import dataclass, field
from typing import (
    Optional,
)
import os


@dataclass
class TrainingArguments:
    run_name: Optional[str] = field(
        default="randomized_individual_calibration",
        metadata={"help": "The name of the run. Used for logging and saving checkpoints."},
    )
    local_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a local checkpoint. Used to resume training."},
    )
    output_dir: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    last_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "The ID of the last run. Used to resume training."},
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per device during training."}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per device for evaluation."}
    )
    optimizer: Optional[str] = field(
        default="adamw_hf", metadata={"help": "The optimizer to use. Can be 'adamw' or 'adafactor'."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "The LR scheduler to use. Can be 'linear' or 'cosine'."}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    coefficient: float = field(
        default=0.8, metadata={"help": "The coefficient for the calibration loss."}
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: int = field(default=0.1, metadata={"help": "The ratio of warmup steps to total training steps."})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    eval_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of eval steps to accumulate before performing backward pass."}
    )
    save_steps: int = field(default=10_000, metadata={"help": "Save checkpoint every X updates steps."})
    # TODO: Switch eval_steps back to 1_000 after debugging.
    eval_steps: int = field(default=200, metadata={"help": "Run evaluation every X updates steps."})
    dataloader_num_workers: int = field(
        default=20, metadata={"help": "Number of subprocesses to use for data loading."}
    )
    deepspeed: str = field(default="zero_stage2.json", metadata={"help": "The path to the deepspeed config file."})
    seed: int = field(default=42, metadata={"help": "The seed to use for training."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        # TODO: Consider switching default to `textattack/bert-base-uncased-yelp-polarity`
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )
    mlp_hidden: Optional[int] = field(
        default=100,
        metadata={"help": "The hidden size of the MLP layer."},
    )
    mlp_dropout: Optional[float] = field(
        default=0.5,
        metadata={"help": "The dropout probability of the MLP layer."},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    tokenizer_batch_size: int = field(
        default=16, metadata={"help": "The batch size used when tokenizing the dataset."}
    )


@dataclass
class DataArguments:
    # For HuggingFace datasets
    dataset_name_or_path: Optional[str] = field(
        default='civil_comments',
        metadata={"help": "Path to dataset or dataset identifier from huggingface.co/datasets."},
    )
    dataset_features_column: str = field(
        default="comment_text", metadata={"help": "The name of the column containing the input text."}
    )
    dataset_labels_column: str = field(
        default="target", metadata={"help": "The name of the column containing the labels."}
    )

    # For custom datasets
    train_file: Optional[str] = field(
        default='data/train_data.csv',
        metadata={"help": "Path to training data file"},
    )
    validation_file: Optional[str] = field(
        default='data/eval_data.csv',
        metadata={"help": "Path to validation data file"},
    )
    test_file: Optional[str] = field(
        default='data/test_data.csv',
        metadata={"help": "Path to testing data file"},
    )
    sample_train_examples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of training examples to use. If None, use all examples."},
    )
    sample_validation_examples: Optional[int] = field(
        default=None,  # TODO: Set this to None once we're done debugging.
        metadata={"help": "The number of validation examples to use. If None, use all examples."},
    )
    sample_test_examples: Optional[int] = field(
        default=None,
        metadata={"help": "The number of testing examples to use. If None, use all examples."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
