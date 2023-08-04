import logging

import numpy as np
from accelerate import Accelerator

import data_preprocessing
import trainer2
import wandb
import plots

import flags
import os
import typing

import metrics
import model as comment_regressor
import datasets
import torch
import transformers

import data
from trainer import RandomizedIndividualCalibrationTrainer
from transformers import TrainingArguments, AutoTokenizer
from plots import end_of_training_plots

deepspeed_plugin = (

)

# accelerator = Accelerator(
#     log_with="wandb",
#     # Parameters for automatic precision and/or mixed precision training
# )


def main():

    print("cuda is available: ", torch.cuda.is_available())

    # Parse arguments
    parser = transformers.HfArgumentParser(
        (flags.ModelArguments, flags.DataArguments, flags.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Establish determinism for reproducibility. Use provided seed
    # if one was provided. Otherwise, generate a random seed.
    if training_args.seed is None:
        training_args.seed = int(torch.randint(2 ** 32, (1,)))
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    np.random.seed(training_args.seed)
    transformers.set_seed(training_args.seed)

    def tokenizer_function(
            example: typing.Dict[str, typing.Any],
    ) -> typing.Dict[str, typing.Any]:
        """
        Tokenize the input text and return the result (including the original label).
        :param example: A dictionary containing the input text and the label.
        :return: A dictionary containing the tokenized input text and the label.
        """
        result = tokenizer(
            example[data_args.dataset_features_column],
            truncation=True,                # Truncate the input sequences to the maximum length the model accepts.
            padding='max_length',           # Pad the input sequences to the maximum length the model accepts.
            max_length=512,                 # The maximum sequence length of the input text for BERT is 512 tokens.
            return_tensors='pt',            # Return the result as PyTorch tensors.
        )
        # Add the labels to the result. This is required by the Trainer.
        result['labels'] = example[data_args.dataset_labels_column]
        for key, value in example.items():
            if key not in result:
                result[key] = value
        return result

    tokenized_train_dataset_name = 'tokenized_train_data.csv'  # TODO: Make this a flag
    tokenized_validation_dataset_name = 'tokenized_eval_data.csv'  # TODO: Make this a flag
    tokenized_test_dataset_name = 'tokenized_test_data.csv'   # TODO: Make this a flag

    # Check if the tokenized datasets already exist. If so, load them. Otherwise, create them.
    datasets_exist = (
        os.path.exists(os.path.join('data', tokenized_train_dataset_name)) and
        os.path.exists(os.path.join('data', tokenized_validation_dataset_name)) and
        os.path.exists(os.path.join('data', tokenized_test_dataset_name))
    )

    if datasets_exist:
        train_dataset = datasets.load_from_disk(os.path.join('data', tokenized_train_dataset_name))
        validation_dataset = datasets.load_from_disk(os.path.join('data', tokenized_validation_dataset_name))
        test_dataset = datasets.load_from_disk(os.path.join('data', tokenized_test_dataset_name))

    # Load datasets
    else:
        train_dataset, validation_dataset, test_dataset = data.create_datasets(
            train_file=data_args.train_file,
            validation_file=data_args.validation_file,
            test_file=data_args.test_file,
            tokenizer_function=tokenizer_function,
            batch_size=model_args.tokenizer_batch_size,
        )
        train_dataset.save_to_disk(
            os.path.join('data', tokenized_train_dataset_name),
        )
        validation_dataset.save_to_disk(
            os.path.join('data', tokenized_validation_dataset_name),
        )
        test_dataset.save_to_disk(
            os.path.join('data', tokenized_test_dataset_name),
        )

    # Sample examples according to user specifications from flags and create a new dataset
    # with the sampled examples.
    if data_args.sample_train_examples:
        train_dataset['train'] = train_dataset['train'].select(range(data_args.sample_train_examples))
    if data_args.sample_validation_examples:
        validation_dataset['train'] = validation_dataset['train'].select(range(data_args.sample_validation_examples))
    if data_args.sample_test_examples:
        test_dataset['train'] = test_dataset['train'].select(range(data_args.sample_test_examples))

    experiment_model = comment_regressor.CommentRegressor(
        mlp_hidden=model_args.mlp_hidden,
        drop_prob=model_args.mlp_dropout,
        text_encoder_model_name=model_args.model_name_or_path,
        # dtype=torch.float16  # Use float16 for faster training. TODO: Make this a flag.
    )

    optimizer = torch.optim.Adam(
        experiment_model.parameters(),
        lr=training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
    )

    # Create a trainer for the experiment model
    trainer = trainer2.CommentRegressorTrainer(
        model=experiment_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset['train'],
        validation_dataset=validation_dataset['train'],
        test_dataset=test_dataset['train'],
        coefficient=training_args.coefficient,
        optimizer=optimizer,
        scheduler=scheduler,
        group_names=data_preprocessing.GROUP_LIST,
        train_batch_size=training_args.per_device_train_batch_size,
        validation_batch_size=training_args.per_device_eval_batch_size,
        test_batch_size=training_args.per_device_eval_batch_size,
        training_accumulation_steps=training_args.training_accumulation_steps,
        validation_accumulation_steps=training_args.eval_accumulation_steps,
    )
    trainer.train(
        epochs=training_args.num_train_epochs,
    )

    # Evaluate the model on the test set
    test_metrics = trainer.eval_iter(model=trainer.model,
                                     validation_loader=trainer.test_loader)
    end_of_training_plots(eval_result=test_metrics,
                          alpha=training_args.coefficient)

if __name__ == '__main__':
    main()

