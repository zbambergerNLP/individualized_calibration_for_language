import logging

import numpy as np
from accelerate import Accelerator
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

"""
To run this script using Newton:

* First, run the following command: `accelerate config`.
  Follow the instructions to set up your acceleration configuration for future runs.
  - You are running on "This machine"
  - You are using "Multi-GPU"
  - You are using "1" machine (single node)
  - You do not wish to optimize your script with torch dynamo
  - You do wish to use deepspeed
  - You do wish to specify a deepspeed configuration file
  - You are using the "zero_stage2.json" deepspeed config file
  - You do not wand to enable `deepspeed.zero.Init` since you are not using ZeRO stage-3
  - You are using "4" GPUs

* Second run the following command:
  WANDB_SERVICE_WAIT=300 srun --partition=nlp --account=nlp --gres=gpu:4  -c 20 accelerate launch main.py

The above command will request 4 GPU and 20 CPU cores. You can change these values as needed.
Note that you will need to adjust the partition and account to match your Newton account.
"""

accelerator = Accelerator(log_with="wandb")
logger = logging.Logger(__name__)


def main():
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

    if accelerator.is_local_main_process:
        # fetch the run_id from your wandb workspace
        last_run_id = training_args.last_run_id if training_args.local_checkpoint_path else None
        wandb_config = {
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_name_or_path": data_args.dataset_name_or_path,
            "max_seq_length": data_args.max_seq_length,
            "mlp_hidden": model_args.mlp_hidden,
            "mlp_dropout": model_args.mlp_dropout,
            "tokenizer_batch_size": model_args.tokenizer_batch_size,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "seed": training_args.seed,
            "warmup_ratio": training_args.warmup_ratio,
            "weight_decay": training_args.weight_decay,
            "sample_train_examples": data_args.sample_train_examples,
            "sample_validation_examples": data_args.sample_validation_examples,
            "sample_test_examples": data_args.sample_test_examples,
        }
        logger.info('Initializing wandb...')
        accelerator.init_trackers(
            project_name="individual_calibration_for_language",
            config=wandb_config,
            init_kwargs={
                "wandb": {
                    "entity": "zbamberger",
                }
            },
        )
        # wandb.init(
        #     project="individual_calibration_for_language",
        #     id=last_run_id,
        #     config=wandb_config,
        #     resume="must" if last_run_id is not None else None,
        # )

    # Load pretrained model
    logger.info('Creating CommentRegressor model...')
    experiment_model = comment_regressor.CommentRegressor(
        mlp_hidden=model_args.mlp_hidden,
        drop_prob=model_args.mlp_dropout,
        text_encoder_model_name=model_args.model_name_or_path,
        dtype=torch.float16  # Use float16 for faster training. TODO: Make this a flag.
    )

    # Define training arguments
    training_arguments = TrainingArguments(
        run_name=training_args.run_name,
        report_to=["wandb"],
        load_best_model_at_end=True,
        output_dir=training_args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=training_args.eval_steps,
        save_strategy="steps",
        save_steps=training_args.save_steps,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=training_args.learning_rate,
        seed=training_args.seed,
        data_seed=training_args.seed,
        eval_accumulation_steps=training_args.eval_accumulation_steps,
        optim=training_args.optimizer,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        label_names=["labels"],
        logging_dir="./logs",
        logging_steps=training_args.logging_steps,
        include_inputs_for_metrics=True,
        # Parameters to increase the efficiency of the training.
        fp16=True,
        dataloader_num_workers=training_args.dataloader_num_workers if torch.cuda.is_available() else 0,
        deepspeed=training_args.deepspeed,
    )

    # Define data collator
    logger.info('Creating data collator...')
    data_collator = data.CommentRegressorDataCollator(
        tokenizer=tokenizer_function,
        max_length=data_args.max_seq_length,
    )

    # Define trainer
    logger.info('Creating trainer...')
    trainer = RandomizedIndividualCalibrationTrainer(
        model=experiment_model,
        args=training_arguments,
        compute_metrics=metrics.compute_metrics,
        data_collator=data_collator,
        train_dataset=train_dataset['train'],
        eval_dataset=validation_dataset['train'],
        coefficient=training_args.coefficient,
        callbacks=[
            transformers.EarlyStoppingCallback(early_stopping_patience=3),  # TODO: Make this a flag.
        ],
    )

    # # See initial evaluation results
    # eval_result = trainer.evaluate(eval_dataset=test_dataset['train'])
    # print(f'Initial evaluation results:\n{eval_result}')

    # Training
    logger.info('Starting training...')
    trainer.train(
        resume_from_checkpoint=(
            training_args.local_checkpoint_path if (
                    training_args.local_checkpoint_path is not None and
                    os.path.isdir(training_args.local_checkpoint_path)
            ) else None
        )
    )
    logger.info('Finished training!')

    # Evaluation
    logger.info('Starting evaluation on the validation set...')
    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset['train'])
    logger.info(f'Validation results:\n{validation_metrics}')
    logger.info('Starting evaluation on the test set...')
    test_metrics = trainer.evaluate(eval_dataset=test_dataset['train'])
    logger.info(f'Test results:\n{test_metrics}')
    logger.info('Finished evaluation!')

    # Plot results
    # logger.info('Plotting results...')
    # plots.end_of_training_plots(
    #     eval_result=test_metrics,
    #     alpha=training_args.coefficient,
    #     wandb_run=accelerator.get_tracker("wandb"),
    # )
    # logger.info('Finished plotting!')

    # Save the model
    logger.info('Saving model...')
    experiment_model.save_pretrained(training_args.output_dir)
    logger.info(f'Finished saving model to {training_args.output_dir}!')


if __name__ == "__main__":
    main()
