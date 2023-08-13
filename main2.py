import logging

import accelerate
import numpy as np
import data_preprocessing
import trainer2
import flags
import os
import typing
import model as comment_regressor
import datasets
import torch
import transformers
import data
from plots import end_of_training_plots
from average_calibration import perform_average_calibration


def main():

    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = transformers.HfArgumentParser(
        (flags.ModelArguments, flags.DataArguments, flags.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=training_args.training_accumulation_steps,
        log_with='wandb',
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)

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
    tokenized_calibration_dataset_name = 'tokenized_calib_data.csv'  # TODO: Make this a flag
    tokenized_test_dataset_name = 'tokenized_test_data.csv'   # TODO: Make this a flag

    # Check if the tokenized datasets already exist. If so, load them. Otherwise, create them.
    datasets_exist = (
        os.path.exists(os.path.join('data', tokenized_train_dataset_name)) and
        os.path.exists(os.path.join('data', tokenized_validation_dataset_name)) and
        os.path.exists(os.path.join('data', tokenized_calibration_dataset_name)) and
        os.path.exists(os.path.join('data', tokenized_test_dataset_name))
    )

    if datasets_exist:
        logger.log(logging.INFO, 'Loading tokenized datasets from disk.')
        train_dataset = datasets.load_from_disk(os.path.join('data', tokenized_train_dataset_name))
        validation_dataset = datasets.load_from_disk(os.path.join('data', tokenized_validation_dataset_name))
        calibration_dataset = datasets.load_from_disk(os.path.join('data', tokenized_calibration_dataset_name))
        test_dataset = datasets.load_from_disk(os.path.join('data', tokenized_test_dataset_name))

    # Load datasets
    else:
        logger.log(logging.INFO, 'Tokenizing datasets...')
        train_dataset, validation_dataset, calibration_dataset, test_dataset = data.create_datasets(
            train_file=data_args.train_file,
            validation_file=data_args.validation_file,
            calib_file=data_args.calib_file,
            test_file=data_args.test_file,
            tokenizer_function=tokenizer_function,
            batch_size=model_args.tokenizer_batch_size,
        )
        logger.log(logging.INFO, 'Saving tokenized datasets to disk')
        train_dataset.save_to_disk(
            os.path.join('data', tokenized_train_dataset_name),
        )
        validation_dataset.save_to_disk(
            os.path.join('data', tokenized_validation_dataset_name),
        )
        calibration_dataset.save_to_disk(
            os.path.join('data', tokenized_calibration_dataset_name),
        )
        test_dataset.save_to_disk(
            os.path.join('data', tokenized_test_dataset_name),
        )

    # Sample examples according to user specifications from flags and create a new dataset
    # with the sampled examples.
    if data_args.sample_train_examples:
        logger.log(logging.INFO, f'Sampling {data_args.sample_train_examples} examples from the training set.')
        train_dataset['train'] = train_dataset['train'].select(range(data_args.sample_train_examples))
    if data_args.sample_validation_examples:
        logger.log(logging.INFO, f'Sampling {data_args.sample_validation_examples} examples from the validation set.')
        validation_dataset['train'] = validation_dataset['train'].select(range(data_args.sample_validation_examples))
    if data_args.sample_test_examples:
        logger.log(logging.INFO, f'Sampling {data_args.sample_test_examples} examples from the test set.')
        test_dataset['train'] = test_dataset['train'].select(range(data_args.sample_test_examples))

    # Create the model
    logger.log(
        logging.INFO,
        f'Creating model with the following parameters:'
        f'\n\tMLP hidden layers: {model_args.mlp_hidden}'
        f'\n\tMLP dropout: {model_args.mlp_dropout}'
        f'\n\tText encoder model name: {model_args.model_name_or_path}'
        f'\n\tInput r dim: {model_args.input_r_dim}'
    )
    experiment_model = comment_regressor.CommentRegressor(
        mlp_hidden=model_args.mlp_hidden,
        drop_prob=model_args.mlp_dropout,
        text_encoder_model_name=model_args.model_name_or_path,
        input_r_dim=model_args.input_r_dim,
        # dtype=torch.float16  # Use float16 for faster training. TODO: Make this a flag.
    )

    optimizer = torch.optim.Adam(
        experiment_model.parameters(),
        lr=training_args.learning_rate)
    total_steps = int(
            (len(train_dataset['train']) / training_args.per_device_train_batch_size) * training_args.num_train_epochs
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * total_steps),
        num_training_steps=total_steps)

    experiment_name = (
        f'seed_{training_args.seed}_'
        f'coefficient_{str(training_args.coefficient).replace(".", "_")}_'
        f'lr_{str(training_args.learning_rate).replace(".", "_")}_'
        f'r_dim_{model_args.input_r_dim}_'
        f'r_upper_bound_{data_args.r_input_upper_bound}_'
    )
    output_dir = os.path.join(training_args.output_dir, experiment_name)

    # Create a trainer for the experiment model
    trainer = trainer2.CommentRegressorTrainer(
        output_dir=output_dir,
        model=experiment_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset['train'],
        validation_dataset=validation_dataset['train'],
        calibration_dataset=calibration_dataset['train'],
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
        r_input_upper_bound=data_args.r_input_upper_bound,
        eval_steps=training_args.eval_steps,
        patience=training_args.patience,
        accelerator=accelerator,
    )

    logger.log(logging.INFO, f'Starting training for experiment {experiment_name}')
    trainer.train(
        epochs=training_args.num_train_epochs,
    )

    # Extract best model from checkpoint/best_model
    trainer.model = trainer.model.from_pretrained(trainer.best_checkpoint_path)

    # Evaluate the model on the test set
    test_metrics = trainer.eval(validation_loader=trainer.test_loader,
                                validation_accumulation_steps=trainer.validation_accumulation_steps,
                                with_wandb=True,
                                )

    if accelerator.is_local_main_process:
        end_of_training_plots(
            eval_result=test_metrics,
            alpha=training_args.coefficient,
            data_title="r in [0, {}], r dim = {}".
            format(data_args.r_input_upper_bound, model_args.input_r_dim),
            wandb_run=accelerator.get_tracker("wandb"),
        )

        # Perform average calibration
        if training_args.do_average_calibration:
            # TODO: Ensure average calibration works in distributed training.
            calibrated_comment_regressor = perform_average_calibration(
                comment_regressor=trainer.model,
                calibration_dataloader=trainer.calibration_loader
            )

            # Evaluate the calibrated model on the test set
            trainer.model = calibrated_comment_regressor.to(trainer.accelerator.device)
            test_metrics_calib = trainer.eval(validation_loader=trainer.validation_loader,
                                              validation_accumulation_steps=trainer.validation_accumulation_steps)

            end_of_training_plots(eval_result=test_metrics_calib,
                                  alpha=training_args.coefficient,
                                  data_title="r in [0, {}], r dim = {}, calibrated".
                                  format(data_args.r_input_upper_bound, model_args.input_r_dim),
                                  wandb_run=None)

    accelerator.print("\n\nDone")


if __name__ == '__main__':
    main()
