import os
import typing
import datasets
import torch
import transformers

import data_preprocessing


class CommentRegressorDataCollator(transformers.DefaultDataCollator):
    """Data collator for comment regression tasks."""

    def __init__(self, tokenizer, max_length=None):
        """Initialize the data collator.

        Args:
            tokenizer: The tokenizer to use.
            max_length: The maximum length of the input text.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
            self,
            features: typing.List[typing.Dict[str, typing.Any]],
            return_tensors=None
    ) -> typing.Dict[str, typing.Any]:
        print(f'\n\nfeatures is: {features}')
        if return_tensors is None:
            return_tensors = self.return_tensors
        result = transformers.default_data_collator(features, return_tensors)
        input_r = torch.rand(
            result.get('input_ids').shape[0],
            1,
            device=result.get('input_ids').device)
        result['input_r'] = input_r
        for group in data_preprocessing.GROUP_LIST:
            group_values = []
            for example in features:
                if group not in example:
                    group_values.append(-1)  # TODO: Note that -1 implies that the feature is not present.
                else:
                    group_values.append(example[group])
            result[group] = torch.tensor(
                group_values,
                dtype=torch.float,
                device=result.get('input_ids').device,
            )
        print(f'result: {result}')
        return result


def create_datasets(
        train_file: str,
        validation_file: str,
        test_file: str,
        tokenizer_function: typing.Callable[
            [typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]] = None,
        batch_size: int = 32,
) -> typing.Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Create the training and test data loaders.

    Args:
        train_file: The path to the training file.
        validation_file: The path to the validation file.
        test_file: The path to the test file.
        tokenizer_function: The function to use to tokenize the input text.
        batch_size: The batch size.

    Returns:
        A 3-tuple containing the training, validation, and test datasets respectively.
    """

    train_dataset = datasets.load_dataset(
        "csv",
        data_files=train_file,
    )
    validation_dataset = datasets.load_dataset(
        "csv",
        data_files=validation_file,
    )
    test_dataset = datasets.load_dataset(
        "csv",
        data_files=test_file,
    )
    tokenized_train_dataset = train_dataset.map(
        function=tokenizer_function,
        batched=True,
        batch_size=batch_size,
        num_proc=os.cpu_count(),  # Use all CPU cores.
    )
    tokenized_validation_dataset = validation_dataset.map(
        function=tokenizer_function,
        batched=True,
        batch_size=batch_size,
        num_proc=os.cpu_count(),  # Use all CPU cores.
    )
    tokenized_test_dataset = test_dataset.map(
        function=tokenizer_function,
        batched=True,
        batch_size=batch_size,
        num_proc=os.cpu_count(),  # Use all CPU cores.
    )
    tokenized_train_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels'],
    )
    tokenized_validation_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels'],
    )
    tokenized_test_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels'],
    )

    return tokenized_train_dataset, tokenized_validation_dataset, tokenized_test_dataset


def tokenize_dataset(
        dataset: datasets.Dataset,
        tokenizer_function: typing.Callable,
        tokenizer_name: str,
        dataset_name: str,
        tokenizer_batch_size: int = 32,
):
    tokenizer_dataset_path = f"{tokenizer_name}_{dataset_name}_tokenized_dataset"
    try:
        tokenized_dataset = datasets.load_from_disk(tokenizer_dataset_path)
    except FileNotFoundError:
        tokenized_dataset = dataset.map(
            tokenizer_function,
            batched=True,
            batch_size=tokenizer_batch_size,
            num_proc=os.cpu_count(),  # Use all CPU cores.
        )

        tokenized_dataset.save_to_disk(tokenizer_dataset_path)
    return tokenized_dataset
