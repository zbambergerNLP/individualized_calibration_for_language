import os
import typing
import datasets


def create_dataloaders(
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

    print(f'Train dataset: {train_dataset}')
    print(f'Validation dataset: {validation_dataset}')
    print(f'Test dataset: {test_dataset}')

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
