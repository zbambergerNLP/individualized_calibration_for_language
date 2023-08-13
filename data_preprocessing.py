import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

# To run this script using Newton, run the following command:
# srun --partition=nlp --account=nlp --gres=gpu:1 -c 10 python data_preprocessing.py
# The above command will request 1 GPU and 10 CPU cores. You can change these values as needed.
# Note that you will need to adjust the partition and account to match your Newton account.

# TODO: Create a perser for this script.
# TODO: Create helper functions for the data processing steps so that they can be called from main.py.
#  We can tell whether we need to do this by checking whether the files created by this function already exist.
#  If they do, we can skip this step.

GROUP_LIST = [
    'black',
    'white',
    'christian',
    'female',
    'male',
    'homosexual_gay_or_lesbian',
    'jewish',
    'muslim',
    'psychiatric_or_mental_illness',
]

BASE_COLUMNS = [
    'id',
    'target',
    'comment_text',
]

seed = 42
# BASE_DIR = '../'
BASE_DIR = './'


# Checks the distribution of the target variable in the given DataFrame.
def target_dist_check(
        df: pd.DataFrame,
        df_name: str,
        print_comments: bool = False,
        num_ranges: int = 2,
):
    """Checks the distribution of the target variable in the given DataFrame.

    Args:
        df: The DataFrame to check.
        df_name: The name of the DataFrame.
        print_comments: Whether to print comments from the DataFrame.
        num_ranges: The number of ranges to use for the target variable.
    """
    if num_ranges == 4:
        ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    elif num_ranges == 2:
        ranges = [(0, 0.5), (0.5, 1)]
    else:
        raise ValueError("num_ranges must be 2 or 4")
    counts = {}

    for lower, upper in ranges:
        count = df[(df['target'] >= lower) & (df['target'] < upper)].shape[0]
        counts[f'[{lower}, {upper})'] = count

    text = df_name + " target counting: "
    for index, count in counts.items():
        text += f'{index}: {count} | '
    print(text)

    if print_comments:
        for lower, upper in ranges:
            sample_row_1 = df[(df['target'] >= lower) & (df['target'] < upper)].sample(n=1)
            sample_row_2 = df[(df['target'] >= lower) & (df['target'] < upper)].sample(n=1)
            print(
                f'Range: [{lower}, {upper}) '
                'Comments:'
                f'\n{sample_row_1["comment_text"].values[0]}\n\n'
                f'{sample_row_2["comment_text"].values[0]}\n'
            )


def drop_nontoxic_rows(
        df: pd.DataFrame,
        percentages: typing.List[float],
) -> pd.DataFrame:
    """Drops rows from the given DataFrame that are not toxic.

    Args:
        df: The DataFrame to drop rows from.
        percentages: The percentages of rows to drop from each range.

    Returns:
        The DataFrame with the rows dropped.
    """
    # Define the ranges
    ranges = [(0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]

    # Make sure the DataFrame is not modified in place
    df = df.copy()

    for (lower, upper), percentage in zip(ranges, percentages):
        # Get the rows in the current range
        mask = (df['target'] >= lower) & (df['target'] < upper)
        rows_in_range = df[mask]

        # Calculate the number of rows to drop
        n_to_drop = int(len(rows_in_range) * percentage)

        # Get the indices of the rows to drop
        indices_to_drop = np.random.choice(rows_in_range.index, size=n_to_drop, replace=False)

        # Drop the rows
        df = df.drop(indices_to_drop)

    return df


# Creates a histogram of the given group labels in the given DataFrame.
def create_group_histogram(
        clean_data: pd.DataFrame,
        group: str,
):
    """Creates a histogram of the given group labels in the given DataFrame.

    Args:
        clean_data: The DataFrame to create the histogram from.
        group: The group to create the histogram for.
    """
    not_toxic = clean_data[clean_data['target'] <= 0.5][group]
    toxic = clean_data[clean_data['target'] > 0.5][group]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Left plot
    axs[0].hist([not_toxic, toxic], bins=30, edgecolor='black', label=['not toxic', 'toxic'])
    axs[0].set_title('Full Data')
    axs[0].set_xlabel('"{}" label'.format(group))
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Right plot
    not_toxic_zoom = not_toxic[not_toxic >= 0.1]
    toxic_zoom = toxic[toxic >= 0.1]
    axs[1].hist([not_toxic_zoom, toxic_zoom], bins=30, edgecolor='black', label=['not toxic', 'toxic'])
    axs[1].set_title('Zoom')
    axs[1].set_xlabel('"{}" label'.format(group))
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    fig.suptitle('Histogram of "{}" label'.format(group))
    plt.savefig(BASE_DIR + 'visualizations/EDA/{}_histogram.png'.format(group))
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    # Download the data and delete the unwanted columns.
    data = pd.read_csv(BASE_DIR + 'data/train.csv')
    data = data[GROUP_LIST + BASE_COLUMNS]
    print(f'Data dimensions: {data.shape}')

    # Check the distribution of the target variable in the datasets
    print()
    target_dist_check(data, "full data")
    print()

    # Split the DataFrame into two datasets based on having null in the group labels
    mask = data[GROUP_LIST].isnull().any(axis=1)
    data_with_nulls = data[mask]
    data_with_nulls.loc[:, GROUP_LIST] = -1.0
    data_without_nulls = data[~mask]

    print(f'Data with nulls dimensions: {data_with_nulls.shape}')
    print(f'Data without nulls dimensions: {data_without_nulls.shape} \n')

    # Check the distribution of the target variable in the two datasets
    print()
    target_dist_check(data_with_nulls, "data with nulls in groups")
    target_dist_check(data_without_nulls, "data without nulls in groups", print_comments=False)
    print()

    # Balancing the toxic and non-toxic comments in the two datasets
    data_with_nulls = drop_nontoxic_rows(data_with_nulls, percentages=[0.98, 0.85, 0.7, 0, 0])
    data_without_nulls = drop_nontoxic_rows(data_without_nulls, percentages=[0.91, 0.5, 0.4, 0, 0])

    print("\nAfter dropping some of the untoxic comments:\n")
    target_dist_check(data_with_nulls, "data with nulls in groups", num_ranges=4)
    target_dist_check(data_with_nulls, "data with nulls in groups", num_ranges=2)
    print()
    target_dist_check(data_without_nulls, "data without nulls in groups", num_ranges=4)
    target_dist_check(data_without_nulls, "data without nulls in groups", num_ranges=2)
    print()

    clean_data = pd.concat([data_with_nulls, data_without_nulls])
    target_dist_check(clean_data, "full data", num_ranges=2)

    # Split the dataset into 4 different datasets
    train_data_with_nulls, temp_data_with_nulls = train_test_split(
        data_with_nulls, test_size=0.25, random_state=seed)
    calib_data_with_nulls, temp_data_with_nulls = train_test_split(
        temp_data_with_nulls, test_size=0.67, random_state=seed)
    eval_data_with_nulls, test_data_with_nulls = train_test_split(
        temp_data_with_nulls, test_size=0.5, random_state=seed)

    train_data_without_nulls, temp_data_without_nulls = train_test_split(
        data_without_nulls, test_size=0.5, random_state=seed)
    calib_data_without_nulls, temp_data_without_nulls = train_test_split(
        temp_data_without_nulls, test_size=0.67, random_state=seed)
    eval_data_without_nulls, test_data_without_nulls = train_test_split(
        temp_data_without_nulls, test_size=0.5, random_state=seed)

    train_data = pd.concat([train_data_with_nulls, train_data_without_nulls])
    calib_data = pd.concat([calib_data_with_nulls, calib_data_without_nulls])
    #eval_data = pd.concat([eval_data_with_nulls, eval_data_without_nulls])
    #test_data = pd.concat([test_data_with_nulls, test_data_without_nulls])

    eval_data = eval_data_without_nulls
    test_data = test_data_without_nulls

    train_data = shuffle(train_data, random_state=seed)

    # Check the distribution of the target variable in each dataset
    print()
    target_dist_check(train_data, "train data")
    target_dist_check(calib_data, "calib data")
    target_dist_check(eval_data, "eval data")
    target_dist_check(test_data, "test data")
    print()

    # Save the datasets
    train_data.to_csv(BASE_DIR + 'data/train_data.csv', index=False)
    calib_data.to_csv(BASE_DIR + 'data/calib_data.csv', index=False)
    eval_data.to_csv(BASE_DIR + 'data/eval_data.csv', index=False)
    test_data.to_csv(BASE_DIR + 'data/test_data.csv', index=False)

    print(f'Train data dimensions: {train_data.shape}')
    print(f'Calib data dimensions: {calib_data.shape}')
    print(f'Eval data dimensions: {eval_data.shape}')
    print(f'Test data dimensions: {test_data.shape}')


    ################################################################################
    # EDA for the data
    print("\n\n############ EDA ############\n\n")

    # Get statistics on the data
    updated_group_list = GROUP_LIST.copy()
    for group in GROUP_LIST:
        print("{}: {} positive labels | {} labels larger than 0.5".
              format(group, (clean_data[group] > 0).sum(), (clean_data[group] > 0.5).sum()))
    print("\n\n")

    # Create and save a histogram of the "target" column
    os.makedirs(BASE_DIR + 'visualizations/EDA', exist_ok=True)

    plt.hist(clean_data['target'], bins=30, edgecolor='black')
    plt.title('Histogram of target')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.savefig(BASE_DIR + 'visualizations/EDA/target_histogram.png')
    plt.savefig(BASE_DIR + 'visualizations/EDA/target_histogram.png')
    plt.close()

    # Create and save a histogram for the groups columns
    for group in updated_group_list:
        create_group_histogram(eval_data, group)

    # Create and save a histogram for distribution of the different groups
    group_counts = []
    for group in updated_group_list:
        group_counts.append((eval_data[group] > 0.5).sum())

    plt.bar(updated_group_list, group_counts, edgecolor='black')
    plt.xticks(rotation=45, ha='right')  # Rotate names 45 degrees
    plt.title('Histogram of distribution of the different groups (with label > 0.5)')
    plt.xlabel('Groups')
    plt.ylabel('Count')
    plt.savefig(BASE_DIR + 'visualizations/EDA/groups_histogram.png')
    plt.tight_layout()
    plt.close()
