import pandas as pd
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

if __name__ == "__main__":
    # 1. Download the data and delete the unwanted columns.
    data = pd.read_csv('data/train.csv')
    data = data[GROUP_LIST + BASE_COLUMNS]
    print(f'Data dimensions: {data.shape}')

    # 2. Get statistics on the data
    updated_group_list = GROUP_LIST.copy()
    for group in GROUP_LIST:
        print("{}: {} positive labels | {} labels larger than 0.5".
              format(group, (data[group] > 0).sum(), (data[group] > 0.5).sum()))
    print("\n\n")

    # 3. Create and save a histogram of the "target" column
    # Create directory for plots
    os.makedirs('../visualizations/EDA', exist_ok=True)

    plt.hist(data['target'], bins=30, edgecolor='black')
    plt.title('Histogram of target')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.savefig('../visualizations/EDA/target_histogram.png')
    plt.close()

    # 4. Create and save a histogram for the groups columns
    for group in updated_group_list:
        not_toxic = data[data['target'] <= 0.5][group]
        toxic = data[data['target'] > 0.5][group]

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
        plt.savefig('../visualizations/EDA/{}_histogram.png'.format(group))
        plt.tight_layout()
        plt.close()

    # 5. Create and save a histogram for distribution of the different groups
    group_counts = []
    for group in updated_group_list:
        group_counts.append((data[group] > 0.5).sum())

    plt.bar(updated_group_list, group_counts, edgecolor='black')
    plt.xticks(rotation=45, ha='right')  # Rotate names 45 degrees
    plt.title('Histogram of distribution of the different groups (with label > 0.5)')
    plt.xlabel('Groups')
    plt.ylabel('Count')
    plt.savefig('../visualizations/EDA/groups_histogram.png')
    plt.tight_layout()
    plt.close()

    # 6. Split the rows into 3 different groups, when we want most of the rows with group values to be in the
    # eval and test sets.

    # Create a mask that is True for rows with null values in any of the specified fields
    mask = data[updated_group_list].isnull().any(axis=1)

    # Split the DataFrame into two datasets based on the mask
    data_with_nulls = data[mask]
    data_without_nulls = data[~mask]

    print(f'Data with nulls dimensions: {data_with_nulls.shape}')
    print(f'Data without nulls dimensions: {data_without_nulls.shape}')

    train_data_wit_nulls, temp_data_with_nulls = train_test_split(data_with_nulls, test_size=0.4, random_state=seed)
    eval_data_with_nulls, test_data_with_nulls = train_test_split(temp_data_with_nulls, test_size=0.5, random_state=seed)

    train_data_without_nulls, temp_data_without_nulls = train_test_split(data_without_nulls, test_size=0.05, random_state=seed)
    eval_data_without_nulls, test_data_without_nulls = train_test_split(temp_data_without_nulls, test_size=0.5, random_state=seed)

    train_data = pd.concat([train_data_wit_nulls, train_data_without_nulls])
    eval_data = pd.concat([eval_data_with_nulls, eval_data_without_nulls])
    test_data = pd.concat([test_data_with_nulls, test_data_without_nulls])

    train_data = shuffle(train_data, random_state=seed)
    eval_data = shuffle(eval_data, random_state=seed)
    test_data = shuffle(test_data, random_state=seed)

    # Save the datasets
    train_data.to_csv('data/train_data.csv', index=False)
    eval_data.to_csv('data/eval_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    print(f'Train data dimensions: {train_data.shape}')
    print(f'Eval data dimensions: {eval_data.shape}')
    print(f'Test data dimensions: {test_data.shape}')

