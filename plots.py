import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch


def end_of_training_plots(
        eval_result: dict,
        alpha: float,
):
    """
    Plots the results of the evaluation at the end of the training.

    :param eval_result: The evaluation results. A dictionary with the following keys:
        - 'loss_nll': The negative log-likelihood loss.
        - 'loss_stddev': The standard deviation loss.
        - 'loss_cdf': The CDF loss.
        - 'Accuracy': The accuracy.
        - 'F1': The F1 score.
        - 'TPR': The true positive rate.
        - 'FPR': The false positive rate.
        - 'BPSN_AUC': The BPSN AUC.
        - 'BNSP_AUC': The BNSP AUC.
        - 'biggest_diffs_loss_nll': The biggest difference in the negative log-likelihood loss between the groups.
        - 'biggest_diffs_loss_stddev': The biggest difference in the standard deviation loss between the groups.
        - 'biggest_diffs_loss_cdf': The biggest difference in the CDF loss between the groups.
        - 'biggest_diffs_Accuracy': The biggest difference in the accuracy between the groups.
        - 'biggest_diffs_F1': The biggest difference in the F1 score between the groups.
        - 'biggest_diffs_TPR': The biggest difference in the true positive rate between the groups.
        - 'biggest_diffs_FPR': The biggest difference in the false positive rate between the groups.
        - 'biggest_diffs_BPSN_AUC': The biggest difference in the BPSN AUC between the groups.
        - 'biggest_diffs_BNSP_AUC': The biggest difference in the BNSP AUC between the groups.
        - 'loss_nll_worst_group': The worst negative log-likelihood loss between the groups.
        - 'loss_stddev_worst_group': The worst standard deviation loss between the groups.
        - 'loss_cdf_worst_group': The worst CDF loss between the groups.
        - 'Accuracy_worst_group': The worst accuracy between the groups.
        - 'F1_worst_group': The worst F1 score between the groups.
        - 'TPR_worst_group': The worst true positive rate between the groups.
        - 'FPR_worst_group': The worst false positive rate between the groups.
        - 'BPSN_AUC_worst_group': The worst BPSN AUC between the groups.
        - 'BNSP_AUC_worst_group': The worst BNSP AUC between the groups.

    :param alpha: The coefficient of the loss function (balances between the performance-oriented loss and the
        fairness-oriented loss).
    """
    # Converts any PyTorch tensor values to regular integers
    for key, value in eval_result.items():
        if torch.is_tensor(value):
            eval_result[key] = value.detach().cpu().item()  # Convert tensor to a Python number

    # Finds the worst metric between all the groups
    # 'key != metric' so we won't take the metric on the all dataset
    for metric in ['loss_nll', 'loss_stddev', 'loss_cdf', 'FPR']:
        eval_result[metric + '_worst_group'] = (
            max(value for key, value in eval_result.items()
                if key.endswith(metric) and key != metric and key != 'biggest_diffs_' + metric)
        )

    for metric in ['Accuracy', 'F1', 'TPR', 'BPSN_AUC', 'BNSP_AUC']:  # TODO: good BNSP_AUC, BPSN_AUC is high, right?
        eval_result[metric + '_worst_group'] = (
            min(value for key, value in eval_result.items()
                if key.endswith(metric) and key != metric and key != 'biggest_diffs_' + metric)
        )

    # TODO: I created graphs like in the article. y axis is for the overall performances (sharpness), and x axis is
    #   for fairness. we need to think what more graphs we want.

    # Loss function graphs
    update_plot(eval_result, key_x='loss_cdf_worst_group', key_y='loss_nll', alpha=alpha)
    update_plot(eval_result, key_x='loss_cdf_worst_group', key_y='loss_stddev', alpha=alpha)

    update_plot(eval_result, key_x='biggest_diffs_loss_nll', key_y='loss_nll', alpha=alpha)
    update_plot(eval_result, key_x='biggest_diffs_loss_cdf', key_y='loss_stddev', alpha=alpha)

    # Classification metrics graphs
    update_plot(eval_result, key_x='Accuracy_worst_group', key_y='Accuracy', alpha=alpha)
    update_plot(eval_result, key_x='F1_worst_group', key_y='F1', alpha=alpha)
    update_plot(eval_result, key_x='TPR_worst_group', key_y='TPR', alpha=alpha)
    update_plot(eval_result, key_x='FPR_worst_group', key_y='FPR', alpha=alpha)

    update_plot(eval_result, key_x='biggest_diffs_Accuracy', key_y='Accuracy', alpha=alpha)
    update_plot(eval_result, key_x='biggest_diffs_F1', key_y='F1', alpha=alpha)
    update_plot(eval_result, key_x='biggest_diffs_TPR', key_y='TPR', alpha=alpha)
    update_plot(eval_result, key_x='biggest_diffs_FPR', key_y='FPR', alpha=alpha)


def update_plot(
        data_dict: dict,
        key_x: str,
        key_y: str,
        alpha: float,
):
    """
    Updates a scatter plot of the given data.

    :param data_dict: A dictionary of the data to plot.
    :param key_x: The key of the x axis in the data dictionary.
    :param key_y: The key of the y axis in the data dictionary.
    :param alpha: The coefficient of the loss function (balances between the performance-oriented loss and the
        fairness-oriented loss).
    """
    dir_path = 'visualizations/scatters/{}_{}'.format(key_x, key_y)
    os.makedirs(dir_path, exist_ok=True)
    # Load existing plot data if it exists
    plot_data_path = dir_path + '/data.csv'
    plot_path = dir_path + '/graph.png'
    if os.path.exists(plot_data_path):
        with open(plot_data_path, 'rb') as f:
            plot_data = pd.read_csv(plot_data_path)
    else:
        plot_data = pd.DataFrame(columns=[key_x, key_y, 'alpha', 'timestamp'])

    # Add new point to plot data
    now = datetime.now()
    plot_data = plot_data.append(
        {key_x: data_dict[key_x], key_y: data_dict[key_y], 'alpha': alpha, 'timestamp': now},
        ignore_index=True,
    )

    # Recreate the plot
    plt.figure()
    sc = plt.scatter(plot_data[key_x], plot_data[key_y], c=plot_data['alpha'], cmap='coolwarm')
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    title = "{}, {}".format(key_x, key_y)
    plt.title(title)

    # Add a colorbar
    plt.colorbar(sc, label='alpha')

    # Save the plot as an image
    plt.savefig(plot_path)

    # Save the updated plot data
    plot_data.to_csv(plot_data_path, index=False)
