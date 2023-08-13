import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch
import wandb


def end_of_training_plots(
        eval_result: dict,
        alpha: float,
        wandb_run: wandb.sdk.wandb_run.Run,
        data_title: str = "",
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
    :param wandb_run: The wandb run object.
    :param data_title: The title of the dataset.
    """
    # TODO: Uncomment code relating to BPSN_AUC and BNSP_AUC when they are implemented
    # TODO: Ensure that this method can operate in a distributed training setting.
    # Converts any PyTorch tensor values to regular integers
    for key, value in eval_result.items():
        if torch.is_tensor(value):
            eval_result[key] = value.detach().cpu().item()  # Convert tensor to a Python number

    # Finds the worst metric between all the groups
    # 'key != metric' so we won't take the metric on the all dataset
    # Metrics that smaller is better
    for metric in ['loss_nll', 'loss_stddev', 'loss_cdf', 'FPR']:
        cuur_list = (
            [value for key, value in eval_result.items()
             if key.endswith(metric) and key != metric and key != 'biggest_diffs_' + metric]
        )
        if len(cuur_list) == 0:
            continue
        eval_result[metric + '_worst_group'] = max(cuur_list)
        eval_result[metric + '_best_group'] = min(cuur_list)

    # Metrics that bigger is better
    for metric in ['Accuracy', 'F1', 'TPR', 'Precision', 'Recall']:
        curr_list = (
            [value for key, value in eval_result.items() if
             key.endswith(metric) and key != metric and key != 'biggest_diffs_' + metric])
        if len(curr_list) == 0:
            continue
        eval_result[metric + '_worst_group'] = min(curr_list)
        eval_result[metric + '_best_group'] = max(curr_list)

    # Finds the mean metric value between all the groups
    for metric in ['loss_nll', 'loss_stddev', 'loss_cdf', 'Accuracy', 'F1', 'TPR', 'FPR', 'Precision', 'Recall']:
        curr_list = (
            [value for key, value in eval_result.items()
             if key.endswith(metric) and key != metric and key != 'biggest_diffs_' + metric])
        if len(curr_list) == 0:
            continue
        eval_result[metric + '_mean'] = sum(curr_list) / len(curr_list)

    # Loss function graphs
    update_plot(eval_result, key_x='loss_cdf_worst_group', key_y='loss_nll', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='loss_cdf_worst_group', key_y='loss_stddev', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)

    # Classification metrics graphs (worst group)
    # update_plot(eval_result, key_x='BPSN_AUC_worst_group', key_y='F1', alpha=alpha, data_title=data_title,
    #             wandb_run=wandb_run)
    # update_plot(eval_result, key_x='BNSP_AUC_worst_group', key_y='F1', alpha=alpha, data_title=data_title,
    #             wandb_run=wandb_run)
    update_plot(eval_result, key_x='Accuracy_worst_group', key_y='Accuracy', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='F1_worst_group', key_y='F1', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='TPR_worst_group', key_y='TPR', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='FPR_worst_group', key_y='FPR', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='Recall_worst_group', key_y='Recall', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='Precision_worst_group', key_y='Precision', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)

    # Classification metrics graphs (best group)
    # update_plot(eval_result, key_x='BPSN_AUC_best_group', key_y='BPSN_AUC_worst_group', alpha=alpha, data_title=data_title,
    #     wandb_run=wandb_run)
    # update_plot(eval_result, key_x='BNSP_AUC_best_group', key_y='BNSP_AUC_worst_group', alpha=alpha, data_title=data_title,
    #             wandb_run=wandb_run)
    update_plot(eval_result, key_x='Accuracy_best_group', key_y='Accuracy_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='F1_best_group', key_y='F1_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='TPR_best_group', key_y='TPR_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='FPR_best_group', key_y='FPR_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='Recall_worst_group', key_y='Recall_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)
    update_plot(eval_result, key_x='Precision_worst_group', key_y='Precision_worst_group', alpha=alpha, data_title=data_title,
                wandb_run=wandb_run)


def update_plot(
        data_dict: dict,
        key_x: str,
        key_y: str,
        alpha: float,
        data_title: str,
        wandb_run,
):
    """
    Updates a scatter plot of the given data.

    :param data_dict: A dictionary of the data to plot.
    :param key_x: The key of the x axis in the data dictionary.
    :param key_y: The key of the y axis in the data dictionary.
    :param alpha: The coefficient of the loss function (balances between the performance-oriented loss and the
        fairness-oriented loss).
    :param data_title: The title of the data.
    :param wandb_run: The wandb run object.
    """
    if key_x not in data_dict or key_y not in data_dict:
        return

    dir_path = 'visualizations/scatters/{}_{}'.format(key_x, key_y)
    os.makedirs(dir_path, exist_ok=True)
    # Load existing plot data if it exists
    plot_data_path = '{}/data_{}.csv'.format(dir_path, data_title)
    plot_path = '{}/graph_{}.png'.format(dir_path, data_title)
    if os.path.exists(plot_data_path):
        with open(plot_data_path, 'rb') as fileHandle:
            plot_data = pd.read_csv(fileHandle)
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
    if data_title == "":
        title = "{} vs. {}".format(key_x, key_y)
    else:
        title = "{} vs. {} - ({})".format(key_x, key_y, data_title)
    plt.title(title, fontsize=9)
    plt.tick_params(axis='both', which='major', labelsize=7)  # Change the font size of the x-axis values

    # Add a colorbar
    plt.colorbar(sc, label='alpha')

    # Save the plot as an image
    plt.savefig(plot_path)

    # Save the updated plot data
    plot_data.to_csv(plot_data_path, index=False)
    wandb_run.log({title: wandb.Image(plot_path)})
