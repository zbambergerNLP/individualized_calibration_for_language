import math
import os

import accelerate
import torch
import model as individualized_calibration_model
from sklearn.metrics import confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import pickle


class MetricManager:
    """Saves and plots the metrics of this run.
        Also open a dir to save the metrics and the plots.

        Args:
            coefficient: The coefficient of the loss.
        """
    def __init__(self, coefficient, accelerator: accelerate.Accelerator):
        self.coefficient = coefficient
        self.accelerator = accelerator
        self.colors = ['red', 'orange', 'gold', 'lime', 'darkcyan', 'blue', 'cyan',
                       'magenta', 'purple', 'black', 'gray', 'brown']
        self.metrics = {}
        self.metrics_names = ['loss', 'loss_cdf', 'loss_stddev', 'loss_nll', 'Recall', 'FPR', 'Precision', 'F1',
                              'Accuracy', 'BPSN_AUC', 'BNSP_AUC', 'lr']


        # Opens a dir to save the metrics and the plots, with the name "exp{}_coef_<>"
        self.metric_dir_path = "./metrics"
        os.makedirs(self.metric_dir_path, exist_ok=True)
        dirs = [d for d in os.listdir(self.metric_dir_path) if os.path.isdir(os.path.join(self.metric_dir_path, d))]

        # Check if the directory is empty
        if not dirs:
            # If it is empty, create a new directory with the name "exp1_coef_{coefficient}"
            self.exp_name = "exp1_coef_{}".format(self.coefficient)
            self.new_dir_name = os.path.join(self.metric_dir_path, "exp1_coef_{}".format(self.coefficient))
            if accelerator.is_local_main_process:
                os.mkdir(self.new_dir_name)
        else:
            # If it is not empty, find the directory with the largest "exp" number
            exp_numbers = [int(d.split('exp')[1].split('_')[0]) for d in dirs if 'exp' in d]
            max_exp_number = max(exp_numbers) if exp_numbers else 0
            # Create a new directory with the name "exp{max_exp_number + 1}_coef_{coefficient}"
            self.exp_name = "exp{}_coef_{}".format(max_exp_number + 1, self.coefficient)
        self.new_dir_name = os.path.join(self.metric_dir_path, self.exp_name)
        if accelerator.is_local_main_process:
            os.mkdir(self.new_dir_name)

    def add_dict_metrics(self, step, metrics_dict):
        for metric_name, metric_value in metrics_dict.items():
            self.add_metric(step, metric_name, metric_value)

    def add_metric(self, step, metric_name, metric_value):
        if metric_name not in self.metrics.keys():
            self.metrics[metric_name] = {}
            self.metrics[metric_name]['steps'] = []
            self.metrics[metric_name]['values'] = []
        self.metrics[metric_name]['steps'].append(step)
        self.metrics[metric_name]['values'].append(metric_value)

    def create_all_metrics_plots(self):
        # Create a new figure for each metric and save it in the directory
        if self.accelerator.is_local_main_process:
            for metric_name in self.metrics_names:
                label_num = 0
                plt.figure(figsize=(12, 6))
                for key, metric_dict in self.metrics.items():
                    # Check if the key ends with the current metric
                    if key.endswith(metric_name):
                        # If it does, plot the values with the key as the label
                        plt.plot(metric_dict['steps'], metric_dict['values'], label=key, color=self.colors[label_num])
                        label_num += 1

                # Add a legend, title and save the figure
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.title("{} (alpha = {})".format(metric_name, self.coefficient))
                plt.tight_layout()  # Adjust the spacing around the plot
                plt.savefig(self.new_dir_name + "/" + metric_name + '.png')
                plt.close()

    def save_metrics(self):
        # Save the metrics in a file in the directory
        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.new_dir_name, 'metrics.pkl'), 'wb') as fp:
                pickle.dump(self.metrics, fp)


def compute_metrics(
        eval_pred: individualized_calibration_model.CommentRegressorPrediction,
        coefficient: float,
        eval_with_sample: bool = False,
        prefix: str = "eval"
) -> dict[str, float]:
    """Compute metrics for the evaluation predictions (e.g., TPR, FPR, AUC, precision, recall, F1).

    :param eval_pred: The output of the model on the evaluation set.
        Contains the following fields:
        - label_ids: The true labels for each sample in the evaluation set.
        - input_r: The input text for each sample in the evaluation set.
        - means: The predicted means of the Gaussian distribution for each sample in the evaluation set.
        - stddevs: The predicted standard deviations of the Gaussian distribution for each sample in the evaluation set.
    :param eval_with_sample: Whether to evaluate with a sample from the predicted distribution or with the mean of the
        predicted distribution.
    :param coefficient: The coefficient of the loss.
    :param prefix: The prefix to use for the metrics.
    :return: A dictionary containing the metrics.
        Contains the following metrics:
        - TPR: True positive rate (TP / (TP + FN))
        - FPR: False positive rate (FP / (FP + TN))
        - AUC: Area under the ROC curve
        - precision: Precision (TP / (TP + FP))
        - recall: Recall (TP / (TP + FN))
        - F1: F1 score (2 * precision * recall / (precision + recall))
    """
    target = torch.as_tensor(eval_pred.label_ids, dtype=torch.float32)
    input_r = torch.as_tensor(eval_pred.input_r, dtype=torch.float32)
    means = torch.as_tensor(eval_pred.means, dtype=torch.float32)
    stddevs = torch.as_tensor(eval_pred.stddevs, dtype=torch.float32)
    groups = {
        group_name: torch.as_tensor(group_values, dtype=torch.float32)
        for group_name, group_values in eval_pred.groups.items()
    }

    tmp_metrics_dict = {}
    metrics_dict = {}

    if eval_with_sample:
        # Sample from the CDF to get the predicted values.
        pred_values = torch.normal(means, stddevs)
    else:
        # Use the mean of the predicted distribution.
        pred_values = means

    # Clip the pred to be between 0 and 1
    pred_values = torch.clamp(pred_values, 0, 1)

    # Splits the range of values for 2 groups: [0, 0.5], (0.5, 1]
    pred_labels = (pred_values > 0.5).int()
    target_labels = (target > 0.5).int()

    # Since we are evaluating on the entire evaluation set, we can compute the metrics for each group in a constant
    # set of groups.
    for group_name in groups.keys():
        group_mask = torch.not_equal(groups[group_name], -1.0) & (groups[group_name] >= 0.5)
        if group_mask.sum() == 0:
            # If there are no samples in the group, skip it.
            continue
        tmp_metrics_dict[group_name] = compute_metrics_from_pred(
            input_r=input_r,
            mean=means,
            stddev=stddevs,
            pred_values=pred_values,
            pred_labels=pred_labels,
            target_labels=target_labels,
            target=target,
            group_mask=group_mask,
            coefficient=coefficient,
        )

    # Update the metrics_dict with the tmp_metrics_dict
    for group_name in tmp_metrics_dict.keys():
        for metric in tmp_metrics_dict[group_name].keys():
            metrics_dict[f"{prefix}_{group_name }_{metric}"] = tmp_metrics_dict[group_name][metric]

    # for each metric, Compute the biggest differences between the groups
    # TODO: Recover the biggest differences between the groups for each metric (by uncommenting the below).
    # for metric in tmp_metrics_dict[groups.keys()[0]].keys():
    #     min_metric = min([tmp_metrics_dict[group_name][metric] for group_name in groups.keys()])
    #     max_metric = max([tmp_metrics_dict[group_name][metric] for group_name in groups.keys()])
    #     metrics_dict[f"biggest_diffs_{metric}"] = max_metric - min_metric

    # Compute metrics for the full dataset
    metrics_dict.update(
        compute_metrics_from_pred(
            input_r=input_r,
            mean=means,
            stddev=stddevs,
            pred_values=pred_values,
            pred_labels=pred_labels,
            target_labels=target_labels,
            target=target,
            group_mask=torch.ones_like(target, dtype=torch.bool),
            full_dataset=True,
            coefficient=coefficient)
    )

    return metrics_dict


def compute_metrics_from_pred(
        coefficient: float,
        input_r: torch.Tensor = None,
        mean: torch.Tensor = None,
        stddev: torch.Tensor = None,
        pred_values: torch.Tensor = None,
        pred_labels: torch.Tensor = None,
        target_labels: torch.Tensor = None,
        target: torch.Tensor = None,
        group_mask: torch.Tensor = None,
        full_dataset: bool = False,
) -> dict[str, float]:
    """
    Compute metrics for the evaluation predictions (e.g., TPR, FPR, AUC, precision, recall, F1).

    :param input_r: The inputted random variable for each sample in the evaluation set.
        A tensor of shape (num_samples,) who's values are in the range [0, 1].
    :param mean: The predicted means of the Gaussian distribution for each sample in the evaluation set.
        A tensor of shape (num_samples,) who's values are in the range [0, 1].
    :param stddev: The predicted standard deviations of the Gaussian distribution for each sample in the evaluation set.
        A tensor of shape (num_samples,).
    :param pred_labels: The predicted labels for each sample in the evaluation set.
    :param target_labels: The target labels for each sample in the evaluation set.
    :param target: The target random variable for each sample in the evaluation set.
    :param group_mask: A mask that determines which samples belong to the group.
    :param full_dataset: Whether to compute the metrics for the full dataset.
    :return: A dictionary of metrics. The keys are the metric names and the values are the metric values.
    """

    # Calculate the losses
    cdf = (
            0.5 * (1.0 + torch.erf((target[group_mask] - mean[group_mask]) / stddev[group_mask] / math.sqrt(2)))
    )
    loss_cdf = torch.abs(cdf - input_r[group_mask]).mean()
    loss_stddev = stddev[group_mask].mean()
    loss_nll = (
            torch.log(stddev[group_mask]) +
            math.log(2 * math.pi) / 2.0 +
            (((target[group_mask] - mean[group_mask]) / stddev[group_mask]) ** 2 / 2.0)
    )
    loss_nll = loss_nll.mean()
    loss = (1 - coefficient) * loss_cdf + coefficient * loss_nll

    # TODO: Consider reporting when denominators are 0.

    # Calculate the average TPR, FPR, precision, F1 and accuracy
    tn, fp, fn, tp = confusion_matrix(target_labels[group_mask], pred_labels[group_mask]).ravel()

    # False Positive Rate - The proportion of negative instances that are incorrectly classified as positive
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # True Positive Rate (recall) - The proportion of positive instances that are correctly classified as positive
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Accuracy - the number of correct predictions made by the model, divided by the total number of predictions
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision - the proportion of true positive predictions among all positive predictions
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 - the harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Return those metrics
    metrics = {
        'loss': loss,
        'loss_cdf': loss_cdf,
        'loss_stddev': loss_stddev,
        'loss_nll': loss_nll,
        'Recall': recall,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy,
    }

    # if not full_dataset:
    #     # BPSN (Background Positive, Subgroup Negative) AUC -
    #     # Restrict the test set to non-abusive examples that mention the identity and abusive examples that do not.
    #     # print(f'group_mask: {group_mask}')
    #     # print(f'group mask has {group_mask.sum()} True values')
    #     # print(f'target_labels: {target_labels}')
    #     # print(f'pred_values: {pred_values}')
    #     print(f'num target labels in group greater than 0.5: {(target_labels[group_mask] > 0.5).sum()}')
    #     print(f'num target labels in group less than 0.5: {(target_labels[group_mask] < 0.5).sum()}')
    #     print(f'num target labels in group equal to 0: {(target_labels[group_mask] == 0).sum()}')
    #     print(f'num target labels in group equal to 1: {(target_labels[group_mask] == 1).sum()}')
    #     bpsn_mask = (target_labels & ~group_mask) | (~target_labels & group_mask)
    #     print(f'bpsn_mask: {bpsn_mask}')
    #     print(f'bpsn mask has {bpsn_mask.sum()} True values')
    #     print(f'target_labels[bpsn_mask]: {target_labels[bpsn_mask]}')
    #     if target_labels[bpsn_mask].sum() == 0:
    #         print('BPSN mask has no positive labels')
    #     bpsn_auc = roc_auc_score(target_labels[bpsn_mask].cpu().numpy(), pred_values[bpsn_mask].cpu().numpy())
    #
    #     # BNSP (Background Negative, Subgroup Positive) AUC -
    #     # Restrict the test set to abusive examples that mention the identity and non-abusive examples that do not.
    #     bnsp_mask = (target_labels & group_mask) | (~target_labels & ~group_mask)
    #     bnsp_auc = roc_auc_score(target_labels[bnsp_mask].cpu().numpy(), pred_values[bnsp_mask].cpu().numpy())
    #
    #     metrics['BPSN_AUC'] = bpsn_auc
    #     metrics['BNSP_AUC'] = bnsp_auc

    return metrics
