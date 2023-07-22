import math

import torch
import model as individualized_calibration_model
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from data_preprocessing import GROUP_LIST


def compute_metrics(
        eval_pred: individualized_calibration_model.CommentRegressorPrediction,
        eval_with_sample: bool = False,
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

    # Load the eval_set.csv file
    # df = pd.read_csv("data/eval_set.csv")
    group_metrics = {}

    # TODO: Uncomment the below when we can access group information directly from the eval_pred object.
    #
    # # Since we are evaluating on the entire evaluation set, we can compute the metrics for each group in a constant
    # # set of groups.
    # for group in GROUP_LIST:
    #     group_mask = df[group].notna() & (df[group] >= 0.5)  # This determines which samples belong to the group
    #     group_metrics = compute_metrics_from_pred(
    #         input_r=input_r[group_mask],
    #         mean=means[group_mask],
    #         stddev=stddevs[group_mask],
    #         pred_labels=pred_labels[group_mask],
    #         target_labels=target_labels[group_mask],
    #         target=target[group_mask],
    #     )
    #
    #     # Add each metric from the group to the group_metrics dictionary
    #     for metric in group_metrics.keys():
    #         metric_name = f"{group}_{metric}"
    #         group_metrics[metric_name] = group_metrics[metric]
    #
    # # for each metric, Compute the biggest differences between the groups
    # for metric in group_metrics[GROUP_LIST[0]].keys():
    #     min_val = min([group_metrics[f'{group}_{metric}'] for group in GROUP_LIST])
    #     max_val = max([group_metrics[f'{group}_{metric}'] for group in GROUP_LIST])
    #     metric_name = f"biggest_diffs_{metric}"
    #     group_metrics[metric_name] = max_val - min_val

    # Compute metrics for the full dataset
    group_metrics.update(compute_metrics_from_pred(
        input_r=input_r,
        mean=means,
        stddev=stddevs,
        pred_labels=pred_labels,
        target_labels=target_labels,
        target=target)
    )

    return group_metrics


def compute_metrics_from_pred(
        input_r: torch.Tensor = None,
        mean: torch.Tensor = None,
        stddev: torch.Tensor = None,
        pred_labels: torch.Tensor = None,
        target_labels: torch.Tensor = None,
        target: torch.Tensor = None,
) -> dict[str, float]:
    # Calculate the losses
    cdf = 0.5 * (1.0 + torch.erf((target - mean) / stddev / math.sqrt(2)))
    loss_cdf = torch.abs(cdf - input_r).mean()
    loss_stddev = stddev.mean()
    loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((target - mean) / stddev) ** 2 / 2.0)
    loss_nll = loss_nll.mean()

    # Calculate the average TPR, FPR, precision, F1 and accuracy
    precision = precision_score(target_labels, pred_labels)
    recall = recall_score(target_labels, pred_labels)  # TPR
    f1 = f1_score(target_labels, pred_labels)
    accuracy = accuracy_score(target_labels, pred_labels)

    # FPR = FP / (FP + TN) = 1 - TNR, where TNR (True Negative Rate) = TN / (FP + TN) = 1 - FPR
    # So, we can calculate FPR as 1 - recall of the negative class
    fpr = 1 - recall_score(1 - target_labels, 1 - pred_labels)

    # Return those metrics
    metrics = {
        'loss_cdf': loss_cdf,
        'loss_stddev': loss_stddev,
        'loss_nll': loss_nll,
        'TPR': recall,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy,
    }
    return metrics
