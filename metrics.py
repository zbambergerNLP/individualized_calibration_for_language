import math

import torch
import model as individualized_calibration_model
from sklearn.metrics import confusion_matrix, roc_auc_score


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
        # group_mask = ~torch.isnan(groups[group_name]) & (groups[group_name] >= 0.5)  # This determines which samples belong to the group
        tmp_metrics_dict[group_name] = compute_metrics_from_pred(
            input_r=input_r,
            mean=means,
            stddev=stddevs,
            pred_labels=pred_labels,
            target_labels=target_labels,
            target=target,
            group_mask=group_mask,
        )

    # Update the metrics_dict with the tmp_metrics_dict
    for group_name in groups.keys():
        for metric in tmp_metrics_dict[group_name].keys():
            metrics_dict[group_name + "_" + metric] = tmp_metrics_dict[group_name][metric]

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
            pred_labels=pred_labels,
            target_labels=target_labels,
            target=target,
            group_mask=torch.ones_like(target, dtype=torch.bool),
            full_dataset=True,)
    )

    return metrics_dict


def compute_metrics_from_pred(
        input_r: torch.Tensor = None,
        mean: torch.Tensor = None,
        stddev: torch.Tensor = None,
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
        'loss_cdf': loss_cdf,
        'loss_stddev': loss_stddev,
        'loss_nll': loss_nll,
        'Recall': recall,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy,
    }

    if not full_dataset:
        # BPSN (Background Positive, Subgroup Negative) AUC -
        # Restrict the test set to non-abusive examples that mention the identity and abusive examples that do not.
        bpsn_mask = (target_labels & ~group_mask) | (~target_labels & group_mask)
        bpsn_auc = roc_auc_score(target_labels[bpsn_mask].cpu().numpy(), pred_labels[bpsn_mask].cpu().numpy())

        # BNSP (Background Negative, Subgroup Positive) AUC -
        # Restrict the test set to abusive examples that mention the identity and non-abusive examples that do not.
        bnsp_mask = (target_labels & group_mask) | (~target_labels & ~group_mask)
        bnsp_auc = roc_auc_score(target_labels[bnsp_mask].cpu().numpy(), pred_labels[bnsp_mask].cpu().numpy())

        metrics['BPSN_AUC'] = bpsn_auc
        metrics['BNSP_AUC'] = bnsp_auc

    return metrics
