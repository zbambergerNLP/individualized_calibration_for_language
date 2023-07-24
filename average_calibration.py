import typing

import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

import datasets
from torch.utils.data import DataLoader

from model import CommentRegressor, CalibratedCommentRegressor
from torch.utils.data import TensorDataset, DataLoader



# Preform calibration on a CommentRegressor
def calibration(comment_regressor: CommentRegressor,
                calibration_dataset: datasets.Dataset,
                batch_size: int = 16,
                ) -> CalibratedCommentRegressor:

    print("Start calibration")

    # Creating the data for training the calibration layer
    # TODO: I created myself dataloader from the dataset, is that ok?
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    mean_list, stddev_list, cdf_list, target = get_cdf_list(calibration_loader, comment_regressor)
    dataset_for_calibration = TensorDataset(mean_list, stddev_list, cdf_list, target)
    dataloader_for_calibration = DataLoader(dataset_for_calibration, batch_size=5_000, shuffle=True)

    # Train the calibration layer
    calibrated_comment_regressor = train_calibration_layer(comment_regressor, dataloader_for_calibration)
    print("Finish calibration")
    return calibrated_comment_regressor


# Get the CommentRegressor cdf list on the calibration dataset
def get_cdf_list(calibration_loader, comment_regressor):
    comment_regressor.eval()
    mean_list = torch.Tensor(0)
    stddev_list = torch.Tensor(0)
    cdf_list = torch.Tensor(0)

    with torch.no_grad():
        for step, inputs in enumerate(calibration_loader):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            target = inputs.get("labels")
            #input_r = inputs.get("input_r")

            # TODO: I sampled input_r from uniform distribution, because I think it isn't exist in the dataset, is that ok?
            input_r = torch.rand(
                input_ids.shape[0],
                1,
                device=input_ids.device)

            outputs = comment_regressor.forward(input_ids, attention_mask, input_r)
            mean, stddev = outputs
            cdf = 0.5 * (1.0 + torch.erf((target - mean) / stddev / math.sqrt(2)))

            mean_list = torch.cat((mean_list, mean.detach()), dim=0)
            stddev_list = torch.cat((stddev_list, stddev.detach()), dim=0)
            cdf_list = torch.cat((cdf_list, cdf.detach()), dim=0)

    cdf_list, indices = torch.sort(cdf_list, dim=0)
    mean_list = mean_list[indices]
    stddev_list = stddev_list[indices]

    lin = np.linspace(0, 1, int(cdf_list.shape[0]))

    # Trim CDF for numerical stability
    cdf_list = np.clip(cdf_list, a_max=1.0 - 1e-6, a_min=1e-6)

    return mean_list, stddev_list, cdf_list, lin



def train_calibration_layer(comment_regressor, dataloader_for_calibration):
    # Model - we will train only the calibration layer
    calibrated_comment_regressor = CalibratedCommentRegressor(comment_regressor)
    calibrated_comment_regressor.comment_regressor.eval()
    calibrated_comment_regressor.calibration_layer.train()

    # Optimizer
    optimizer = torch.optim.SGD(calibrated_comment_regressor.calibration_layer.parameters(), lr=0.001)

    # Loss function
    loss_function = nn.MSELoss()

    for step, batch in enumerate(dataloader_for_calibration):
        mean_list, stddev_list, cdf_list, target = batch
        optimizer.zero_grad()
        cali_mean, calib_std = calibrated_comment_regressor.calibration_layer(mean_list, stddev_list)
        cdf_pred = 0.5 * (1.0 + torch.erf((target - cali_mean) / calib_std / math.sqrt(2)))
        loss = loss_function(cdf_pred, target)
        loss.backward()
        optimizer.step()

        if step % int(len(dataloader_for_calibration)/10) == 0:
            print(f"Step: {step}, Calibration MSE loss: {loss}")

    return calibrated_comment_regressor
