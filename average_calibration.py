import typing

import numpy as np
import math
import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import datasets
from torch.utils.data import DataLoader

from model import CommentRegressor, CalibratedCommentRegressor
from torch.utils.data import TensorDataset, DataLoader



# Preform calibration on a CommentRegressor
def perform_average_calibration(comment_regressor: CommentRegressor,
                                calibration_dataloader,
                                ) -> CalibratedCommentRegressor:

    print("Start calibration")

    # Creating the data for training the calibration layer
    mean_list, stddev_list, cdf_list, target = get_cdf_list(calibration_dataloader, comment_regressor)
    dataset_for_calibration = CalibrationLayerDataSet(mean_list=mean_list,
                                                      stddev_list=stddev_list,
                                                      cdf_list=cdf_list,
                                                      target=target)
    dataloader_for_calibration = DataLoader(dataset_for_calibration, batch_size=1_000, shuffle=True)

    # Train the calibration layer
    calibrated_comment_regressor = train_calibration_layer(comment_regressor, dataloader_for_calibration)
    print("Finish calibration")
    return calibrated_comment_regressor


# Get the CommentRegressor cdf list on the calibration dataset
def get_cdf_list(calibration_loader, comment_regressor):
    comment_regressor.eval()
    means_list = []
    stddevs_list = []
    cdf_list = []

    with torch.no_grad():
        for step, batch in tqdm.tqdm(
                enumerate(calibration_loader),
                desc="Creating CDFs for the calibration",
                total=len(calibration_loader),
                position=2,
        ):

            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "input_r": batch["input_r"],
            }
            model_outputs = comment_regressor(**model_inputs)
            mean, stddev = model_outputs
            cdf = 0.5 * (1.0 + torch.erf((batch["labels"] - mean) / stddev / math.sqrt(2)))

            means_list.append(mean.cpu())
            stddevs_list.append(stddev.cpu())
            cdf_list.append(cdf.cpu())

    mean_list_final = np.array(torch.cat(means_list))
    stddev_list_final = np.array(torch.cat(stddevs_list))
    cdf_list_final = np.array(torch.cat(cdf_list))

    # Sorting the lists
    sorted_indices = np.argsort(cdf_list_final)

    mean_list_final = mean_list_final[sorted_indices]
    stddev_list_final = stddev_list_final[sorted_indices]
    cdf_list_final = cdf_list_final[sorted_indices]

    lin = np.linspace(0, 1, int(cdf_list_final.shape[0]))

    # Trim CDF for numerical stability
    cdf_list_final = np.clip(cdf_list_final, a_max=1.0 - 1e-6, a_min=1e-6)

    return mean_list_final, stddev_list_final, cdf_list_final, lin



def train_calibration_layer(comment_regressor, dataloader_for_calibration):
    # Model - we will train only the calibration layer
    calibrated_comment_regressor = CalibratedCommentRegressor(comment_regressor)
    calibrated_comment_regressor.comment_regressor.eval()
    calibrated_comment_regressor.calibration_layer.train()

    # Optimizer
    optimizer = torch.optim.SGD(calibrated_comment_regressor.calibration_layer.parameters(), lr=0.01)

    # Loss function
    loss_function = nn.MSELoss()

    for epoch in range(1):
        for step, batch in tqdm.tqdm(
                enumerate(dataloader_for_calibration),
                desc="Calibrate",
                total=len(dataloader_for_calibration),
                position=1,
        ):
            mean_list, stddev_list, cdf_list, target = batch
            optimizer.zero_grad()
            calib_mean, calib_std = calibrated_comment_regressor.calibration_layer(mean_list, stddev_list)
            cdf_pred = 0.5 * (1.0 + torch.erf((target - calib_mean) / calib_std / math.sqrt(2)))
            loss = loss_function(cdf_pred, target)
            loss.backward()
            optimizer.step()

            if step % int(len(dataloader_for_calibration)/10) == 0:
                print("Epoch: {}, Step: {}, Calibration MSE loss: {:.4f}".format(epoch, step, loss.item()))

    return calibrated_comment_regressor


# Will be used to train the calibration layer
class CalibrationLayerDataSet(Dataset):
    def __init__(self, mean_list, stddev_list, cdf_list, target):
        if isinstance(mean_list, np.ndarray):
            self.mean_list = torch.from_numpy(mean_list)
        else:
            self.mean_list = mean_list

        if isinstance(stddev_list, np.ndarray):
            self.stddev_list = torch.from_numpy(stddev_list)
        else:
            self.stddev_list = stddev_list

        if isinstance(cdf_list, np.ndarray):
            self.cdf_list = torch.from_numpy(cdf_list)
        else:
            self.cdf_list = cdf_list

        if isinstance(target, np.ndarray):
            self.target = torch.from_numpy(target)
        else:
            self.target = target

    def __len__(self):
        return len(self.mean_list)

    def __getitem__(self, idx):
        return self.mean_list[idx], self.stddev_list[idx], self.cdf_list[idx], self.target[idx]