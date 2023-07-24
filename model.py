import typing

import numpy as np
from sklearn.isotonic import IsotonicRegression
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertForSequenceClassification


class CommentRegressorPrediction(typing.NamedTuple):
    """
    Prediction output for a CommentRegressor.
    """
    means: np.ndarray       # The predicted means of the Gaussian distribution. Of shape [batch_size, 1]
    stddevs: np.ndarray     # The predicted standard deviations of the Gaussian distribution. Of shape [batch_size, 1]
    label_ids: np.ndarray   # The true labels. Of shape [batch_size, 1]
    input_r: np.ndarray     # The random values used as input to the model. Of shape [batch_size, 1]

    # The group information for each sample. Of shape [batch_size, 1]. The string key is the group name.
    groups: typing.Dict[str, np.ndarray]


class CommentRegressor(nn.Module):
    """
    A model for predicting the mean and standard deviation of a Gaussian distribution
    """

    def __init__(
            self,
            mlp_hidden: int = 100,
            drop_prob: float = 0.5,
            text_encoder_model_name: str = "textattack/bert-base-uncased-yelp-polarity",
            dtype=torch.float32,
    ):
        super(CommentRegressor, self).__init__()
        self.dtype = dtype
        self.mlp_hidden = mlp_hidden
        self.drop_prob = drop_prob

        # Extract the text encoder from the pretrained model for text classification.
        # TODO: Change verbosity so that loading BERT does not print a wall of text.
        self.bert = BertForSequenceClassification.from_pretrained(
            text_encoder_model_name,
        ).bert

        # Construct the MLP for predicting the mean and standard deviation following the text encoder.
        self.fc1 = nn.Linear(self.bert.config.hidden_size + 1, self.mlp_hidden, dtype=dtype)
        self.fc2 = nn.Linear(self.mlp_hidden + 1, self.mlp_hidden, dtype=dtype)
        self.drop = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(self.mlp_hidden, 2, dtype=dtype)

    def forward(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size]
            attention_mask: torch.Tensor = None,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor = None,  # Tensor of random values of shape [batch_size, 1]
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = output.to(self.dtype)
        input_r = input_r.to(self.dtype)

        h = torch.cat([output, input_r], dim=1)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = torch.cat([h, input_r], dim=1)
        h = self.fc2(h)
        h = F.leaky_relu(h)
        h = self.drop(h)
        h = self.fc3(h)
        mean = h[:, 0]
        stddev = torch.sigmoid(h[:, 1]) * 5.0 + 0.01

        return mean, stddev

class CalibrationLayer(nn.Module):
    """
    Calibration layer for a CommentRegressor.
    """

    def __init__(self):
        super(CalibrationLayer, self).__init__()

        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([0]))
        self.c = nn.Parameter(torch.Tensor([0]))
        self.d = nn.Parameter(torch.Tensor([1]))

    def forward(
            self,
            mean: torch.Tensor,
            std: torch.Tensor,
    ):
        calibrated_mean = self.a * mean + self.b * std
        calibrated_std = self.c * mean + self.d * std

        return calibrated_mean, torch.abs(calibrated_std)


class CalibratedCommentRegressor(nn.Module):
    """
    A trained model for predicting the mean and standard deviation of a Gaussian distribution, and on top a calibration layer.
    """

    def __init__(
            self,
            comment_regressor: CommentRegressor):
        super(CalibratedCommentRegressor, self).__init__()

        self.comment_regressor = comment_regressor
        self.calibration_layer = CalibrationLayer()

    def forward(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size]
            attention_mask: torch.Tensor = None,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor = None,  # Tensor of random values of shape [batch_size, 1]
        ):
        mean, stddev = self.comment_regressor(input_ids, attention_mask, input_r)
        return self.calibration_layer(mean, stddev)

