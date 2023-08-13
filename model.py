import typing

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel


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
            input_r_dim: int = 1,
            dtype=torch.float32,
    ):
        super(CommentRegressor, self).__init__()
        self.input_r_dim = input_r_dim
        self.dtype = dtype
        self.mlp_hidden = mlp_hidden
        self.drop_prob = drop_prob

        # Extract the text encoder from the pretrained model for text classification.
        # TODO: Change verbosity so that loading BERT does not print a wall of text.
        # TODO: Change BERT model to BertModel as opposed to BertForSequenceClassification to avoid removing head.
        self.bert = BertForSequenceClassification.from_pretrained(
            text_encoder_model_name,
        ).bert

        # Construct the MLP for predicting the mean and standard deviation following the text encoder.
        self.fc1 = nn.Linear(self.bert.config.hidden_size + self.input_r_dim, self.mlp_hidden, dtype=dtype)
        self.fc2 = nn.Linear(self.mlp_hidden + self.input_r_dim, self.mlp_hidden, dtype=dtype)
        self.drop = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(self.mlp_hidden, 2, dtype=dtype)

    def forward(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size]
            attention_mask: torch.Tensor,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor,  # Tensor of random values of shape [batch_size, 1]
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = output.to(self.dtype)
        input_r = input_r.to(self.dtype)

        h = torch.cat([output, input_r.repeat(1, self.input_r_dim)], dim=1)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = torch.cat([h, input_r.repeat(1, self.input_r_dim)], dim=1)
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

    def __init__(self, dtype=torch.float32):
        super(CalibrationLayer, self).__init__()
        self.fc1 = nn.Linear(2, 16, dtype=dtype)
        self.fc2 = nn.Linear(16, 2, dtype=dtype)

    def forward(
            self,
            mean: torch.Tensor,
            std: torch.Tensor,
    ):
        h = torch.stack([mean, std], dim=1)
        out = self.fc2(F.relu(self.fc1(h)))
        mean = out[:, 0]
        stddev = torch.sigmoid(out[:, 1]) * 5.0 + 0.01

        return mean, stddev


class CalibratedCommentRegressor(nn.Module):
    """
    A model for predicting the mean and standard deviation of a Gaussian distribution, and on top a calibration layer.

    The calibration layer is a simple MLP. The output units are the mean and standard deviation of the Gaussian.
    """

    def __init__(
            self,
            comment_regressor: CommentRegressor):
        """Initializes the model.

        Add a calibration layer on top of the CommentRegressor model. This calibration layer is a simple MLP, and
        performs "average calibration."

        Args:
            comment_regressor: The CommentRegressor model to use for predicting the mean and standard deviation.
        """
        super(CalibratedCommentRegressor, self).__init__()

        self.comment_regressor = comment_regressor
        self.calibration_layer = CalibrationLayer(dtype=comment_regressor.dtype)

    def forward(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size]
            attention_mask: torch.Tensor = None,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor = None,  # Tensor of random values of shape [batch_size, 1]
    ):
        """Predicts the mean and standard deviation of a Gaussian distribution for each sample in the batch.

        Args:
            input_ids: Tensor of input token ids of shape [batch_size, max_seq_len]. The input IDs are valid entries
                in the vocabulary of the text encoder.
            attention_mask: Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: Tensor of random values of shape [batch_size, 1]
        """
        mean, stddev = self.comment_regressor(input_ids, attention_mask, input_r)
        return self.calibration_layer(mean, stddev)
