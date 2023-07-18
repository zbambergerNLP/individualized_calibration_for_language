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
    means: np.ndarray
    stddevs: np.ndarray
    label_ids: np.ndarray
    inputs: typing.Optional[typing.Dict[str, typing.Any]]


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
        self.bert = BertForSequenceClassification.from_pretrained(text_encoder_model_name).bert

        # Construct the MLP for predicting the mean and standard deviation following the text encoder.
        # TODO: Consider running experiments with different model architectures following the text classifier.
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

    def recalibrate(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size],
            attention_mask: torch.Tensor = None,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor = None,  # Tensor of random values of shape [batch_size, 1]
            label_ids: torch.Tensor = None,  # Tensor of label ids of shape [batch_size]
    ):
        with torch.no_grad():
            outputs = self.model.forward(input_ids, attention_mask, input_r)
            mean, stddev = outputs
            cdf = (0.5 * (1.0 + torch.erf((label_ids - mean) / stddev / math.sqrt(2)))).cpu().numpy()[:, 0].astype(np.float)

        cdf = np.sort(cdf)
        lin = np.linspace(0, 1, int(cdf.shape[0]))

        # Insert an extra 0 and 1 to ensure the range is always [0, 1], and trim CDF for numerical stability
        cdf = np.clip(cdf, a_max=1.0 - 1e-6, a_min=1e-6)
        cdf = np.insert(np.insert(cdf, -1, 1), 0, 0)
        lin = np.insert(np.insert(lin, -1, 1), 0, 0)

        iso_transform = IsotonicRegression()
        iso_transform.fit_transform(cdf, lin)
        self.iso_transform = iso_transform

    def apply_recalibrate(self, cdf):
        if self.iso_transform is not None:
            original_shape = cdf.shape
            return np.reshape(self.iso_transform.transform(cdf.flatten()), original_shape)
        else:
            return cdf
