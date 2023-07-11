import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import BertModel, BertTokenizer

class CommentRegressor(nn.Module):
    def __init__(self, mlp_hidden=100, drop_prob=0.5):
        super(CommentRegressor, self).__init__()
        self.mlp_hidden = mlp_hidden
        self.drop_prob = drop_prob

        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity").bert

        self.fc1 = nn.Linear(self.bert.config.hidden_size + 1, self.mlp_hidden)
        self.fc2 = nn.Linear(self.mlp_hidden + 1, self.mlp_hidden)
        self.drop = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(self.mlp_hidden, 2)


    def forward(self, input_ids, attention_mask, input_r=None):
        if input_r is None:
            input_r = torch.rand(input_ids.shape[0], 1, device=input_ids.device)

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = torch.cat([output.pooler_output, input_r], dim=1)
        h = F.leaky_relu(self.fc1(h))
        h = torch.cat([h, input_r], dim=1)
        h = F.leaky_relu(self.fc2(h))
        h = self.drop(h)
        h = self.fc3(h)
        mean = h[:, 0]
        stddev = torch.sigmoid(h[:, 1]) * 5.0 + 0.01

        return mean, stddev


    # Taken from Individual Calibration with Randomized Forecasting, Zhao et al. 2020
    # https://github.com/ShengjiaZhao/Individual-Calibration/tree/master
    def eval_all(self, input_ids, attention_mask, target):
        input_r = torch.rand(input_ids.shape[0], 1, device=input_ids.device)
        mean, stddev = self.forward(input_ids, attention_mask, input_r)
        cdf = 0.5 * (1.0 + torch.erf((target - mean) / stddev / math.sqrt(2)))

        loss_cdf = torch.abs(cdf - input_r).mean()
        loss_stddev = stddev.mean()

        # Log likelihood of by under the predicted Gaussian distribution
        loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((target - mean) / stddev) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        return cdf, loss_cdf, loss_stddev, loss_nll