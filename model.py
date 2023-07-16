import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertForSequenceClassification


class CommentRegressor(nn.Module):
    """
    A model for predicting the mean and standard deviation of a Gaussian distribution
    """

    def __init__(
            self,
            mlp_hidden: int = 100,
            drop_prob: float = 0.5,
            text_encoder_model_name: str = "textattack/bert-base-uncased-yelp-polarity",
    ):
        super(CommentRegressor, self).__init__()
        self.mlp_hidden = mlp_hidden
        self.drop_prob = drop_prob

        # Extract the text encoder from the pretrained model for text classification.
        self.bert = BertForSequenceClassification.from_pretrained(text_encoder_model_name).bert

        # Construct the MLP for predicting the mean and standard deviation following the text encoder.
        # TODO: Consider running experiments with different model architectures following the text classifier.
        self.fc1 = nn.Linear(self.bert.config.hidden_size + 1, self.mlp_hidden)
        self.fc2 = nn.Linear(self.mlp_hidden + 1, self.mlp_hidden)
        self.drop = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(self.mlp_hidden, 2)

    def forward(
            self,
            input_ids: torch.Tensor,  # Tensor of input token ids of shape [batch_size, max_seq_len, vocab_size]
            attention_mask: torch.Tensor = None,  # Tensor of attention masks of shape [batch_size, max_seq_len]
            input_r: torch.Tensor = None,  # Tensor of random values of shape [batch_size, 1]
    ):
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
