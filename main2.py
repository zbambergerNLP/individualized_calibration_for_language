import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import AutoTokenizer
import pandas as pd
import argparse

from backbone import CommentRegressor
from trainer import Trainer


# The dataset is "Jigsaw Unintended Bias in Toxicity Classification"
# from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv
class CommentDataset(Dataset):
    def __init__(self, comments, targets, tokenizer, max_len):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

def main(args):
    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

    train_dataset = CommentDataset(
        comments=df_train.comment_text.to_numpy(),
        targets=df_train.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=args.max_len
    )

    test_dataset = CommentDataset(
        comments=df_test.comment_text.to_numpy(),
        targets=df_test.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=args.max_len
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommentRegressor(mlp_hidden=args.mlp_hidden, drop_prob=args.drop_prob).to(device)
    print("device", device)

    # Training the model
    trainer = Trainer(model, train_data_loader, test_data_loader, device, args)
    trainer.train_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data hyperparameters
    parser.add_argument('--train_file', type=str, default='../data/train_data.csv', help='Path to training data file')
    parser.add_argument('--test_file', type=str, default='../data/test_data.csv', help='Path to testing data file')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of comments')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--iteration_eval', type=int, default=100, help='Number of evaluations to do in each epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Save the best model according to the validation loss')
    parser.add_argument('--coeff', type=float, default=1.0, help='When coeff is 0 completely fairness loss,'
                                                                  ' when 1 completely NLL loss')

    # Model hyperparameters
    parser.add_argument('--mlp_hidden', type=int, default=100, help='The width of the regression head')
    parser.add_argument('--drop_prob', type=float, default=0.05, help='The dropout probability for the regression head')

    args = parser.parse_args()

    main(args)

