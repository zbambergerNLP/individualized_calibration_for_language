import torch
from torch import nn, optim

import os
from pathlib import Path
from datetime import datetime


# Used to train and evaluate the model.
class Trainer():
    def __init__(self, model, train_data_loader, test_data_loader, device, args):
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.args = args

        self.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Lists to store metrics
        self.loss_list, self.loss_cdf_list, self.loss_stddev_list, self.loss_nll_list\
            = [], [], [], []
        self.loss_list_test, self.loss_cdf_list_test, self.loss_stddev_list_test,\
            self.loss_nll_list_test = [], [], [], []

        # Early stopping
        self.best_model = None
        self.best_loss = 1e10

        # Current time
        now = datetime.now()
        self.date_time = now.strftime("%Y_%m_%d")
        self.hour_time = now.strftime("%H_%M")

    def train_loop(self):
        for epoch in range(self.args.epochs):
            loss_list_tpm, loss_cdf_list_tpm, loss_stddev_list_tpm, loss_nll_list_tpm \
                = [], [], [], []
            self.model = self.model.train()

            for i, d in enumerate(self.train_data_loader):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                cdf, loss_cdf, loss_stddev, loss_nll = self.model.eval_all(input_ids=input_ids,
                                                                           attention_mask=attention_mask,
                                                                           target=targets)
                # Current train iteration metrics
                loss_cdf_list_tpm.append(loss_cdf.item())
                loss_stddev_list_tpm.append(loss_stddev.item())
                loss_nll_list_tpm.append(loss_nll.item())

                loss = (1 - self.args.coeff) * loss_cdf + self.args.coeff * loss_nll
                loss_list_tpm.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % int(len(self.train_data_loader) / self.args.iteration_eval) == 0:
                    # Save train metrics so far
                    self.loss_cdf_list.append(sum(loss_cdf_list_tpm) / len(loss_cdf_list_tpm))
                    self.loss_stddev_list.append(sum(loss_stddev_list_tpm) / len(loss_stddev_list_tpm))
                    self.loss_nll_list.append(sum(loss_nll_list_tpm) / len(loss_nll_list_tpm))
                    self.loss_list.append(sum(loss_list_tpm) / len(loss_list_tpm))

                    loss_list_tpm, loss_cdf_list_tpm, loss_stddev_list_tpm, loss_nll_list_tpm \
                        = [], [], [], []
                    print("enter eval")
                    self.eval_iter()
                    print("{}\{} | iter {} | Train Loss: {:.3f} | Test Loss: {:.3f}".
                          format(epoch, self.args.epochs, i, self.loss_list[-1], self.loss_list_test[-1]))
                    self.model = self.model.train()

                    if self.args.early_stopping:
                        # Saves the best model so far, according to the test loss
                        self.perform_early_stopping()
                    else:
                        self.best_model = self.model

        print("Done Training")
        self.saves_experiment()


    def eval_iter(self):
        self.model = self.model.eval()
        with torch.no_grad():
            for d in self.test_data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                cdf, loss_cdf, loss_stddev, loss_nll = self.model.eval_all(input_ids=input_ids,
                                                                           attention_mask=attention_mask,
                                                                           target=targets)

                self.loss_cdf_list_test.append(loss_cdf.item())
                self.loss_stddev_list_test.append(loss_stddev.item())
                self.loss_nll_list_test.append(loss_nll.item())

                loss = (1 - self.args.coeff) * loss_cdf + self.args.coeff * loss_nll
                self.loss_list_test.append(loss.item())


    # Saves the best model so far, according to the test loss
    def perform_early_stopping(self):
        if self.loss_list_test[-1] < self.best_loss:
            self.best_loss = self.loss_list_test[-1]
            self.best_model = self.model
            print("New best model saved")

    def saves_experiment(self):
        save_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', "{}-{}".format(self.date_time,self.hour_time))
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Saving descriptor of the experiment
        path_to_save_text = os.path.join(save_path, "descriptor.txt")
        file = open(path_to_save_text, "w")  # saving the dataset description
        file.write(self.__str__())

        # Saving the best model
        path_to_save_model = os.path.join(save_path, "merged_model.pt")
        torch.save(self.best_model.state_dict(), path_to_save_model)


    def __str__(self):
        text = "{}-{} \n".format(self.date_time, self.hour_time)
        for i, elem in enumerate(vars(self.args).items()):
            field_name, value = elem
            text += f'{field_name}: {value} '
            if (i + 1) % 4 == 0:
                text += "\n"
        return text
