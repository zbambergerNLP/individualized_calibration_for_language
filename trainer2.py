import math

import accelerate
import datasets
import tqdm
import torch
import typing

import transformers

import data


class CommentRegressorTrainer:

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: transformers.PreTrainedTokenizer,
            train_dataset: datasets.Dataset,
            validation_dataset: datasets.Dataset,
            test_dataset: datasets.Dataset,
            train_batch_size: int,
            validation_batch_size: int,
            test_batch_size: int,
            training_accumulation_steps: int,
            validation_accumulation_steps: int,
            coefficient: float,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LinearLR,
            group_names: typing.List[str] = None,
            accelerator: accelerate.Accelerator = None,
            seed: int = 42,
            logging_steps: int = 50,
            eval_steps: int = 200,
            save_steps: int = 1000,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train.
            coefficient: The coefficient to use.
            optimizer: The optimizer to use.
            scheduler: The scheduler to use.
            group_names: The names of the hidden groups (columns) in the dataset.
            accelerator: The accelerator to use.
        """
        if accelerator is None:
            accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=training_accumulation_steps,

            )
        self.accelerator = accelerator

        self.tokenizer = tokenizer
        self.seed = seed

        self.set_up_dataloaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            train_batch_size,
            validation_batch_size,
            test_batch_size,
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.validation_loader,
         ) = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
            self.train_loader,
            self.validation_loader,
        )

        self.coefficient = coefficient
        self.scheduler = scheduler
        self.group_names = group_names
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.training_accumulation_steps = training_accumulation_steps
        self.validation_accumulation_steps = validation_accumulation_steps

    def set_up_dataloaders(
            self,
            train_dataset: datasets.Dataset,
            validation_dataset: datasets.Dataset,
            test_dataset: datasets.Dataset,
            train_batch_size: int,
            validation_batch_size: int,
            test_batch_size: int,
    ):
        """Set up the dataloaders."""
        collator_fn = data.CommentRegressorDataCollator(
            tokenizer=self.tokenizer,
            seed=self.seed,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=collator_fn,
        )
        self.validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            collate_fn=collator_fn,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            collate_fn=collator_fn,
        )

    def train(
            self,
            epochs: int,
    ):
        """Train the model.

        Note:
            * Each batch in the dataset consists of the following keys:
                - input_ids: The input IDs.
                - attention_mask: The attention mask.
                - labels: The labels. In the case of CivilCommentsIdentities, this is a toxicity score between 0 and 1.
                - input_r: A random number between 0 and 1.
                - <group_name>: The hidden group (column) values. These are also between 0 and 1.
            * The model predicts outputs that consist of:
                - mean: The mean of the predictions.
                - std: The standard deviation of the predictions.

        Args:
            train_dataset: The training dataset.
            validation_dataset: The validation dataset.
            epochs: The number of epochs to train for.
            train_batch_size: The batch size for training.
            validation_batch_size: The batch size for validation.
            training_accumulation_steps: The number of steps to accumulate gradients for during training.
            validation_accumulation_steps: The number of steps to accumulate gradients for during validation.
        """
        self.accelerator.print(
            f"Training on {self.accelerator.device} using {self.accelerator.distributed_type} "
            f"with {self.accelerator.num_processes} processes."
        )
        for epoch in tqdm.trange(
                epochs,
        ):
            self.model.train()

            for step, batch in tqdm.tqdm(enumerate(self.train_loader)):
                # Assign the batch to the device.
                batch = {key: value for key, value in batch.items()}
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "input_r": batch["input_r"],
                }
                model_outputs = self.model(**model_inputs)
                mean_pred, output_pred = model_outputs
                loss = self.compute_loss(
                    input_r=batch["input_r"],
                    mean_pred=mean_pred,
                    std_pred=output_pred,
                    labels=batch["labels"],
                )
                self.accelerator.backward(loss)

                if step % self.training_accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                if step % self.eval_steps == 0:
                    self.eval(self.validation_loader, self.validation_accumulation_steps)
                if step % self.save_steps == 0:
                    self.accelerator.save(self.model.state_dict(), f"model_{step}.pt")

    def eval(
            self,
            validation_loader: torch.utils.data.DataLoader,
            validation_accumulation_steps: int,
    ):
        self.model.eval()
        losses = []
        batch_count = 0
        for step, batch in tqdm.tqdm(enumerate(validation_loader), desc="Validation Batch"):
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "input_r": batch["input_r"],
            }
            with torch.no_grad():
                model_outputs = self.model(**model_inputs)
            mean_pred, output_pred = model_outputs
            loss = self.compute_loss(
                input_r=batch["input_r"],
                mean_pred=mean_pred,
                std_pred=output_pred,
                labels=batch["labels"],
            )
            losses.append(loss)
            batch_count += 1
            if len(losses) == validation_accumulation_steps:
                loss = torch.mean(torch.stack(losses))
                # self.accelerator.log({"eval_loss": loss.item()})
                break
        self.model.train()

    def compute_loss(
            self,
            input_r: torch.Tensor,
            mean_pred: torch.Tensor,
            std_pred: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_r: The random input.
            mean_pred: The mean predictions.
            std_pred: The standard deviation predictions.
            labels: The labels.

        Returns:
            The loss.
        """
        cdf = 0.5 * (1.0 + torch.erf((labels - mean_pred) / std_pred / torch.sqrt(torch.tensor(2))))
        loss_cdf = torch.abs(cdf - input_r).mean()

        # Log likelihood under the predicted Gaussian distribution
        # TODO: Describe the nll loss function.
        loss_nll = torch.log(std_pred) + math.log(2 * math.pi) / 2.0 + (((labels - mean_pred) / std_pred) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        # Total loss is a function of both the cdf loss (fairness) and the nll loss (sharpness/accuracy).
        loss = (1 - self.coefficient) * loss_cdf + self.coefficient * loss_nll
        return loss


