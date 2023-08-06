import math

import accelerate
import datasets
import tqdm
import torch
import numpy as np
import typing

import transformers
#import evaluate
import wandb
from typing import List, Dict, Tuple, Optional, Union, Any

import data
from model import CommentRegressorPrediction
from metrics import MetricManager, compute_metrics

class CommentRegressorTrainer:

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: transformers.PreTrainedTokenizer,
            train_dataset: datasets.Dataset,
            validation_dataset: datasets.Dataset,
            train_batch_size: int,
            validation_batch_size: int,
            training_accumulation_steps: int,
            validation_accumulation_steps: int,
            coefficient: float,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LinearLR,
            calibration_dataset: datasets.Dataset = None,
            test_dataset: datasets.Dataset = None,
            test_batch_size: int = None,
            group_names: List[str] = None,
            accelerator: accelerate.Accelerator = None,
            seed: int = 42,
            logging_steps: int = 50,
            eval_steps: int = 200,
            save_steps: int = 1000,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train.
            tokenizer: The tokenizer to use.
            train_dataset: The training dataset.
            validation_dataset: The validation dataset.
            calibration_dataset: The calibration dataset.
            test_dataset: The test dataset.
            train_batch_size: The training batch size.
            validation_batch_size: The validation batch size.
            test_batch_size: The test batch size.
            training_accumulation_steps: The number of steps to accumulate gradients for during training.
            validation_accumulation_steps: The number of steps to accumulate gradients for during validation.
            coefficient: The coefficient to use for the loss function. Larger values will increase the importance
                of sharpness/accuracy while smaller values will increase the importance of fairness. Bounded between
                0 and 1.
            optimizer: The optimizer to use.
            scheduler: The scheduler to use.
            group_names: The names of the protected groups.
            accelerator: The accelerator to use.
            seed: The seed to use.
            logging_steps: The number of steps to take before logging.
            eval_steps: The number of steps to take before evaluating.
            save_steps: The number of steps to take before saving.
        """
        # Create metric manager to save and plot the metrics
        self.metric_manager = MetricManager(coefficient=coefficient)


        if accelerator is None:
            accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=training_accumulation_steps,
                #log_with='wandb',

            )
        self.accelerator = accelerator
        self.initial_learning_rate = optimizer.defaults.get('lr')

        self.tokenizer = tokenizer
        self.seed = seed

        self._set_up_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            calibration_dataset=calibration_dataset,
            test_dataset=test_dataset,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size,
            calibration_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.validation_loader,
            self.calibration_loader,
            self.test_loader,
         ) = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
            self.train_loader,
            self.validation_loader,
            self.calibration_loader,
            self.test_loader,
        )

        self.coefficient = coefficient
        self.scheduler = scheduler
        self.group_names = group_names
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.training_accumulation_steps = training_accumulation_steps
        self.validation_accumulation_steps = validation_accumulation_steps
        self.training_step = 0

    def _set_up_dataloaders(
            self,
            train_dataset: datasets.Dataset,
            validation_dataset: datasets.Dataset,
            train_batch_size: int,
            validation_batch_size: int,
            calibration_dataset: datasets.Dataset = None,
            test_dataset: datasets.Dataset = None,
            calibration_batch_size: int = None,
            test_batch_size: int = None,
    ):
        """Set up the dataloaders.

        Save the dataloaders as attributes of the Trainer.

        Args:
            train_dataset: The training dataset.
            validation_dataset: The validation dataset.
            test_dataset: The test dataset.
            train_batch_size: The training batch size.
            validation_batch_size: The validation batch size.
            test_batch_size: The test batch size.
        """
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
        if calibration_dataset is not None and calibration_batch_size is not None:
            self.calibration_loader = torch.utils.data.DataLoader(
                calibration_dataset,
                batch_size=calibration_batch_size,
                collate_fn=collator_fn,
            )
        if test_dataset is not None and test_batch_size is not None:
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

        Args:
            epochs: The number of epochs to train for.
        """
        self.accelerator.print(
            f"Training on {self.accelerator.device} using {self.accelerator.distributed_type} "
            f"with {self.accelerator.num_processes} processes."
        )
        self.accelerator.init_trackers(
            project_name='individualized_calibration_for_language',
            config={
                'epochs': epochs,
                'train_batch_size': self.train_loader.batch_size,
                'validation_batch_size': self.validation_loader.batch_size,
                'calibration_batch_size': self.calibration_loader.batch_size,
                'test_batch_size': self.test_loader.batch_size,
                'training_accumulation_steps': self.training_accumulation_steps,
                'validation_accumulation_steps': self.validation_accumulation_steps,
                'coefficient': self.coefficient,
                'seed': self.seed,
                'logging_steps': self.logging_steps,
                'eval_steps': self.eval_steps,
                'save_steps': self.save_steps,
                'groups': self.group_names,
            },
        )

        for epoch in tqdm.trange(
                epochs,
                desc="Epoch",
                total=epochs,
                disable=not self.accelerator.is_local_main_process,
                position=0,
        ):
            self.model.train()

            for step, batch in tqdm.tqdm(
                    enumerate(self.train_loader),
                    desc="Training",
                    total=len(self.train_loader),
                    disable=not self.accelerator.is_local_main_process,
                    position=1,
            ):
                self.training_step += 1

                with self.accelerator.accumulate(self.model):
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
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if step % self.logging_steps == 0:
                        self.accelerator.print(
                            f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.3f}"
                        )
                        learning_rate = self.optimizer.param_groups[0]["lr"]
                        self.accelerator.log(
                            {
                                "training_step": step,
                                "training_loss": loss.item(),
                                "learning_rate": learning_rate,
                            },
                            step=step,
                        )
                        self.metric_manager.add_metric(step=self.training_step, metric_name="training_loss",
                                                       metric_value=loss.item())
                        self.metric_manager.add_metric(step=self.training_step, metric_name="lr",
                                                       metric_value=learning_rate)

                    if step % self.eval_steps == 0:
                        val_metrics = self.eval(self.validation_loader, self.validation_accumulation_steps)
                        self.metric_manager.add_dict_metrics(step=self.training_step, metrics_dict=val_metrics)
                        self.metric_manager.create_all_metrics_plots()

                    if step % self.save_steps == 0:
                        experiment_name = (
                            f'seed_{self.seed}_'
                            f'coefficient_{str(self.coefficient).replace(".", "_")}_'
                            f'lr_{str(self.initial_learning_rate).replace(".", "_")}_'
                            f'step_{self.training_step-1}')
                        #self.accelerator.print(f"Saving model to `checkpoints/{experiment_name}`")
                        #self.accelerator.save_state(f"checkpoints/{experiment_name}")
        self.metric_manager.save_metrics()
        self.accelerator.end_training()

    def eval(
            self,
            validation_loader: torch.utils.data.DataLoader,
            validation_accumulation_steps: int,
            with_wandb: bool = False,
    ):
        """
        Evaluate the model on the validation set.

        Args:
            validation_loader: The validation dataloader.
            validation_accumulation_steps: The number of steps to accumulate gradients over.
        """
        self.model.eval()

        all_losses = []                                                   # type: List[torch.Tensor]
        losses = []                                                   # type: List[torch.Tensor]
        groups = {group_name: [] for group_name in self.group_names}  # type: Dict[str, List[torch.Tensor]]
        means = []                                                    # type: List[torch.Tensor]
        stddevs = []                                                  # type: List[torch.Tensor]
        labels = []                                                   # type: List[torch.Tensor]
        input_rs = []                                                 # type: List[torch.Tensor]

        for step, batch in tqdm.tqdm(
                enumerate(validation_loader),
                desc="Validation step",
                total=len(validation_loader),
                disable=not self.accelerator.is_local_main_process,
                position=2,
        ):

            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "input_r": batch["input_r"],
            }
            with torch.no_grad():
                model_outputs = self.model(**model_inputs)
            mean_pred, std_pred = model_outputs
            loss = self.compute_loss(
                input_r=batch["input_r"],
                mean_pred=mean_pred,
                std_pred=std_pred,
                labels=batch["labels"],
            )

            # Accumulate losses, groups, predictions, labels, and input_rs
            losses.append(loss)
            for group_name in self.group_names:
                groups[group_name].append(batch[group_name].cpu())
            means.append(mean_pred.cpu())
            stddevs.append(std_pred.cpu())
            labels.append(batch["labels"].cpu())
            input_rs.append(batch["input_r"].cpu())

            # If we have accumulated enough, then append to all_losses, all_groups, all_predictions, all_labels,
            # and all_input_rs
            if len(losses) % validation_accumulation_steps:
                loss = torch.mean(torch.stack(losses))
                all_losses.append(loss)
                losses = []

        loss = torch.mean(torch.stack(all_losses))
        if with_wandb:
            self.accelerator.print(f"Validation Loss: {loss.item():.3f}")
            self.accelerator.log({"validation_loss": loss.item()}, step=self.training_step-1)
        self.model.train()

        # Accumulate the predictions, labels, groups and input_r.
        mean_list = np.array(torch.cat(means))
        stddev_list = np.array(torch.cat(stddevs))
        input_r_list = np.array(torch.cat(input_rs))
        targets_list = np.array(torch.cat(labels))

        for group_name in self.group_names:
            groups[group_name] = np.array(torch.cat(groups[group_name]))

        comment_reg_pred = CommentRegressorPrediction(means=mean_list,
                                                      stddevs=stddev_list,
                                                      input_r=input_r_list,
                                                      label_ids=targets_list,
                                                      groups=groups)

        # Creates the metrics
        metric_eval = compute_metrics(eval_pred=comment_reg_pred, coefficient=self.coefficient)
        if with_wandb:
            self.accelerator.log(metric_eval, step=self.training_step - 1)
        return metric_eval

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
            The loss scalar.
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