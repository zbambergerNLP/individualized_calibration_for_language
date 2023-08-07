import copy
import math

import accelerate
import datasets
import tqdm
import torch
import numpy as np
import typing
import os
import transformers
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import denumpify_detensorize

import wandb
from typing import List, Dict, Tuple, Optional, Union, Any

import data
from model import CommentRegressorPrediction
from metrics import MetricManager, compute_metrics


class CommentRegressorTrainer:

    def __init__(
            self,
            output_dir: str,
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
            patience: int = 3,
    ):
        """Initialize the trainer.

        Args:
            output_dir: The directory to save the model and results to.
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
            patience: The number of steps to wait before early stopping.
        """
        self.output_dir = output_dir
        if accelerator is None:
            accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=training_accumulation_steps,
                log_with='wandb',

            )
        self.accelerator = accelerator
        # Create metric manager to save and plot the metrics
        # self.metric_manager = MetricManager(coefficient=coefficient, accelerator=accelerator)
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
        self.training_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        # Early stopping
        self.patience = patience
        self.patience_counter = 0
        self.best_model_checkpoint = None
        self.best_model_loss = math.inf
        self.best_checkpoint_path = os.path.join(self.output_dir, 'best_model')
        os.makedirs(self.best_checkpoint_path, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.best_checkpoint_path, 'best_model.pt')

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
                    cdf_loss, nll_loss, total_loss = self.compute_loss(
                        input_r=batch["input_r"],
                        mean_pred=mean_pred,
                        std_pred=output_pred,
                        labels=batch["labels"],
                    )
                    self.accelerator.backward(total_loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if step % self.logging_steps == 0:
                        learning_rate = self.optimizer.param_groups[0]["lr"]
                        self.accelerator.print(
                            f"Epoch: {epoch} | "
                            f"Step: {self.training_step-1} | "
                            f"Loss: {total_loss.item():.3f} | "
                            f"CDF Loss: {cdf_loss.item():.3f} | "
                            f"NLL Loss: {nll_loss.item():.3f} | "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                        )
                        self.accelerator.log(
                            {
                                "training_step": step,
                                "training_loss": total_loss.item(),
                                "learning_rate": learning_rate,
                                "training_cdf_loss": cdf_loss.item(),
                                "training_nll_loss": nll_loss.item(),
                            },
                            step=step,
                        )
                        # self.metric_manager.add_metric(step=self.training_step, metric_name="training_loss",
                        #                                metric_value=total_loss.item())
                        # self.metric_manager.add_metric(step=self.training_step, metric_name="training_cdf_loss",
                        #                                metric_value=cdf_loss.item())
                        # self.metric_manager.add_metric(step=self.training_step, metric_name="training_nll_loss",
                        #                                metric_value=nll_loss.item())
                        # self.metric_manager.add_metric(step=self.training_step, metric_name="lr",
                        #                                metric_value=learning_rate)

                    if step % self.eval_steps == 0:
                        val_metrics = self.eval(self.validation_loader, self.validation_accumulation_steps)
                        self.accelerator.print(f"Validation metrics: {val_metrics}")
                        self.accelerator.log(val_metrics, step=step)
                        # self.metric_manager.add_dict_metrics(step=self.training_step, metrics_dict=val_metrics)
                        # self.metric_manager.create_all_metrics_plots()
                        stop_early = self._perform_early_stopping(val_metrics["eval_loss"])
                        if stop_early:
                            # self.metric_manager.save_metrics()
                            self.accelerator.end_training()
                            return

                    if step % self.save_steps == 0:
                        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{self.training_step-1}")

                        if not os.path.exists(checkpoint_dir) and self.accelerator.is_local_main_process:
                            os.makedirs(checkpoint_dir)
                        self.accelerator.print(f"Saving model to `{checkpoint_dir}`")
                        self.accelerator.save_state(checkpoint_dir)

        # self.metric_manager.save_metrics()
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
            with_wandb: Whether to log the results to wandb.
        """
        self.model.eval()

        all_total_losses = None                                     # type: Union[np.ndarray, None]
        all_cdf_losses = None                                       # type: Union[np.ndarray, None]
        all_nll_losses = None                                       # type: Union[np.ndarray, None]
        all_groups = {}                                             # type: Dict[str, np.ndarray]
        all_means = None                                            # type: Union[np.ndarray, None]
        all_stddevs = None                                          # type: Union[np.ndarray, None]
        all_labels = None                                           # type: Union[np.ndarray, None]
        all_input_rs = None                                         # type: Union[np.ndarray, None]

        total_losses_host = None                                    # type: Union[List[torch.Tensor], None]
        cdf_losses_host = None                                      # type: Union[List[torch.Tensor], None]
        nll_losses_host = None                                      # type: Union[List[torch.Tensor], None]
        groups_host = {}                                            # type: Dict[str, Union[List[torch.Tensor]]]
        means_host = None                                           # type: Union[List[torch.Tensor], None]
        stddevs_host = None                                         # type: Union[List[torch.Tensor], None]
        labels_host = None                                          # type: Union[List[torch.Tensor], None]
        input_rs_host = None                                        # type: Union[List[torch.Tensor], None]

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
            groups = {}
            for group_name in self.group_names:
                groups[group_name] = batch.pop(group_name)

            with torch.no_grad():
                model_outputs = self.model(**model_inputs)
            mean_pred, std_pred = model_outputs
            cdf_loss, nll_loss, total_loss = self.compute_loss(
                input_r=batch["input_r"],
                mean_pred=mean_pred,
                std_pred=std_pred,
                labels=batch["labels"],
            )

            if total_loss is not None:
                total_losses = self.accelerator.gather_for_metrics(total_loss.repeat(self.validation_batch_size))
                total_losses_host = (
                    total_losses if total_losses_host is None
                    else nested_concat(total_losses_host, total_losses)
                )

            if cdf_loss is not None:
                cdf_losses = self.accelerator.gather_for_metrics(cdf_loss.repeat(self.validation_batch_size))
                cdf_losses_host = cdf_losses if cdf_losses_host is None else nested_concat(cdf_losses_host, cdf_losses)

            if nll_loss is not None:
                nll_losses = self.accelerator.gather_for_metrics(nll_loss.repeat(self.validation_batch_size))
                nll_losses_host = nll_losses if nll_losses_host is None else nested_concat(nll_losses_host, nll_losses)

            if mean_pred is not None:
                mean_preds = self.accelerator.gather_for_metrics(mean_pred)
                means_host = mean_preds if means_host is None else nested_concat(means_host, mean_preds)

            if std_pred is not None:
                std_preds = self.accelerator.gather_for_metrics(std_pred)
                stddevs_host = std_preds if stddevs_host is None else nested_concat(stddevs_host, std_preds)

            if batch["labels"] is not None:
                labels = self.accelerator.gather_for_metrics(batch["labels"])
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels)

            if batch["input_r"] is not None:
                input_rs = self.accelerator.gather_for_metrics(batch["input_r"])
                input_rs_host = input_rs if input_rs_host is None else nested_concat(input_rs_host, input_rs)

            if groups != {}:
                for group in groups:
                    groups[group] = self.accelerator.gather_for_metrics((groups[group]))
                    if group not in groups_host:
                        groups_host[group] = groups[group]
                    else:
                        groups_host[group] = nested_concat(groups_host[group], groups[group])

            if step % validation_accumulation_steps == 0 and self.accelerator.sync_gradients:
                if total_losses_host is not None:
                    total_losses = nested_numpify(total_losses_host)
                    all_losses = (
                        total_losses if all_total_losses is None
                        else np.concatenate((all_total_losses, total_losses), axis=0)
                    )

                if cdf_losses_host is not None:
                    cdf_losses = nested_numpify(cdf_losses_host)
                    all_cdf_losses = (
                        cdf_losses if all_cdf_losses is None
                        else np.concatenate((all_cdf_losses, cdf_losses), axis=0)
                    )

                if nll_losses_host is not None:
                    nll_losses = nested_numpify(nll_losses_host)
                    all_nll_losses = (
                        nll_losses if all_nll_losses is None
                        else np.concatenate((all_nll_losses, nll_losses), axis=0)
                    )

                if means_host is not None:
                    means = nested_numpify(means_host)
                    all_means = means if all_means is None else np.concatenate((all_means, means), axis=0)

                if stddevs_host is not None:
                    stddevs = nested_numpify(stddevs_host)
                    all_stddevs = stddevs if all_stddevs is None else np.concatenate((all_stddevs, stddevs), axis=0)

                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)

                if input_rs_host is not None:
                    input_rs = nested_numpify(input_rs_host)
                    all_input_rs = input_rs if all_input_rs is None else np.concatenate((all_input_rs, input_rs), axis=0)

                if groups_host is not None:
                    for group_name in groups_host:
                        group_value = nested_numpify(groups_host[group_name])
                        all_groups[group_name] = (
                            group_value if group_name not in all_groups else nested_concat(
                                all_groups[group_name],
                                group_value,
                            )
                        )

                total_losses_host = None
                cdf_losses_host = None
                nll_losses_host = None
                means_host = None
                stddevs_host = None
                labels_host = None
                input_rs_host = None
                groups_host = {}

        if total_losses_host is not None:
            total_losses = nested_numpify(total_losses_host)
            all_total_losses = (
                total_losses if all_total_losses is None
                else np.concatenate(
                    (all_total_losses, total_losses),
                    axis=0)
            )

        if cdf_losses_host is not None:
            cdf_losses = nested_numpify(cdf_losses_host)
            all_cdf_losses = cdf_losses if all_cdf_losses is None else np.concatenate((all_cdf_losses, cdf_losses), axis=0)

        if nll_losses_host is not None:
            nll_losses = nested_numpify(nll_losses_host)
            all_nll_losses = nll_losses if all_nll_losses is None else np.concatenate((all_nll_losses, nll_losses), axis=0)

        if means_host is not None:
            means = nested_numpify(means_host)
            all_means = means if all_means is None else np.concatenate((all_means, means), axis=0)

        if stddevs_host is not None:
            stddevs = nested_numpify(stddevs_host)
            all_stddevs = stddevs if all_stddevs is None else np.concatenate((all_stddevs, stddevs), axis=0)

        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)

        if input_rs_host is not None:
            input_rs = nested_numpify(input_rs_host)
            all_input_rs = input_rs if all_input_rs is None else np.concatenate((all_input_rs, input_rs), axis=0)

        if groups_host is not None:
            for group_name in groups_host:
                group_value = nested_numpify(groups_host[group_name])
                all_groups[group_name] = (
                    group_value if group_name not in all_groups else nested_concat(
                        all_groups[group_name],
                        group_value,
                    )
                )
        comment_reg_pred = CommentRegressorPrediction(
            means=all_means,
            stddevs=all_stddevs,
            input_r=all_input_rs,
            label_ids=all_labels,
            groups=all_groups)

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"{self.training_step-1} - Validation\n:"
                f"\tTotal loss: {np.mean(all_total_losses):.4f}\n"
                f"\tCDF loss: {np.mean(all_cdf_losses):.4f}\n"
                f"\tNLL loss: {np.mean(all_nll_losses):.4f}"
            )

        metric_eval = {}
        # Creates the metrics
        if (
                all_means is not None and
                all_stddevs is not None and
                all_labels is not None
        ):
            metric_eval = compute_metrics(
                eval_pred=comment_reg_pred,
                coefficient=self.coefficient,
                prefix="eval",
            )
            self.accelerator.print(metric_eval)
            metric_eval["step"] = self.training_step - 1
            metric_eval["eval_loss"] = all_total_losses.mean().item()
            metric_eval["eval_cdf_loss"] = all_cdf_losses.mean().item()
            metric_eval["eval_nll_loss"] = all_nll_losses.mean().item()
            self.accelerator.log(metric_eval, step=self.training_step - 1)

        metrics = denumpify_detensorize(metric_eval)
        if all_total_losses is not None:
            metrics[f"eval_loss"] = all_total_losses.mean().item()
        if all_cdf_losses is not None:
            metrics[f"eval_cdf_loss"] = all_cdf_losses.mean().item()
        if all_nll_losses is not None:
            metrics[f"eval_nll_loss"] = all_nll_losses.mean().item()

        return metric_eval

    def compute_loss(
            self,
            input_r: torch.Tensor,
            mean_pred: torch.Tensor,
            std_pred: torch.Tensor,
            labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss.

        Args:
            input_r: The random input.
            mean_pred: The mean predictions.
            std_pred: The standard deviation predictions.
            labels: The labels.

        Returns:
            A three-tuple consisting of the cdf loss, the nll loss, and the total loss.
        """
        cdf = 0.5 * (1.0 + torch.erf((labels - mean_pred) / std_pred / torch.sqrt(torch.tensor(2))))
        loss_cdf = torch.abs(cdf - input_r).mean()

        # Log likelihood under the predicted Gaussian distribution
        # TODO: Describe the nll loss function.
        loss_nll = torch.log(std_pred) + math.log(2 * math.pi) / 2.0 + (((labels - mean_pred) / std_pred) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        # Total loss is a function of both the cdf loss (fairness) and the nll loss (sharpness/accuracy).
        total_loss = (1 - self.coefficient) * loss_cdf + self.coefficient * loss_nll
        return loss_cdf, loss_nll, total_loss

    def _perform_early_stopping(
            self,
            val_loss: float,
    ) -> bool:
        if val_loss < self.best_model_loss:
            self.best_model_loss = val_loss
            self.patience_counter = 0
            self.best_model_checkpoint = copy.deepcopy(self.model)
            return False
        else:
            self.patience_counter += 1
            print("\n----------- patience_counter += 1 -----------")
            if self.patience_counter >= self.patience:
                self.accelerator.print(f"Early stopping with best validation loss: {self.best_model_loss}")
                self.model = self.best_model_checkpoint
                print("\n----------Perform early stopping----------")
                return True
            return False
