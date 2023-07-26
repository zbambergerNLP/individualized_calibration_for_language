import logging
import math
import typing

import numpy as np
import torch
import transformers
from datetime import datetime

import data_preprocessing
import model as individualized_calibration_model

from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    nested_detach,
    find_batch_size,
    nested_concat,
    nested_numpify,
    IterableDatasetShard,
)
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize

logger = logging.getLogger(__name__)


class RandomizedIndividualCalibrationTrainer(transformers.Trainer):
    """
    Trainer for Randomized Individual Calibration.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            args: transformers.TrainingArguments,
            coefficient: float = 1.0,
            data_collator: transformers.DataCollator = None,
            train_dataset: torch.utils.data.Dataset = None,
            eval_dataset: torch.utils.data.Dataset = None,
            tokenizer: transformers.PreTrainedTokenizerBase = None,
            model_init: typing.Callable[[], torch.nn.Module] = None,
            compute_metrics: typing.Callable[
                [individualized_calibration_model.CommentRegressorPrediction, typing.Optional[bool]],
                typing.Dict] = None,
            callbacks: typing.List[transformers.TrainerCallback] = None,
            optimizers: typing.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        """Initialize a Trainer class for Randomized Individual Calibration.

        Args:
            model: The model to train.
            args: The training arguments.
            coefficient: The coefficient to use to balance between the cdf loss and nll loss. Bounded between 0 and 1.
                Closer to 1 is less fair, but more accurate.
            data_collator: The data collator to use.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            tokenizer: The tokenizer to use.
            model_init: The model initialization function.
            compute_metrics: The metrics computation function.
            callbacks: The callbacks to use.
            preprocess_logits_for_metrics: The logits preprocessing function.

        Returns:
            A Trainer instance.
        """
        # Initialize a few variables that will be overriden by super().__init__().
        # This is primarily done for type checking.
        self.accelerator = None
        self.is_deepspeed_enabled = None
        self.is_fsdp_enabled = None

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            optimizers=optimizers,
            model_init=model_init,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.compute_metrics = compute_metrics
        self.coefficient = coefficient

        # Current time
        now = datetime.now()
        self.date_time = now.strftime("%Y_%m_%d")
        self.hour_time = now.strftime("%H_%M")

    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: typing.Dict[str, typing.Any],
            return_outputs: bool = False,
    ) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compute the loss for the given inputs.

        Args:
            model: The model to use.
            inputs: The inputs to use.
            return_outputs: Whether to return the outputs.

        Returns:
            The loss and the outputs if return_outputs is True. Otherwise, only the loss.
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        target = inputs.get("labels")
        input_r = inputs.get("input_r")

        outputs = self.model.forward(input_ids, attention_mask, input_r)
        mean, stddev = outputs

        # TODO: Generate both a deterministic and non-deterministic version of the model's prediction.
        #  In the deterministic case, use the mean as the prediction. In the non-deterministic case, sample from
        #  the predicted Gaussian distribution.
        cdf = 0.5 * (1.0 + torch.erf((target - mean) / stddev / math.sqrt(2)))
        loss_cdf = torch.abs(cdf - input_r).mean()

        # Log likelihood under the predicted Gaussian distribution
        # TODO: Describe the nll loss function.
        loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((target - mean) / stddev) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        # Total loss is a function of both the cdf loss (fairness) and the nll loss (sharpness/accuracy).
        loss = (1 - self.coefficient) * loss_cdf + self.coefficient * loss_nll
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Any]],
        prediction_loss_only: bool,
        ignore_keys: typing.Optional[typing.List[str]] = None,
    ) -> typing.Tuple[
        typing.Optional[torch.Tensor],
        typing.Optional[torch.Tensor],
        typing.Optional[torch.Tensor],
        typing.Optional[torch.Tensor],
    ]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model: The model to evaluate.
            inputs: The inputs and targets of the model. The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument `labels`. Check your model's documentation for all
                accepted arguments.
            prediction_loss_only: Whether to return the loss only.
            ignore_keys: A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                mean, stddev = outputs
                loss = loss.mean().detach()
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                mean, stddev = outputs

        if prediction_loss_only:
            return loss, None, None, None

        mean = nested_detach(mean)
        stddev = nested_detach(stddev)

        return loss, mean, stddev, labels

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: typing.Optional[bool] = None,
        ignore_keys: typing.Optional[typing.List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        means_host = None
        stddevs_host = None
        labels_host = None
        inputs_host = None
        input_rs_host = None
        groups_host = {}

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_means = None
        all_stddevs = None
        all_labels = None
        all_inputs = None
        all_input_rs = None
        all_groups = {}
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, mean, stddev, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
            input_r = inputs["input_r"].to(loss.device)

            groups = {}
            for group_name in data_preprocessing.GROUP_LIST:
                groups[group_name] = inputs.pop(group_name).to(loss.device)

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if mean is not None:
                means = self.accelerator.pad_across_processes(mean, dim=1, pad_index=-100).contiguous()
                mean = self.accelerator.gather_for_metrics((means))
                means_host = mean if means_host is None else nested_concat(means_host, mean, padding_index=-100)
            if stddev is not None:
                stddevs = self.accelerator.pad_across_processes(stddev, dim=1, pad_index=-100)
                stddevs = self.accelerator.gather_for_metrics((stddevs))
                stddevs_host = (
                    stddevs
                    if stddevs_host is None
                    else nested_concat(stddevs_host, stddevs, padding_index=-100)
                )
            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if input_r is not None:
                # TODO: Investigate why input_r does not need to be repeated like loss, mean, stddev, labels above.
                input_rs = self.accelerator.gather_for_metrics((input_r))
                input_rs_host = input_rs if input_rs_host is None else nested_concat(
                    input_rs_host, input_rs, padding_index=-100)
            if groups != {}:
                for group in groups:
                    groups[group] = self.accelerator.gather_for_metrics((groups[group]))
                    if group not in groups_host:
                        groups_host[group] = groups[group]
                    else:
                        groups[group] = self.accelerator.pad_across_processes(
                            groups_host[group], dim=1, pad_index=-100)
                        groups[group] = self.accelerator.gather_for_metrics((groups[group]))
                        groups_host[group] = nested_concat(groups_host[group], groups[group], padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if means_host is not None:
                    means = nested_numpify(means_host)
                    all_means = (
                        means if all_means is None else nested_concat(all_means, means, padding_index=-100)
                    )
                if stddevs_host is not None:
                    stddevs = nested_numpify(stddevs_host)
                    all_stddevs = (
                        stddevs if all_stddevs is None else nested_concat(all_stddevs, stddevs, padding_index=-100)
                    )
                if input_rs_host is not None:
                    input_rs = nested_numpify(input_rs_host)
                    all_input_rs = (
                        input_rs if all_input_rs is None else nested_concat(all_input_rs, input_rs, padding_index=-100)
                    )
                if groups_host is not None:
                    for group_name in groups_host:
                        group_value = nested_numpify(groups_host[group_name])
                        all_groups[group_name] = (
                            group_value if group_name not in all_groups else nested_concat(
                                all_groups[group_name], group_value, padding_index=-100,
                            )
                        )

                # Set back to None to begin a new accumulation
                (losses_host,
                 means_host,
                 stddevs_host,
                 inputs_host,
                 labels_host,
                 input_rs_host) = (None, None, None, None, None, None)
                groups_host = {}

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if means_host is not None:
            means = nested_numpify(means_host)
            all_means = means if all_means is None else nested_concat(all_means, means, padding_index=-100)
        if stddevs_host is not None:
            stddevs = nested_numpify(stddevs_host)
            all_stddevs = stddevs if all_stddevs is None else nested_concat(all_stddevs, stddevs, padding_index=-100)
        if input_rs_host is not None:
            input_rs = nested_numpify(input_rs_host)
            all_input_rs = (
                input_rs if all_input_rs is None else nested_concat(all_input_rs, input_rs, padding_index=-100)
            )
        if groups_host != {}:
            for group_name in groups_host:
                group = nested_numpify(groups_host[group_name])
                all_groups[group_name] = (
                    group if group_name not in all_groups else nested_concat(
                        all_groups[group], group, padding_index=-100,
                    )
                )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
                self.compute_metrics is not None and
                all_means is not None and
                all_stddevs is not None and
                all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    individualized_calibration_model.CommentRegressorPrediction(
                        means=all_means,
                        stddevs=all_stddevs,
                        label_ids=all_labels,
                        input_r=all_input_rs,
                        groups=all_groups,
                    ),
                )
            else:
                metrics = self.compute_metrics(
                    individualized_calibration_model.CommentRegressorPrediction(
                        means=all_means,
                        stddevs=all_stddevs,
                        label_ids=all_labels,
                        input_r=all_input_rs,
                        groups=all_groups,
                    )
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # TODO: Note that currently the predictions are the means of the CDF produced by the model.
        return EvalLoopOutput(predictions=all_means, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
