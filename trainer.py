import math
import typing

import torch
import transformers
from datetime import datetime


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
            tokenizer: torch.utils.data.Dataset = None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            preprocess_logits_for_metrics=None,
    ):
        """Initialize a Trainer class for Randomized Individual Calibration.

        Args:
            model: The model to train.
            args: The training arguments.
            coefficient: The coefficient to use to balance between the cdf loss and nll loss.
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

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

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

        if 'input_r' in inputs:
            input_r = inputs.get("input_r")
        else:
            input_r = torch.rand(input_ids.shape[0], 1, device=input_ids.device)

        outputs = self.model.forward(input_ids, attention_mask, input_r)
        mean, stddev = outputs
        cdf = 0.5 * (1.0 + torch.erf((target - mean) / stddev / math.sqrt(2)))

        loss_cdf = torch.abs(cdf - input_r).mean()

        # TODO: Move loss stddev to the compute metrics function.
        loss_stddev = stddev.mean()

        # Log likelihood of by under the predicted Gaussian distribution
        # TODO: Describe the nll loss function.
        loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((target - mean) / stddev) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        loss = (1 - self.coefficient) * loss_cdf + self.coefficient * loss_nll

        self.log({
            'loss': loss,
            'loss_cdf': loss_cdf,
            'loss_nll': loss_nll,
            'loss_stddev': loss_stddev,
        })

        return (loss, outputs) if return_outputs else loss
