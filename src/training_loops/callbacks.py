"""Custom HuggingFace TrainerCallbacks shared across training loops."""

from __future__ import annotations

import math

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ThresholdStopCallback(TrainerCallback):
    """Stop training 1 epoch after eval_exact_match first exceeds a threshold.

    When exact_match >= threshold is first observed at epoch E (floored),
    training stops after the first eval that occurs at epoch >= E + 1, giving
    the model one full additional epoch beyond the point it crossed the
    threshold.

    Args:
        threshold: eval_exact_match value that triggers the countdown.
    """

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold
        self.triggered_epoch: float | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> TrainerControl:
        exact_match = metrics.get("eval_exact_match", 0.0)
        current_epoch = state.epoch  # continuous float, e.g. 2.4 = 40% through epoch 3

        if self.triggered_epoch is None and exact_match >= self.threshold:
            self.triggered_epoch = math.floor(current_epoch) + 1
            print(
                f"[ThresholdStop] eval_exact_match={exact_match:.4f} >= {self.threshold:.4f} "
                f"at epoch {current_epoch:.2f}. Will stop after epoch {self.triggered_epoch}."
            )

        if self.triggered_epoch is not None and current_epoch >= self.triggered_epoch:
            control.should_training_stop = True
            print(f"[ThresholdStop] Stopping at epoch {current_epoch:.2f}.")

        return control
