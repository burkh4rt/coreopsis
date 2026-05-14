#!/usr/bin/env python3

"""
train a model
"""

import logging
import pathlib

from flwr.common.logger import log
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers import Trainer as t_Trainer

from coreopsis.loader import Loader
from cotorra.trainer import Trainer as CotorraTrainer


class TrainerWithCustomLoss(t_Trainer):
    def __init__(self, compute_loss_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        log(
            logging.INFO,
            f"Initialized TrainerWithCustomLoss with {compute_loss_func=}",
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.compute_loss_func is not None:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = self.compute_loss_func(outputs, labels)
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)


class Trainer(CotorraTrainer):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    def __init__(
        self,
        main_cfg: pathlib.Path | str = None,
        mdl_cfg: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(main_cfg=main_cfg, mdl_cfg=mdl_cfg, **kwargs)
        self.loader = Loader(self.cfg, self.processed_data_home)
        log(
            logging.INFO,
            f"Initialized Trainer with {self.cfg=}",
        )

    def _make_trainer(self) -> TrainerWithCustomLoss:
        return TrainerWithCustomLoss(
            model_init=self.model_init,
            data_collator=self.collate_fn,
            compute_loss_func=self.loss,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_home), **self.cfg.training_args
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )


if __name__ == "__main__":
    self = Trainer()
    # self.train(verbose=True)
