#!/usr/bin/env python3

"""
classes supporting model training
"""

import pathlib

from transformers import TrainingArguments

from coreopsis.loader import Loader
from cotorra.trainer import Trainer as CotorraTrainer
from cotorra.trainer import TrainerWithCustomLoss


class Trainer(CotorraTrainer):
    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(
            training_cfg=training_cfg,
            processed_data_home=processed_data_home,
            output_home=output_home,
            **kwargs,
        )
        self.loader = Loader(self.cfg, self.processed_data_home)

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
        )  # no early stopping


if __name__ == "__main__":
    self = Trainer()
    # self.train(verbose=True)
