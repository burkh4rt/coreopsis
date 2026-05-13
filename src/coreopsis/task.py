#!/usr/bin/env python3

"""
utilities
"""

import collections
import pathlib
import warnings

import torch
import transformers
from cotorra.trainer import Trainer as c_Trainer
from datasets.utils.logging import disable_progress_bar
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers import Trainer as t_Trainer

from coreopsis.loader import Loader

warnings.filterwarnings("ignore")
disable_progress_bar()
transformers.logging.set_verbosity_error()


def load_data(partition_id: int, num_partitions: int):
    """partition MIMIC into `num_partitions-1` parts of similar size and load UCMC
    as partition number `num-partitions-1`"""
    if partition_id < num_partitions - 1:
        dataset = Loader(
            cfg=c_Trainer().cfg,
            processed_data_home=pathlib.Path("./processed/mimic").resolve(),
            num_partitions=num_partitions - 1,
            partition_id=partition_id,
        )
    else:
        dataset = Loader(
            cfg=c_Trainer().cfg,
            processed_data_home=pathlib.Path("./processed/ucmc").resolve(),
        )
    return (dataset.get_train_data(), dataset.get_tuning_data())


def train(net, trainloader, testloader):
    trainer = t_Trainer(
        model=net,
        data_collator=c_Trainer().collate_fn,
        train_dataset=trainloader,
        eval_dataset=testloader,
        args=TrainingArguments(
            output_dir=str(c_Trainer().output_home), **c_Trainer().cfg.training_args
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()


def test(net, testloader):
    trainer = t_Trainer(
        model=net,
        data_collator=c_Trainer().collate_fn,
        eval_dataset=testloader,
        compute_loss_func=c_Trainer().loss,
        args=TrainingArguments(
            output_dir=str(c_Trainer().output_home), **c_Trainer().cfg.training_args
        ),
    )
    return trainer.evaluate()["eval_loss"]


def get_net():
    return c_Trainer().model_init()


def get_weights(net):
    return [val.cpu().to(torch.float32).numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
