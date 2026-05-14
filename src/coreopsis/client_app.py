#!/usr/bin/env python3

"""
things clients do;
note, new clients are created at the start of every round
"""

import hashlib
import logging
import time

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log

from coreopsis.task import get_net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, context: Context):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.created = time.time()
        self.id = hashlib.md5(
            (f"{id(self)}-{self.created}").encode("utf-8")
        ).hexdigest()[:7]
        self.pid = context.node_config["partition-id"]
        log(logging.INFO, f"Client {self.id} initialized (pid={self.pid})")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        self.net.to(self.device)
        log(logging.INFO, f"training {self.id} (pid={self.pid})...")
        train(self.net, self.trainloader, self.testloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        self.net.to(self.device)
        loss = test(self.net, self.testloader)
        log(logging.INFO, f"Validation {self.id} (pid={self.pid}): {loss=:.3f}")
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    trainloader, valloader = load_data(
        context.node_config["partition-id"], context.node_config["num-partitions"]
    )
    return FlowerClient(get_net(), trainloader, valloader, context).to_client()


app = ClientApp(client_fn)
