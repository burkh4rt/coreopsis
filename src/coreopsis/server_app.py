#!/usr/bin/env python3

"""
things the server does
"""

from cotorra.logger import Logger
from cotorra.trainer import Trainer
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .save_fed_avg import SaveFedAvg
from .task import get_weights


def server_fn(context: Context):

    Logger().info(f"{context.run_config=}")

    net = Trainer().model_init()
    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    strategy = SaveFedAvg(
        fraction_fit=1.0,  # Fraction of clients used during training
        fraction_evaluate=1.0,  # Fraction of clients used during validation
        initial_parameters=initial_parameters,
        net=net,
        context=context,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
