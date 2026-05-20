#!/usr/bin/env python3

"""
CLI for coreopsis - configurable, choreographed federated training
"""

from importlib.metadata import version

from flwr.cli.app import app as flwr_app

__version__ = version("coreopsis")


def main():
    flwr_app.info.help = (
        f"Choreographed federated learning with flower (v{__version__})"
    )
    flwr_app(prog_name="coreopsis")


if __name__ == "__main__":
    main()
    # breakpoint()
