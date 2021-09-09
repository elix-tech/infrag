"""
Commandline logger class

Copyright (c) 2021 Elix, Inc.
"""

import logging
from pprint import pprint
from typing import List, Union

from .abstract_logger import AbstractLogger


class CommandLineLogger(AbstractLogger):
    def __init__(self, args):
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.DEBUG,
        )
        pprint(vars(args))

    def log_metric(self, name: str, value: Union[int, float]):
        logging.info("{}: \t {}".format(name, value))

    def log_text(self, name: str, text: str):
        logging.info("{}: \t {}".format(name, text))

    def log_values(self, name: str, values: List[float]):
        logging.info("{}: \t {}".format(name, values))
