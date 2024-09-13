"""Module arguments.py"""
import os
import typing

import config


class Arguments(typing.NamedTuple):
    """
    For setting terms that are particular to a pre-trained model
    architecture type; T5 specific arguments

    input_prefix: str
        The T5 prefix for summarisation

    checkpoint: str
        The name of the pre-trained model of interest, in focus.

    MODEL_OUTPUT_DIRECTORY: str
        A directory for model outputs
    """

    input_prefix: str = 'summarize: '
    checkpoint: str = 'google-t5/t5-small'
    MODEL_OUTPUT_DIRECTORY: str = os.path.join(config.Config().warehouse, 't5')
