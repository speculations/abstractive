"""Module parameters.py"""
import typing

import transformers


class Parameters(typing.NamedTuple):
    """
    Class: Parameters

    Crazy?
    For setting terms that are particular to a pre-trained model architecture type
    """

    input_prefix: str = 'summarize: '
    checkpoint: str = 'google-t5/t5-small'
    tokenizer: transformers.PreTrainedTokenizerFast = (
        transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint))
    n_trials: int = 4
    n_cpu: int = 8
    n_gpu: int = 1
