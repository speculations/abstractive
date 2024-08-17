"""Module parameters.py"""
import typing

import transformers


class Parameters(typing.NamedTuple):
    """
    For setting terms that are particular to a pre-trained model architecture type

    Attributes
    ----------
    input_prefix: str

    checkpoint: str

    tokenizer: transformers.PreTrainedTokenizerFast

    perturbation_interval: int
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html

    n_trials: int

    """

    input_prefix: str = 'summarize: '
    checkpoint: str = 'google-t5/t5-small'
    tokenizer: transformers.PreTrainedTokenizerFast = (
        transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint))
    perturbation_interval: int = 2
    n_trials: int = 4
