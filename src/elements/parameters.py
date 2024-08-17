"""Module parameters.py"""
import collections

import transformers


class Parameters:
    """
    For setting terms that are particular to a pre-trained model
    architecture type; T5 specific parameters.

    input_prefix: str = 'summarize: '
    checkpoint: str = 'google-t5/t5-small'
    tokenizer: transformers.PreTrainedTokenizerFast = (
        transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint))
    n_trials: int = 4
    """

    def __init__(self):
        """
        Constructor
        """

        checkpoint: str = 'google-t5/t5-small'
        input_prefix: str
        tokenizer: transformers.PreTrainedTokenizerFast
        n_trials: int
        ModelArchitectureParameters = collections.namedtuple(
            typename='ModelArchitectureParameters',
            field_names=['input_prefix', 'checkpoint', 'tokenizer', 'n_trials'])

        self.parameters = ModelArchitectureParameters(
            input_prefix='summarize',
            checkpoint=checkpoint,
            tokenizer=transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint),
            n_trials=4)
