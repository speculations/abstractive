"""Module parameters.py"""
import transformers

import src.modelling.arguments


class Parameters:
    """
    Rename

    This class sets up the model's tokenizer.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__arguments: src.modelling.arguments.Arguments = src.modelling.arguments.Arguments()

    def __call__(self):
        """

        :return:
            tokenizer: transformers.PreTrainedTokenizerFast
        """

        return transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.__arguments.checkpoint)

