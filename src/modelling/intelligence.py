"""Module intelligence.py"""
import collections
import logging

import transformers

import src.elements.variable as vr


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, parameters: collections.namedtuple(typename='ModelArchitectureParameters',
                                                          field_names=['input_prefix', 'checkpoint', 'tokenizer'])):
        """

        :param parameters: T5 specific parameters
        """

        self.__parameters = parameters
        self.__variable = vr.Variable()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def model(self):
        """

        :return:
        """

        config = transformers.GenerationConfig.from_pretrained(
            pretrained_model_name=self.__parameters.checkpoint, **{'max_new_tokens': self.__variable.MAX_NEW_TOKENS})
        self.__logger.info('max_length: %s', config.max_length)
        self.__logger.info('max_new_tokens: %s', config.max_new_tokens)

        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=config
        )
