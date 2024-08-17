"""Module skeleton.py"""
import logging

import transformers

import src.modelling.t5.custom
import src.modelling.t5.parameters as pr


class Skeleton:
    """
    Class Model
    """

    def __init__(self):
        """
        Constructor
        """

        self.__variable = src.modelling.t5.custom.Custom().custom
        self.__parameters = pr.Parameters()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
            model
        """

        # Configurations
        config = transformers.GenerationConfig.from_pretrained(
            pretrained_model_name=self.__parameters.checkpoint, **{'max_new_tokens': self.__variable.MAX_NEW_TOKENS})
        self.__logger.info('max_length: %s', config.max_length)
        self.__logger.info('max_new_tokens: %s', config.max_new_tokens)

        return config
