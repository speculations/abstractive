"""Module rays.py"""
import logging

import datasets
import ray.data
import transformers

import src.elements.parameters as pr
import src.elements.variable as vr


class Rays:
    """
    Ray data set.
    """

    def __init__(self, source: datasets.DatasetDict, variable: vr.Variable, parameters: pr.Parameters):
        """

        :param source:
        :param variable:
        :param parameters:
        """

        self.__source = source
        self.__variable = variable

        # The T5 specific parameters
        self.__parameters = parameters
        self.__tokenizer: transformers.PreTrainedTokenizerFast = self.__parameters.tokenizer

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __data(self) -> dict[str, ray.data.dataset.MaterializedDataset]:
        """

        :return:
        """

        data = {
            'train': ray.data.from_huggingface(self.__source['train']),
            'validate': ray.data.from_huggingface(self.__source['validate']),
            'test': ray.data.from_huggingface(self.__source['test'])
        }

        self.__logger.info(data)

        return data

    def exc(self) -> dict[str, ray.data.dataset.MaterializedDataset]:
        """

        :return:
        """

        return self.__data()
