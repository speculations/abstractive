import logging
import ray.data
import datasets
import transformers


import src.elements.variable as vr
import src.modelling.t5.parameters as pr


class Rays:

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

    def __data(self):

        return {
            'train': ray.data.from_huggingface(self.__source['train']),
            'validate': ray.data.from_huggingface(self.__source['validate']),
            'test': ray.data.from_huggingface(self.__source['test'])
        }

    def exc(self):

        data: dict[str, ray.data.dataset.MaterializedDataset] = self.__data()
        self.__logger.info(type(data))
        self.__logger.info(data)

        # [self.__tokenization(data[k]) for k, _ in data.keys()]
