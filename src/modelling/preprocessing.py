"""Module preprocessing.py"""
import logging
import transformers
import torch

import src.modelling.custom
import src.modelling.parameters as pr


class Preprocessing:
    """
    Class Preprocessing

    This class preprocesses a data batch, e.g., splits for model
    development, in line with T5 (Text-To-Text Transfer Transformer)
    architecture expectations.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__variable = src.modelling.custom.Custom().custom

        # The T5 specific parameters
        self.__parameters = pr.Parameters()
        self.__tokenizer: transformers.PreTrainedTokenizerFast = self.__parameters.tokenizer

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __tokenization(self, blob):
        """

        :param blob:
        :return:
        """

        # Independent Variable
        inputs = [self.__parameters.input_prefix + segment for segment in blob['text']]
        structure = self.__tokenizer(text=inputs, max_length=self.__variable.MAX_LENGTH_INPUT, truncation=True)

        # Dependent Variable; targets has a dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets = self.__tokenizer(text_target=blob['summary'], max_length=self.__variable.MAX_LENGTH_TARGET, truncation=True)
        structure['labels']  = torch.LongTensor(targets['input_ids'])

        return structure

    def exc(self, blob) -> transformers.BatchEncoding:
        """
        blob | datasets.formatting.formatting.LazyBatch

        :param blob: training or testing data batch
        :return:
        """

        structure = self.__tokenization(blob=blob)
        self.__logger.info(type(structure))
        self.__logger.info(structure)

        return structure
