"""Module preprocessing.py"""
import logging

import datasets.formatting.formatting
import ray.data
import torch
import transformers

import src.elements.variable as vr
import src.modelling.arguments as ag


class Preprocessing:
    """
    Class Preprocessing

    This class preprocesses a data batch, e.g., splits for model
    development, in line with T5 (Text-To-Text Transfer Transformer)
    architecture expectations.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerFast):
        """

        :param tokenizer: T5 specific
        """

        self.__tokenizer = tokenizer

        self.__variable = vr.Variable()
        self.__arguments = ag.Arguments()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def batches(self, blob: datasets.formatting.formatting.LazyBatch) -> transformers.BatchEncoding:
        """

        :param blob: training or testing data batch
        :return:
        """

        # Independent Variable
        inputs = [self.__arguments.input_prefix + segment for segment in blob['text']]
        structure: transformers.BatchEncoding = self.__tokenizer(
            text=inputs, max_length=self.__variable.MAX_LENGTH_INPUT, truncation=True)

        # Dependent Variable; targets has a dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets: transformers.BatchEncoding = self.__tokenizer(
            text_target=blob['summary'], max_length=self.__variable.MAX_LENGTH_TARGET, truncation=True)
        structure['labels']  = targets['input_ids']

        return structure

    def __tokenization(self, blob):
        """
        blob | datasets.formatting.formatting.LazyBatch

        :param blob: training or testing data batch
        :return:
        """

        # Input: A dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        entries = [self.__arguments.input_prefix + segment for segment in blob['text']]
        inputs = self.__tokenizer(text=entries, max_length=self.__variable.MAX_LENGTH_INPUT,
                                     truncation=True, padding='max_length')

        # Targets: A dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets = self.__tokenizer(text_target=blob['summary'].tolist(), max_length=self.__variable.MAX_LENGTH_TARGET,
                                   truncation=True, padding='max_length')

        # Beware
        structure = {'input_ids': torch.LongTensor(inputs['input_ids']),
                     'attention_mask': torch.LongTensor(inputs['attention_mask']),
                     'labels': torch.LongTensor(targets['input_ids'])}
        self.__logger.info(structure)

        return structure

    def iterables(self, part: ray.data.DataIterator, batch_size: int):
        """

        :param part:
        :param batch_size:
        :return:
        """

        return part.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__tokenization)
