"""Module intelligence.py"""
import logging

import ray.train
import transformers
import torch
import numpy as np

import collections
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
        self.__tokenizer: transformers.PreTrainedTokenizerFast = self.__parameters.tokenizer
        self.__variable = vr.Variable()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __tokenization(self, blob):
        """
        blob | datasets.formatting.formatting.LazyBatch

        :param blob: training or testing data batch
        :return:
        """

        # Input
        inputs = [self.__parameters.input_prefix + segment for segment in blob['text']]
        structure = self.__tokenizer(text=inputs, max_length=self.__variable.MAX_LENGTH_INPUT,
                                     truncation=True, padding='max_length')
        self.__logger.info(structure.keys())

        # Targets: A dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets = self.__tokenizer(text_target=blob['summary'].tolist(), max_length=self.__variable.MAX_LENGTH_TARGET,
                                   truncation=True, padding='max_length')
        self.__logger.info(targets.keys())

        # Beware
        temporary = dict()
        temporary['input_ids']  = torch.LongTensor(structure['input_ids'])
        temporary['attention_mask']  = torch.LongTensor(structure['attention_mask'])
        temporary['labels']  = torch.LongTensor(targets['input_ids'])
        self.__logger.info(temporary)

        return temporary

    def iterable(self, segment: str, batch_size: int):
        """

        :param segment:
        :param batch_size:
        :return:
        """

        part = ray.train.get_dataset_shard(segment)
        return part.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__tokenization)

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
