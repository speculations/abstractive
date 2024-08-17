"""Module intelligence.py"""
import logging

import ray.train
import transformers

import src.modelling.t5.parameters as pr
import src.modelling.t5.preprocessing
import src.modelling.t5.skeleton


class Intelligence:
    """
    The model development class.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__parameters = pr.Parameters()

        # Configuration
        self.__skeleton = src.modelling.t5.skeleton.Skeleton().exc()
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing()

    def iterable(self, segment: str, batch_size: int):
        """

        :param segment:
        :param batch_size:
        :return:
        """

        part = ray.train.get_dataset_shard(segment)
        return part.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__preprocessing.exc)

    def data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def model(self):
        """

        :return:
        """

        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=self.__skeleton
        )
