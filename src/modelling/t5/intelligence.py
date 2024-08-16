"""Module intelligence.py"""
import logging

import ray.train
import transformers

import src.elements.variable as vr
import src.modelling.t5.parameters as pr
import src.modelling.t5.preprocessing
import src.modelling.t5.skeleton


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, variable: vr.Variable, parameters: pr.Parameters):
        """

        :param variable: A suite of values for machine learning
                         model development
        :param parameters: T5 specific parameters
        """

        self.__variable = variable
        self.__parameters = parameters

        # Configuration
        self.__skeleton = src.modelling.t5.skeleton.Skeleton(variable=variable, parameters=self.__parameters).exc()
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing(variable=self.__variable, parameters=self.__parameters)

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
