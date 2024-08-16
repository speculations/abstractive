"""Module intelligence.py"""
import logging

import ray.train
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.parameters as pr
import src.modelling.t5.preprocessing
import src.modelling.t5.settings
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

        # Setting: scheduler, arguments, ...
        self.__settings = src.modelling.t5.settings.Settings(variable=variable)

        # Instances
        self.__metrics = src.modelling.t5.metrics.Metrics(parameters=self.__parameters)

        # Configuration
        self.__skeleton = src.modelling.t5.skeleton.Skeleton(variable=variable, parameters=self.__parameters).exc()
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing(variable=self.__variable, parameters=self.__parameters)

    def __data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def __model(self):
        """

        :return:
        """

        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=self.__skeleton
        )

    def __iterable(self, segment: str, batch_size: int):
        """

        :param segment:
        :param batch_size:
        :return:
        """

        part = ray.train.get_dataset_shard(segment)
        return part.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__preprocessing.exc)

    def __call__(self):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :return:
        """

        training = self.__iterable(segment='train', batch_size=self.__variable.TRAIN_BATCH_SIZE)
        validating = self.__iterable(segment='eval', batch_size=self.__variable.VALIDATE_BATCH_SIZE)

        trainer = transformers.Seq2SeqTrainer(
            model_init=self.__model,
            args=self.__settings.args(),
            train_dataset=training,
            eval_dataset=validating,
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator(),
            compute_metrics=self.__metrics.exc
        )
