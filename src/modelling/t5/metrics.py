"""Module metrics.py"""
import logging

import evaluate
import numpy as np

import src.modelling.t5.parameters as pr


class Metrics:
    """
    For the compute_metrics parameter of Trainer modules, e.g.,
    transformers.Seq2SeqTrainer
    """

    def __init__(self, parameters: pr.Parameters):
        """

        :param parameters: T5 specific parameters
        """

        self.__tokenizer = parameters.tokenizer

        # ROUGE
        self.__rouge = evaluate.load('rouge')

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __predictions(self, predictions: np.ndarray):
        """

        :param predictions: A model's predictions
        :return:
        """

        # Predictions: Skipping special tokens
        return self.__tokenizer.batch_decode(predictions, skip_special_tokens=True)

    def __labels(self, labels: np.ndarray):
        """

        :param labels: The labels vis-à-vis the dependent variable data
        :return:
        """

        # Labels, Active Labels: Re-visit.  Replacing ...
        active = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)

        return self.__tokenizer.batch_decode(active, skip_special_tokens=True)

    def __extra_median(self, predictions: np.ndarray):
        """

        :param predictions: A model's predictions
        :return:
        """

        lengths = [np.count_nonzero(prediction != self.__tokenizer.pad_token_id)
                   for prediction in predictions]

        return np.median(lengths)


    def exc(self, bucket):
        """

        :param bucket: A data object of predictions, and the corresponding
                       dependent variable data
        :return:
        """

        self.__logger.info('BUCKET:\n%s', type(bucket))
        predictions: np.ndarray
        labels: np.ndarray
        predictions, labels = bucket

        # Calculations: Initially, Aggregator = True
        calculations: dict = self.__rouge.compute(
            predictions=self.__predictions(predictions=predictions),
            references=self.__labels(labels=labels), use_aggregator=True, use_stemmer=True)
        calculations['median'] = self.__extra_median(predictions=predictions)
        self.__logger.info(calculations.keys())

        # Rounding
        return {key: round(value, 5) for key, value in calculations.items()}
