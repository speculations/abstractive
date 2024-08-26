"""Module steps.py"""
import logging

import datasets
import ray.data
import ray.tune

import src.modelling.reduced
import src.modelling.rays
import src.modelling.storage


class Steps:
    """
    Class Steps
    """

    def __init__(self, source: datasets.DatasetDict):
        """

        :param source: A dictionary of data splits; training, validation, etc., splits.
        """

        self.__source = source

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """
        model.save_model()

        :return:
        """

        # Storage
        src.modelling.storage.Storage().exc()

        # Data
        rays: dict[str, ray.data.dataset.MaterializedDataset] = src.modelling.rays.Rays(source=self.__source).exc()

        # Modelling
        results: ray.tune.ResultGrid = src.modelling.reduced.Reduced(data=rays).exc()
        self.__logger.info(results.__dir__())

        best = results.get_best_result()
        self.__logger.info(best.checkpoint)
        self.__logger.info(best.best_checkpoints)
        self.__logger.info(best.metrics_dataframe)
