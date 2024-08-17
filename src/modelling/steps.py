"""Module steps.py"""
import logging

import datasets
import ray.data

import src.elements.parameters as pr
import src.elements.variable as vr
import src.modelling.assemble
import src.modelling.depositories
import src.modelling.intelligence
import src.modelling.preprocessing
import src.modelling.rays


class Steps:
    """
    Class Steps
    """

    def __init__(self, source: datasets.DatasetDict):
        """

        :param source: A dictionary of data splits; training, validation, etc., splits.
        """

        self.__source = source

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__parameters = pr.Parameters()

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

        # Re-write
        src.modelling.depositories.Depositories().exc(path=self.__variable.MODEL_OUTPUT_DIRECTORY)

        # Temporary: Data
        rays: dict[str, ray.data.dataset.MaterializedDataset] = src.modelling.rays.Rays(
            source=self.__source, variable=self.__variable, parameters=self.__parameters).exc()

        # Temporary: Modelling
        results = src.modelling.assemble.Assemble(data=rays).exc()
        self.__logger.info(results.__dir__())

        best = results.get_best_result()
        self.__logger.info(best.checkpoint)
        self.__logger.info(best.best_checkpoints)
        self.__logger.info(best.metrics_dataframe)
