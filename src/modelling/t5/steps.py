"""Module steps.py"""
import logging
import os

import datasets
import ray.data

import config
import src.elements.variable as vr
import src.modelling.t5.assemble
import src.modelling.t5.depositories
import src.modelling.t5.intelligence
import src.modelling.t5.parameters as pr
import src.modelling.t5.preprocessing
import src.modelling.t5.rays


class Steps:
    """
    Class Steps
    """

    def __init__(self, source: datasets.DatasetDict, device: str):
        """

        :param source: A dictionary of data splits; training, validation, etc., splits.
        :param device: A string denoting graphics or central processing unit, i.e., 'cuda' or 'cpu', respectively.
        """

        self.__source = source
        self.__device = device

        # A set of values for machine learning model development
        self.__n_epochs = 2
        self.__variable = self.__get_variable()
        self.__parameters = pr.Parameters()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __get_variable(self):
        """

        :return:
        """

        variable = vr.Variable()

        # ... steps & epochs
        max_steps_per_epoch = self.__source['train'].shape[0] // (variable.TRAIN_BATCH_SIZE * variable.N_GPU)
        max_steps = max_steps_per_epoch * self.__n_epochs

        # Update
        variable = variable._replace(MODEL_OUTPUT_DIRECTORY=os.path.join(config.Config().warehouse, 't5'),
                                     DEVICE=self.__device, EPOCHS=self.__n_epochs, MAX_STEPS=max_steps)

        return variable

    def exc(self):
        """
        model.save_model()

        :return:
        """

        # Re-write
        src.modelling.t5.depositories.Depositories().exc(path=self.__variable.MODEL_OUTPUT_DIRECTORY)

        # Temporary: Data
        rays: dict[str, ray.data.dataset.MaterializedDataset] = src.modelling.t5.rays.Rays(
            source=self.__source, variable=self.__variable, parameters=self.__parameters).exc()
        self.__logger.info(rays)

        # Temporary: Modelling
        results = src.modelling.t5.assemble.Assemble(data=rays, variable=self.__variable, parameters=self.__parameters).__call__()
        self.__logger.info(results.__dict__)
        self.__logger.info(results.__dir__())

        best = results.get_best_result()
        self.__logger.info(best.checkpoint)
        self.__logger.info(best.best_checkpoints)
        self.__logger.info(best.metrics_dataframe)
