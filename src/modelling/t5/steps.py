"""Module steps.py"""
import logging
import os

import datasets
import ray.data

import config
import src.elements.variable as vr
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
        max_steps_per_epoch = self.__source['train'].shape[0] // (self.__variable.TRAIN_BATCH_SIZE * self.__variable.N_GPU)
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

        # Temporary
        rays: dict[str, ray.data.dataset.MaterializedDataset] = src.modelling.t5.rays.Rays(
            source=self.__source, variable=self.__variable, parameters=self.__parameters).exc()

        # Preprocessing Instance: For tokenization.  Converting each split into a T5 tokenized split
        # preprocessing = src.modelling.t5.preprocessing.Preprocessing(variable=self.__variable, parameters=self.parameters)
        # data: datasets.DatasetDict = self.__source.map(preprocessing.exc, batched=True)

        # Model
        # intelligence = src.modelling.t5.intelligence.Intelligence(variable=self.__variable, parameters=self.parameters)
        # model = intelligence(data=data)
        # self.__logger.info(dir(model))
