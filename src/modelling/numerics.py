"""Module numerics.py"""
import ray.data

import src.elements.variable as vr


class Numerics:
    """
    Class Numerics

    Calculates max_steps for TrainerArguments ...
    """

    def __init__(self, data: dict[str, ray.data.dataset.MaterializedDataset], variable: vr.Variable):
        """

        :param data:
        :param variable:
        """

        self.__data = data
        self.__variable = variable


    def __call__(self) -> int:
        """

        :return:
        """

        max_steps_per_epoch: int = self.__data['train'].count() // (self.__variable.TRAIN_BATCH_SIZE * self.__variable.N_GPU)
        max_steps: int = max_steps_per_epoch * self.__variable.EPOCHS

        return max_steps