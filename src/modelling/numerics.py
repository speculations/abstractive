"""Module numerics.py"""
import ray.data

import src.elements.variable as vr


class Numerics:
    """
    Class Numerics

    Calculates max_steps for TrainerArguments ...
    """

    def __init__(self, n_training_instances: int, variable: vr.Variable):
        """

        :param n_instances:
        :param variable:
        """

        self.__n_training_instances = n_training_instances
        self.__variable = variable


    def __call__(self) -> int:
        """

        :return:
        """

        max_steps_per_epoch: int = self.__n_training_instances // (self.__variable.TRAIN_BATCH_SIZE * self.__variable.N_GPU)
        max_steps: int = max_steps_per_epoch * self.__variable.EPOCHS

        return max_steps