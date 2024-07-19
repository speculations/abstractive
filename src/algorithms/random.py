"""
Module random.py
"""
import numpy as np

import config


class Random:
    """
    Description
    -----------
    This class returns a random number within [0.0   999.99)    
    """

    def __init__(self) -> None:
        """
        The constructor
        """

        # Constructing a random generator
        seed = config.Config().seed()
        self.__rng = np.random.default_rng(seed=seed)

        # The random number boundaries
        self.__minimum: float = 0.0
        self.__maximum: float = 999.99

    def exc(self) -> float:
        """

        :return: A random number
        """

        value: float = (self.__maximum - self.__minimum) * self.__rng.random() + self.__minimum

        return value
