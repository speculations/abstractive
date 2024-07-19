"""
Module root.py
"""

import math


class Roots:
    """
    Description
    -----------
    A class that returns the square root of values >= 0.
    """

    def __init__(self) -> None:
        pass

    def exc(self, value: float) -> float:
        """

        :param value: A float value
        """

        try:
            value = math.sqrt(value)
            return value
        except ValueError as err:
            raise err from err
