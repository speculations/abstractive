"""
Test module test_roots.py
"""
import logging
import math
import pytest

import src.algorithms.roots
import src.functions.cache


class TestRoots:
    """
    This class tests src.algorithms.roots
    """

    @pytest.fixture
    def value(self) -> float:
        """
        
        :return: A number
        """

        return -2

    def test_exc(self, value: float):
        """
        Testing src.algorithms.roots.Roots().exc(...)
        """

        roots = src.algorithms.roots.Roots()

        try:
            root: float = roots.exc(value=value)
            assert root  == math.sqrt(value), "The calculation is incorrect"
        except ValueError as err:
            logging.log(level=logging.INFO, msg=err)

        src.functions.cache.Cache().exc()
