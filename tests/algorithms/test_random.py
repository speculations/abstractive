"""
Test module ...
"""

import pytest
import src.algorithms.random
import src.functions.cache

class TestRandom:
    """
    The test class of src.algortithms.random
    """

    @pytest.fixture
    def minimum(self) -> float:
        """
        
        :return: The expected minimum value 
        """

        return 0.0

    @pytest.fixture
    def maximum(self) -> float:
        """
        
        :return: The expected upper boundary value 
        """

        return 999.99

    def test_exc(self, minimum: float, maximum: float):
        """
        Testing src.algortithms.random.Random().exc()
        """

        random = src.algorithms.random.Random()
        value = random.exc()

        assert (value >= minimum) & (value < maximum), "The random value must be a value within [0.00 999.99)"

        src.functions.cache.Cache().exc()
