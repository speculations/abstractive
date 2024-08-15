"""
Module config.py
"""
import os


class Config:
    """
    Description
    -----------

    A class for configurations
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        # Warehouse
        self.warehouse = os.path.join(os.getcwd(), 'warehouse')

        # For reproducibility purposes
        self.seed = 5

        # For data splitting purposes
        self.fraction_validate = 0.2
        self.fraction_test = 0.25
