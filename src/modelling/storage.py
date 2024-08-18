"""Module storage.py"""
import src.functions.directories

import src.elements.variable as vr


class Storage:
    """
    Prepares storage area
    """

    def __init__(self):
        """
        Constructor
        """

        # The directories instance; for deleting, re-creating, directories.
        self.__directories = src.functions.directories.Directories()

        # A set of values for machine learning model development
        self.__variable = vr.Variable()

    def exc(self) -> None:
        """

        :return:
        """

        self.__directories.cleanup(path=self.__variable.MODEL_OUTPUT_DIRECTORY)
        self.__directories.create(path=self.__variable.MODEL_OUTPUT_DIRECTORY)
