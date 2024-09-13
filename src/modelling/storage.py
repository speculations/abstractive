"""Module storage.py"""
import src.functions.directories

import src.modelling.arguments as ag


class Storage:
    """
    Prepares storage area
    """

    def __init__(self):
        """
        Constructor
        """

        self.__arguments = ag.Arguments()

        # The directories instance; for deleting, re-creating, directories.
        self.__directories = src.functions.directories.Directories()

    def exc(self) -> None:
        """

        :return:
        """

        self.__directories.cleanup(path=self.__arguments.MODEL_OUTPUT_DIRECTORY)
        self.__directories.create(path=self.__arguments.MODEL_OUTPUT_DIRECTORY)
