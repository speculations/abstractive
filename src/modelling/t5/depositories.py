"""Module depositories.py"""
import src.functions.directories

class Depositories:
    """
    Class Depositories

    This class re-prepares output directories.  It will be
    moved outwith src.modelling.t5
    """

    def __init__(self):
        """
        Constructor
        """

    def exc(self, path):
        """

        :param path: The directory path that will be re-created.
        :return:
        """

        directories = src.functions.directories.Directories()
        directories.cleanup(path=path)
        directories.create(path=path)
