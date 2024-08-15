"""Module interface.py"""
import datasets

import src.modelling.t5.steps


class Interface:
    """
    An interface to one or more model development packages
    """

    def __init__(self, source: datasets.DatasetDict, device: str):
        """

        :param source: A datasets.DatasetDict for modelling
        :param device:
        """

        self.__source = source
        self.__device = device

    def exc(self):
        """
        Design I
        --------

        A container/instance of an image of this repository package will expect a string argument.  The
        argument will determine the model development activity that the instance will focus on.

        This method, i.e., exc(), will receive a string argument.  Let the argument's name be
        architecture, then

        match architecture:
            case 't5':
                src.modelling.t5.steps.Steps(...)
            case 'pegasus':
                src.modelling.pegasus.steps.Steps(...)
            case 'bart':
                src.modelling.bart.steps.Steps(...)
            case _:
                return 'Unknown architecture'

        :return:
        """

        src.modelling.t5.steps.Steps(source=self.__source, device=self.__device).exc()
