
import datasets
import ray.data

import src.data.source
import src.data.rays


class Interface:

    def __init__(self):
        """
        Constructor
        """

        self.__source: datasets.DatasetDict = src.data.source.Source().exc()

    def get_dataset(self) -> datasets.DatasetDict:
        """

        :return:
            A dictionary of data splits; training, validation, etc., splits.
        """

        return self.__source

    def get_rays(self):
        """

        :return:
            A dictionary of data splits; training, validation, etc., splits.
        """


        rays: dict[str, ray.data.dataset.MaterializedDataset] = src.data.rays.Rays(
            source=self.__source).exc()

        return rays
