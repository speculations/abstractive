"""Module rays.py"""
import logging

import datasets
import ray.data


class Rays:
    """
    Ray data set.
    """

    def __init__(self, source: datasets.DatasetDict):
        """

        :param source: The data in focus
        """

        self.__source = source


        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __data(self) -> dict[str, ray.data.dataset.MaterializedDataset]:
        """

        :return:
        """

        data = {
            'train': ray.data.from_huggingface(self.__source['train']),
            'validate': ray.data.from_huggingface(self.__source['validate']),
            'test': ray.data.from_huggingface(self.__source['test'])
        }

        self.__logger.info(data)

        return data

    def exc(self) -> dict[str, ray.data.dataset.MaterializedDataset]:
        """

        :return:
        """

        return self.__data()
