"""Module source.py"""
import logging

import datasets

import config


class Source:
    """
    A class for data preparation.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

        # The Data: Herein, the dictionary segments are being reset such that
        # the segments are <training>, <validate>, and <test>; initially <training>,
        # <test>, and <ca_test> respectively.
        self.__dataset: datasets.DatasetDict = datasets.load_dataset('billsum')
        validate = self.__dataset.pop('test')
        test = self.__dataset.pop('ca_test')
        self.__dataset['validate'] = validate
        self.__dataset['test'] = test

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __temporary(self) -> datasets.DatasetDict:
        """

        :return:
        """

        splits: datasets.DatasetDict = self.__dataset['test'].train_test_split(
            test_size=self.__configurations.fraction_validate)

        parts: datasets.DatasetDict = splits['test'].train_test_split(
            test_size=self.__configurations.fraction_test)

        splittings: datasets.DatasetDict = datasets.DatasetDict({
            'train': splits['train'],
            'validate': parts['train'],
            'test': parts['test']
        })

        return splittings

    def exc(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # The initial focus
        temporary: datasets.DatasetDict = self.__temporary()
        self.__logger.info('Initially focusing on a small data segment\n%s\n%s', type(temporary), temporary.keys())
        self.__logger.info('train:\n%s', temporary['train'].shape)
        self.__logger.info('validate:\n%s', temporary['validate'].shape)
        self.__logger.info('test:\n%s', temporary['test'].shape)

        self.__logger.info('The parts of a data record:\n%s', temporary['train'][0].keys())

        return temporary
