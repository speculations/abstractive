"""
Module: streams.py
"""
import csv
import pathlib

import pandas as pd
import requests


class Streams:
    """
    Class Streams

    Description
    -----------
    Reads 'csv' files, and exports data frames to 'csv' files.
    """

    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def write(blob: pd.DataFrame, path: str) -> str:
        """
        :param blob: The data being stored in a `csv` file.
        :param path: The path + file + extension string.
        :return:
        """

        name = pathlib.Path(path).stem

        if blob.empty:
            return f'{name}: empty'

        try:
            blob.to_csv(path_or_buf=path, index=False, header=True, encoding='utf-8',
                        quoting=csv.QUOTE_NONNUMERIC)
            return f'{name}: succeeded'
        except OSError as err:
            raise ValueError(err.strerror) from err

    @staticmethod
    def read(uri: str, header: int = 0, usecols: list = None, dtype: dict = None,
             date_fields: list = None, date_format: dict = None) -> pd.DataFrame:
        """

        :param uri: The uniform resource identifier; path + file + extension string.
        :param header: The header row of the `csv` file
        :param usecols: The fields in focus
        :param dtype: Dictionary of type per field
        :param date_fields: The list of data fields, if any.
        :param date_format: The date format per date field
        :return:
        """

        if date_fields is None:
            parse_dates = False
        else:
            parse_dates = date_fields

        try:
            return pd.read_csv(filepath_or_buffer=uri, header=header, usecols=usecols, dtype=dtype,
                               encoding='utf-8', parse_dates=parse_dates, date_format=date_format)
        except ImportError:
            return pd.DataFrame()

    def api(self, uri: str, header: int = 0, usecols: list = None, dtype: dict = None,
            date_fields: list = None, date_format: dict = None) -> pd.DataFrame:
        """

        :param uri: The uniform resource identifier; path + file + extension string.
        :param header: The header row of the `csv` file
        :param usecols: The fields in focus
        :param dtype: Dictionary of type per field
        :param date_fields: The list of data fields, if any.
        :param date_format: The date format per date field
        :return:
        """

        data = pd.DataFrame()

        try:
            response = requests.head(url=uri, timeout=300)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise ValueError(f'HTTP Error: {err}') from err

        if response.status_code == 200:
            data = self.read(uri=uri, header=header, usecols=usecols,
                             dtype=dtype, date_fields=date_fields, date_format=date_format)

        return data
