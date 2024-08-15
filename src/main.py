"""Module main.py"""
import logging
import os
import sys

import datasets
import torch


def main() -> None:
    """
    Entry point

    :return:
        None
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Device Message
    if device == 'cuda':
        logger.info('# of %s devices: %s', device.upper(),
                    torch.cuda.device_count())
    else:
        logger.info('Device: %s', device.upper())

    # Explorations
    source: datasets.DatasetDict = src.data.source.Source().exc()
    src.modelling.interface.Interface(source=source, device=device).exc()

    # Delete Cache Points
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Activate graphics processing units
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TOKENIZERS_PARALLELISM']='true'

    # Modules
    import src.functions.cache
    import src.data.source
    import src.modelling.interface

    main()
