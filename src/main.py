"""Module main.py"""
import logging
import os
import sys

import torch
import ray


def main() -> None:
    """

    Notes
    -----
    https://docs.ray.io/en/latest/cluster/usage-stats.html

    :return:
        None
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ray
    ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # Device Message
    if device == 'cuda':
        logger.info('# of %s devices: %s', device.upper(),
                    torch.cuda.device_count())
    else:
        logger.info('Device: %s', device.upper())

    # Explorations
    src.modelling.steps.Steps().exc()

    # Delete Cache Points
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    # Paths
    # noinspection DuplicatedCode
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
    os.environ['RAY_USAGE_STATS_ENABLED']='0'

    # Modules
    import src.functions.cache
    import src.modelling.steps

    main()
