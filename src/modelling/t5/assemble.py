"""Module assemble.py"""
import logging
import ray.data

import ray.train.torch
import ray.tune
import ray.tune.schedulers

import src.elements.variable as vr
import src.modelling.t5.parameters as pr
import src.modelling.t5.settings
import src.modelling.t5.architecture


class Assemble:
    """
    Assemble
    """

    def __init__(self, data: dict[str, ray.data.dataset.MaterializedDataset],
                 variable: vr.Variable, parameters: pr.Parameters):
        """

        :param data: The project's data; parts train, validate, test
        :param variable: A suite of values for machine learning
                         model development
        :param parameters: T5 specific parameters
        """

        self.__data = data
        self.__variable = variable
        self.__parameters = parameters

        # The trainer & train loop configuration
        self.__arc = src.modelling.t5.architecture.Architecture()
        self.__train_loop_config = {'variable': self.__variable, 'parameter': self.__parameters}

        # Settings
        self.__settings = src.modelling.t5.settings.Settings(variable=self.__variable, parameters=self.__parameters)

    def exc(self):
        """
        https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html

        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html

        :return:
        """

        trainable: ray.train.torch.TorchTrainer = ray.train.torch.TorchTrainer(
            self.__arc.exc,
            train_loop_config=self.__train_loop_config,
            scaling_config=ray.train.ScalingConfig(
                num_workers=self.__variable.N_GPU,
                use_gpu=True),
            datasets={
                'train': self.__data['train'],
                'eval': self.__data['validate']
            }
        )

        tuner = ray.tune.Tuner(
            trainable=trainable,
            param_space=self.__settings.param_space(),
            tune_config=ray.tune.TuneConfig(
                scheduler=self.__settings.scheduler(),
                num_samples=1
            ),
            run_config=ray.train.RunConfig(
                name='tuning',
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min'),
                progress_reporter=self.__settings.reporting()
            )

        )

        tuner.fit()
