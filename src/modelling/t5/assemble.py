"""Module assemble.py"""
import logging
import ray.data

import ray.train.torch
import ray.tune
import ray.tune.schedulers

import src.modelling.t5.custom
import src.modelling.t5.parameters as pr
import src.modelling.t5.settings
import src.modelling.t5.architecture


class Assemble:
    """
    Assemble
    """

    def __init__(self, data: dict[str, ray.data.dataset.MaterializedDataset]):
        """

        :param data: The project's data; parts train, validate, test
        """

        self.__data = data
        self.__variable = src.modelling.t5.custom.Custom().custom
        self.__parameters = pr.Parameters()

        # The trainer & train loop configuration
        self.__arc = src.modelling.t5.architecture.Architecture()
        self.__train_loop_config = {'learning_rate': self.__variable.LEARNING_RATE, 'weight_decay': self.__variable.WEIGHT_DECAY,
                                    'per_device_train_batch_size': self.__variable.TRAIN_BATCH_SIZE}

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
            # train_loop_config=self.__train_loop_config,
            scaling_config=ray.train.ScalingConfig(
                num_workers=self.__variable.N_GPU,
                use_gpu=True, trainer_resources={'CPU': self.__variable.N_CPU}),
            datasets={
                'train': self.__data['train'],
                'eval': self.__data['validate']
            }
        )

        tuner = ray.tune.Tuner(
            trainable=trainable,
            param_space=self.__train_loop_config,
            tune_config=ray.tune.TuneConfig(
                scheduler=self.__settings.scheduler(),
                num_samples=2,
                reuse_actors=True
            ),
            run_config=ray.train.RunConfig(
                name='tuning',
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min'),
                progress_reporter=self.__settings.reporting()
            )

        )

        tuner.fit()
