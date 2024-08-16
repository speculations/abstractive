"""Module assemble.py"""
import ray
import ray.data
import ray.train.torch
import ray.tune
import ray.tune.schedulers

import src.elements.variable as vr
import src.modelling.t5.settings


class Assemble:
    """
    Assemble
    """

    def __init__(self, data: dict[str, ray.data.dataset.MaterializedDataset],
                 variable: vr.Variable):
        """

        :param data:
        :param variable:
        """

        self.__data = data
        self.__variable = variable

        # Settings
        self.__settings = src.modelling.t5.settings.Settings(variable=self.__variable)

        # ... steps & epochs
        max_steps_per_epoch = self.__data['train'].count() // (self.__variable.TRAIN_BATCH_SIZE * self.__variable.N_GPU)
        max_steps = max_steps_per_epoch * self.__variable.EPOCHS

    def __trainable(self) -> ray.train.torch.TorchTrainer:
        """
        https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html

        :return:
        """

        return ray.train.torch.TorchTrainer(

            scaling_config=ray.train.ScalingConfig(
                num_workers=self.__variable.N_GPU,
                use_gpu=True),
            datasets={
                'train': self.__data['train'],
                'eval': self.__data['validate']
            },
            run_config=ray.train.RunConfig(
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min')
            )
        )


    def __call__(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html

        :return:
        """

        ray.tune.Tuner(
            trainable=self.__trainable,
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
                    checkpoint_score_order='min')
            )

        )