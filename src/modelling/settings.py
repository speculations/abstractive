"""Module settings"""
import ray
import ray.tune
import ray.tune.schedulers as rts

import src.elements.variable as vr


class Settings:
    """
    Class Settings
    """

    def __init__(self):
        """
        Constructor
        """

        self.__variable = vr.Variable()

    def initiating(self):
        """
        Initialises

        :return:
        """

        return {
            'learning_rate': self.__variable.LEARNING_RATE,
            'weight_decay': self.__variable.WEIGHT_DECAY,
            'per_device_train_batch_size': self.__variable.TRAIN_BATCH_SIZE
        }

    def scheduler(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html

        Leads on from initiating

        :return:
        """

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            perturbation_interval=self.__variable.PERTURBATION_INTERVAL,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
                'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
                'per_device_train_batch_size': [16, 32]
            },
            quantile_fraction=0.25,
            resample_probability=0.25
        )

    @staticmethod
    def reporting():
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html

        :return:
        """

        return ray.tune.CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size', 'num_train_epochs'],
            metric_columns=['eval_loss', 'rouge1', 'rouge2', 'rougeLsum', 'median']
        )
