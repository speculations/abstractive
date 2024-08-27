"""Module assemble.py"""
import logging

import ray.data
import ray.train.torch
import ray.tune
import ray.tune.search.bayesopt

import src.elements.variable as vr
import src.modelling.architecture
import src.modelling.numerics
import src.modelling.settings


class Reduced:
    """
    Assemble
    """

    def __init__(self):
        """
        Constructor
        """

        self.__variable = vr.Variable()

        # Settings
        self.__settings = src.modelling.settings.Settings()


    def exc(self):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        A note vis-à-vis TuneConfig.  Search algorithms, e.g.,
            search_alg=ray.tune.search.bayesopt.BayesOptSearch()
        cannot be used with PopulationBasedTraining schedulers.

        :return:
        """

        arc = src.modelling.architecture.Architecture()

        # From Hugging Face Trainer -> Ray Trainer
        # trainable = ray.train.torch.TorchTrainer(train_loop_per_worker=arc.exc)

        # Tuner
        tuner = ray.tune.Tuner(
            arc.exc,
            param_space={
                "scaling_config": ray.train.ScalingConfig(
                    num_workers=self.__variable.N_GPU, use_gpu=True,
                    trainer_resources={'CPU': self.__variable.N_CPU}),
                "learning_rate": self.__variable.LEARNING_RATE,
                "weight_decay": self.__variable.WEIGHT_DECAY,
                "per_device_train_batch_size": self.__variable.TRAIN_BATCH_SIZE
            },
            tune_config=ray.tune.TuneConfig(
                metric='eval_loss', mode='min',
                scheduler=self.__settings.scheduler(),
                num_samples=1, reuse_actors=True
            ),
            run_config=ray.train.RunConfig(
                name='tuning',
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min')
            )
        )

        return tuner.fit()
