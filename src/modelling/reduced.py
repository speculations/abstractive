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

    def __init__(self, data: dict[str, ray.data.dataset.MaterializedDataset]):
        """

        :param data: The project's data; parts train, validate, test
        """

        self.__data = data
        self.__variable = vr.Variable()


        self.__numerics = src.modelling.numerics.Numerics(data=data, variable=self.__variable)
        logging.info('max_steps: %s', self.__numerics())

        # Settings
        self.__settings = src.modelling.settings.Settings()


    def exc(self):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :return:
        """

        arc = src.modelling.architecture.Architecture()

        trainable = ray.train.torch.TorchTrainer(train_loop_per_worker=arc.exc)

        tuner = ray.tune.Tuner(
            trainable,
            param_space={
                "scaling_config": ray.train.ScalingConfig(
                    num_workers=self.__variable.N_GPU, use_gpu=True,
                    trainer_resources={'CPU': self.__variable.N_CPU}),
                "datasets": {'train': self.__data['train'], 'eval': self.__data['validate']},
                "train_loop_config": {'learning_rate': self.__variable.LEARNING_RATE,
                                      'weight_decay': self.__variable.WEIGHT_DECAY,
                                      'per_device_train_batch_size': self.__variable.TRAIN_BATCH_SIZE,
                                      'max_steps': self.__numerics()}
            },
            tune_config=ray.tune.TuneConfig(
                metric='eval_loss', mode='min',
                scheduler=self.__settings.scheduler(),
                num_samples=1, reuse_actors=True,
                search_alg=ray.tune.search.bayesopt.BayesOptSearch()
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
