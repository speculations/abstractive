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
import src.data.interface


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

        # Data -> data: dict[str, MaterializedDataset]
        data = src.data.interface.Interface().get_rays()

        # Maximum steps
        numerics = src.modelling.numerics.Numerics(
            n_training_instances=data['train'].count(), variable=self.__variable)
        logging.info('maximum steps: %s', numerics())

        # From Hugging Face Trainer -> Ray Trainer
        trainable = ray.train.torch.TorchTrainer(
            arc.exc,
            train_loop_config={"learning_rate": 2e-5, "max_steps": numerics()},
            datasets={"train": data["train"], "eval": data["validate"]})

        # Tuner
        tuner = ray.tune.Tuner(
            trainable,
            param_space={
                "scaling_config": ray.train.ScalingConfig(
                    num_workers=self.__variable.N_GPU, use_gpu=True,
                    trainer_resources={'CPU': self.__variable.N_CPU})
            },
            tune_config=ray.tune.TuneConfig(
                metric='eval_loss', mode='min',
                # scheduler=self.__settings.scheduler(),
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
