"""Module assemble.py"""
import logging

import ray.data
import ray.train.torch
import ray.tune
import ray.tune.search.bayesopt
import ray.tune.schedulers as rts

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
            train_loop_config={
                'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
                'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
                'per_device_train_batch_size': ray.tune.grid_search([16, 32]),
                'max_steps': numerics()},
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
                metric='eval_loss',
                mode='min',
                scheduler=rts.ASHAScheduler(time_attr='training_iteration', max_t=25, grace_period=3),
                num_samples=1, reuse_actors=True
            ),
            run_config=ray.train.RunConfig(
                name='tuning',
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min')
            )
        )

        return tuner.fit()
