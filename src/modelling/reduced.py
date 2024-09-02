"""Module assemble.py"""
import logging

import ray.data
import ray.train.torch
import ray.tune
import ray.tune.schedulers as rts

import src.data.interface
import src.elements.variable as vr
import src.modelling.architecture


class Reduced:
    """
    Assemble
    """

    def __init__(self):
        """
        Constructor
        """

        self.__variable = vr.Variable()

        # Data -> data: dict[str, MaterializedDataset]
        self.__data: dict[str, ray.data.dataset.MaterializedDataset] = src.data.interface.Interface().get_rays()
        self.__max_steps_per_epoch: int = self.__data['train'].count() // (self.__variable.TRAIN_BATCH_SIZE * self.__variable.N_GPU)

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
        trainable = ray.train.torch.TorchTrainer(
            arc.exc,
            datasets={"train": self.__data["train"], "eval": self.__data["validate"]})

        # Tuner
        tuner = ray.tune.Tuner(
            trainable,
            param_space={
                "train_loop_config": {
                    'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
                    'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
                    'per_device_train_batch_size': ray.tune.grid_search([16, 32]),
                    'max_steps_per_epoch': self.__max_steps_per_epoch},
                "scaling_config": ray.train.ScalingConfig(
                    num_workers=self.__variable.N_GPU,
                    use_gpu=True,
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
                progress_reporter=ray.tune.CLIReporter(
                    parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size', 'num_train_epochs'],
                    metric_columns=['eval_loss', 'rouge1', 'rouge2', 'rougeLsum', 'median']),
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute='eval_loss',
                    checkpoint_score_order='min')
            )
        )

        return tuner.fit()
