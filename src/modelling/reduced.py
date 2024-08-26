"""Module assemble.py"""
import logging

import ray.data
import ray.train.torch
import ray.tune
import ray.tune.search.bayesopt
import transformers
import ray.train.huggingface.transformers as rt

import src.elements.variable as vr
import src.modelling.architecture
import src.modelling.settings
import src.modelling.intelligence
import src.modelling.metrics
import src.modelling.parameters
import src.modelling.numerics


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
        self.__parameters = src.modelling.parameters.Parameters().parameters

        self.__numerics = src.modelling.numerics.Numerics(data=data, variable=self.__variable)
        logging.info('max_steps: %s', self.__numerics())

        # Settings
        self.__settings = src.modelling.settings.Settings()

    def __trainer(self):
        """

        :return:
        """

        arc = src.modelling.architecture.Architecture(max_steps=self.__numerics())

        # Directives
        metrics = src.modelling.metrics.Metrics(parameters=self.__parameters)
        intelligence = src.modelling.intelligence.Intelligence(parameters=self.__parameters)

        # Trainer
        trainer = transformers.Seq2SeqTrainer(
            model_init=intelligence.model, args=arc.args,
            train_dataset=intelligence.iterable(segment='train', batch_size=self.__variable.TRAIN_BATCH_SIZE),
            eval_dataset=intelligence.iterable(segment='eval', batch_size=self.__variable.VALIDATE_BATCH_SIZE),
            tokenizer=self.__parameters.tokenizer,
            data_collator=intelligence.collator(),
            compute_metrics=metrics.exc
        )
        trainer.add_callback(rt.RayTrainReportCallback())
        trainer = rt.prepare_trainer(trainer)

        return trainer.train()

    def exc(self):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :return:
        """

        trainable = ray.train.torch.TorchTrainer(train_loop_per_worker=self.__trainer)

        tuner = ray.tune.Tuner(
            trainable,
            # param_space={
            #     'lr': self.__variable.LEARNING_RATE,
            #     'weight_decay': self.__variable.WEIGHT_DECAY,
            #     'per_device_train_batch_size': self.__variable.TRAIN_BATCH_SIZE
            # },
            
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