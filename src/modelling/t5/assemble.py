"""Module assemble.py"""
import ray
import ray.data
import ray.train.huggingface.transformers as rt
import ray.train.torch
import ray.tune
import ray.tune.schedulers
import transformers

import src.elements.variable as vr
import src.modelling.t5.intelligence
import src.modelling.t5.metrics
import src.modelling.t5.parameters as pr
import src.modelling.t5.settings


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

        # Settings
        self.__metrics = src.modelling.t5.metrics.Metrics(parameters=self.__parameters)
        self.__settings = src.modelling.t5.settings.Settings(variable=self.__variable, parameters=self.__parameters)
        self.__intelligence = src.modelling.t5.intelligence.Intelligence(variable=variable, parameters=parameters)

    def __trainer(self):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :return:
        """

        trainer = transformers.Seq2SeqTrainer(
            model_init=self.__intelligence.model,
            args=self.__settings.args(),
            train_dataset=self.__intelligence.iterable(segment='train', batch_size=self.__variable.TRAIN_BATCH_SIZE),
            eval_dataset=self.__intelligence.iterable(segment='eval', batch_size=self.__variable.VALIDATE_BATCH_SIZE),
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__intelligence.data_collator(),
            compute_metrics=self.__metrics.exc
        )
        trainer.add_callback(rt.RayTrainReportCallback())
        trainer = rt.prepare_trainer(trainer)
        trainer.train()

    def __trainable(self) -> ray.train.torch.TorchTrainer:
        """
        https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
        https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html

        :return:
        """

        return ray.train.torch.TorchTrainer(
            self.__trainer,
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

        tuner = ray.tune.Tuner(
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

        return tuner.fit()