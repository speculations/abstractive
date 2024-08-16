"""Module settings"""
import os

import ray
import ray.tune
import ray.tune.schedulers as rts
import transformers

import src.elements.variable as vr


class Settings:
    """
    Class Settings
    """

    def __init__(self, variable: vr.Variable):
        """

        :param variable: A suite of values for machine learning
                         model development
        """

        self.__variable = variable

        # Re-visit
        self.__perturbation_interval = 2

    def param_space(self):
        """
        Initialises

        :return:
        """

        return {
            'learning_rate': 0.05,
            'weight_decay': 0.05,
            'per_device_train_batch_size': 2*self.__variable.TRAIN_BATCH_SIZE,
            'num_train_epochs': ray.tune.choice([2, 3, 4, 5])
        }

    def scheduler(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html

        Leads on from hp_space

        :return:
        """

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_loss', mode='min',
            perturbation_interval=self.__perturbation_interval,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
                'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
                'per_device_train_batch_size': [16, 32, 64]
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

    def args(self) -> transformers.Seq2SeqTrainingArguments:
        """
        https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
        Opt foci: learning rate, weight decay, the batch sizes, number of training epochs

        :return:
        """

        return transformers.Seq2SeqTrainingArguments(
            output_dir=self.__variable.MODEL_OUTPUT_DIRECTORY,
            do_train=True,
            do_eval=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            weight_decay=0.01,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALIDATE_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            logging_dir=os.path.join(self.__variable.MODEL_OUTPUT_DIRECTORY, '.logs'),
            no_cuda=False,
            seed=5,
            save_total_limit=2,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False
        )
