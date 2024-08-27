"""Module architecture.py"""
import os

import datasets
import ray.train.huggingface.transformers as rt
import transformers

import src.data.interface
import src.elements.variable as vr
import src.modelling.intelligence
import src.modelling.metrics
import src.modelling.numerics
import src.modelling.parameters
import src.modelling.preprocessing


class Architecture:
    """
    https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    Hyperparameters, etc.: learning rate, weight decay, the batch sizes, number of training epochs
    """

    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def exc(config: dict):
        """
        Important, this must be a static method.

        :param config:
        :return:
        """

        variable = vr.Variable()
        parameters = src.modelling.parameters.Parameters().parameters

        # Directives
        metrics = src.modelling.metrics.Metrics(parameters=parameters)
        intelligence = src.modelling.intelligence.Intelligence(parameters=parameters)

        # Data & Tokenization: Each split is converted into a T5 tokenized split.
        source: datasets.DatasetDict = src.data.interface.Interface().get_dataset()
        preprocessing = src.modelling.preprocessing.Preprocessing(parameters=parameters)
        data: datasets.DatasetDict = source.map(preprocessing.batches, batched=True)

        # For maximum steps
        numerics = src.modelling.numerics.Numerics(
            n_training_instances=data['train'].shape[0], variable=variable)

        # Arguments
        args: transformers.Seq2SeqTrainingArguments = transformers.Seq2SeqTrainingArguments(
            output_dir=variable.MODEL_OUTPUT_DIRECTORY,
            do_train=True,
            do_eval=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=variable.VALIDATE_BATCH_SIZE,
            num_train_epochs=variable.EPOCHS,
            max_steps=numerics(),
            warmup_steps=0,
            logging_dir=os.path.join(variable.MODEL_OUTPUT_DIRECTORY, '.logs'),
            no_cuda=False,
            seed=5,
            save_total_limit=2,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False
        )

        # Trainer
        trainer = transformers.Seq2SeqTrainer(
            model_init=intelligence.model, args=args,
            train_dataset=data['train'],
            eval_dataset=data['validate'],
            tokenizer=parameters.tokenizer,
            data_collator=intelligence.collator(),
            compute_metrics=metrics.exc
        )
        trainer.add_callback(rt.RayTrainReportCallback())
        trainer = rt.prepare_trainer(trainer)

        return trainer.train()
