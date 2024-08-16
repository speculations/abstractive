"""Module architecture.py"""
import os
import logging
import transformers

import ray.train.huggingface.transformers as rt

import src.modelling.t5.metrics
import src.modelling.t5.intelligence

import src.elements.variable as vr
import src.modelling.t5.parameters as pr


class Architecture:
    """
    https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    Hyperparameters, etc.: learning rate, weight decay, the batch sizes, number of training epochs
    """

    def __init__(self):
        pass

    @staticmethod
    def exc(config: dict):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :param config:
        :return:
        """

        logging.info('Architecture:\n%s', config.keys())

        variable: vr.Variable = config.get('variable')
        parameters: pr.Parameters = config.get('parameters')

        metrics = src.modelling.t5.metrics.Metrics(parameters=parameters)
        intelligence = src.modelling.t5.intelligence.Intelligence(variable=variable, parameters=parameters)

        args: transformers.Seq2SeqTrainingArguments = transformers.Seq2SeqTrainingArguments(
            output_dir=variable.MODEL_OUTPUT_DIRECTORY,
            do_train=True,
            do_eval=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=variable.LEARNING_RATE,
            weight_decay=variable.WEIGHT_DECAY,
            per_device_train_batch_size=variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=variable.VALIDATE_BATCH_SIZE,
            num_train_epochs=variable.EPOCHS,
            max_steps=variable.MAX_STEPS,
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

        trainer = transformers.Seq2SeqTrainer(
            model_init=intelligence.model,
            args=args,
            train_dataset=intelligence.iterable(segment='train', batch_size=variable.TRAIN_BATCH_SIZE),
            eval_dataset=intelligence.iterable(segment='eval', batch_size=variable.VALIDATE_BATCH_SIZE),
            tokenizer=parameters.tokenizer,
            data_collator=intelligence.data_collator(),
            compute_metrics=metrics.exc
        )
        trainer.add_callback(rt.RayTrainReportCallback())
        trainer = rt.prepare_trainer(trainer)
        trainer.train()
