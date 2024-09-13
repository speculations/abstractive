"""Module architecture.py"""
import logging
import os

import ray.train
import ray.train.huggingface.transformers as rt
import transformers

import src.data.interface
import src.elements.variable as vr
import src.modelling.intelligence
import src.modelling.metrics
import src.modelling.parameters
import src.modelling.preprocessing
import src.modelling.arguments as ag


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
        Important, this must be a static method.  Additionally, initialise the model, metrics, and tokenizer
        within this function

        :return:
        """

        logging.info(config)

        variable = vr.Variable()
        arguments = ag.Arguments()

        # Re-designing
        tokenizer = src.modelling.parameters.Parameters()()

        # Metric & Model
        metrics = src.modelling.metrics.Metrics(tokenizer=tokenizer)
        intelligence = src.modelling.intelligence.Intelligence(tokenizer=tokenizer)

        # Data & Tokens
        training = ray.train.get_dataset_shard("train")
        evaluating = ray.train.get_dataset_shard("eval")
        tokens = src.modelling.preprocessing.Preprocessing(tokenizer=tokenizer)

        # Arguments
        args: transformers.Seq2SeqTrainingArguments = transformers.Seq2SeqTrainingArguments(
            output_dir=arguments.MODEL_OUTPUT_DIRECTORY,
            do_train=True,
            do_eval=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=config.get('learning_rate', 5e-3),
            weight_decay=config.get('weight_decay', 0.0),
            per_device_train_batch_size=variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=variable.VALIDATE_BATCH_SIZE,
            num_train_epochs=variable.EPOCHS,
            max_steps=config.get('max_steps_per_epoch') * variable.EPOCHS,
            warmup_steps=0,
            logging_dir=os.path.join(arguments.MODEL_OUTPUT_DIRECTORY, '.logs'),
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
            train_dataset=tokens.iterables(part=training, batch_size=variable.TRAIN_BATCH_SIZE),
            eval_dataset=tokens.iterables(part=evaluating, batch_size=variable.VALIDATE_BATCH_SIZE),
            tokenizer=tokenizer,
            data_collator=intelligence.collator(),
            compute_metrics=metrics.exc
        )
        trainer.add_callback(rt.RayTrainReportCallback())
        trainer = rt.prepare_trainer(trainer)

        trainer.train()
