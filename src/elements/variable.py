"""Module variable.py"""
import typing


class Variable(typing.NamedTuple):
    """
    Generic attributes for machine learning model development

    Attributes
    ----------
    TRAIN_BATCH_SIZE: int
        The batch size for the training stage; default 16.

    VALIDATE_BATCH_SIZE: int
        The batch size for the validation evaluation stage; default 16.

    TEST_BATCH_SIZE: int
        The batch size for the test evaluation stage; default 16.

    EPOCHS: int
        The number of epochs: default 8.

    LEARNING_RATE: float
        The learning rate; default 2e-05.

    WEIGHT_DECAY: float
        https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/trainer
            #transformers.Seq2SeqTrainingArguments.weight_decay
        https://arxiv.org/abs/1711.05101

    MAX_NEW_TOKENS: int
        [max_new_tokens](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/\
            text_generation#transformers.GenerationConfig)

    MAX_LENGTH_INPUT: int
        The maximum sequence length of the independent variable.  In the case of the California Bills data,
        the <text> key represents the independent variable.

    MAX_LENGTH_TARGET: int
        The maximum sequence length of the dependent/target variable.  In the case of the California Bills data,
        the <summary> key represents the independent variable.

    N_CPU: int
        An initial number of central processing units for computation

    N_GPU: int
        The number of graphics processing units

    MAX_STEPS: int
        https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/trainer
            #transformers.Seq2SeqTrainingArguments.max_steps

    MODEL_OUTPUT_DIRECTORY: str
        A directory for model outputs
    """

    TRAIN_BATCH_SIZE: int = 16
    VALIDATE_BATCH_SIZE: int = 16
    TEST_BATCH_SIZE: int = 16
    EPOCHS: int = 8
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 0.05
    MAX_NEW_TOKENS: int = 32
    MAX_LENGTH_INPUT: int = 1024
    MAX_LENGTH_TARGET: int = 32
    N_CPU: int = 8
    N_GPU: int = 1
    MAX_STEPS: int = -1
    MODEL_OUTPUT_DIRECTORY: str = ''
