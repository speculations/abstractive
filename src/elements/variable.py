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

    MAX_NEW_TOKENS: int
        [max_new_tokens](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/\
            text_generation#transformers.GenerationConfig)

    MAX_LENGTH_INPUT: int
        The maximum sequence length of the independent variable.  In the case of the California Bills data,
        the <text> key represents the independent variable.

    MAX_LENGTH_TARGET: int
        The maximum sequence length of the dependent/target variable.  In the case of the California Bills data,
        the <summary> key represents the independent variable.

    MODEL_OUTPUT_DIRECTORY: str
        A directory for model outputs

    DEVICE: str
        A string denoting graphics or central processing unit, i.e., 'cuda' or 'cpu', respectively.
    """

    TRAIN_BATCH_SIZE: int = 16
    VALIDATE_BATCH_SIZE: int = 16
    TEST_BATCH_SIZE: int = 16
    EPOCHS: int = 8
    LEARNING_RATE: float = 2e-05
    MAX_NEW_TOKENS: int = 32
    MAX_LENGTH_INPUT: int = 1024
    MAX_LENGTH_TARGET: int = 32
    MODEL_OUTPUT_DIRECTORY: str = ''
    DEVICE: str = ''
