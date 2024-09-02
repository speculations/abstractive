"""Module intelligence.py"""
import logging

import transformers

import src.elements.variable as vr
import src.modelling.arguments as ag


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerFast):
        """

        :param tokenizer: T5 specific
        """

        self.__tokenizer = tokenizer

        self.__variable = vr.Variable()
        self.__arguments = ag.Arguments()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__tokenizer, model=self.__arguments.checkpoint)

    def model(self):
        """

        :return:
        """

        config = transformers.GenerationConfig.from_pretrained(
            pretrained_model_name=self.__arguments.checkpoint, **{'max_new_tokens': self.__variable.MAX_NEW_TOKENS})
        self.__logger.info('max_length: %s', config.max_length)
        self.__logger.info('max_new_tokens: %s', config.max_new_tokens)

        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__arguments.checkpoint, config=config
        )
