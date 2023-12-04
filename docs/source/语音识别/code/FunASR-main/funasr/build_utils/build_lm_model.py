import logging

from funasr.train.abs_model import AbsLM
from funasr.train.abs_model import LanguageModel
from funasr.models.seq_rnn_lm import SequentialRNNLM
from funasr.models.transformer_lm import TransformerLM
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices

lm_choices = ClassChoices(
    "lm",
    classes=dict(
        seq_rnn=SequentialRNNLM,
        transformer=TransformerLM,
    ),
    type_check=AbsLM,
    default="seq_rnn",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        lm=LanguageModel,
    ),
    default="lm",
)

class_choices_list = [
    # --lm and --lm_conf
    lm_choices,
    # --model and --model_conf
    model_choices
]


def build_lm_model(args):
    # token_list
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    elif isinstance(args.token_list, (tuple, list)):
        token_list = list(args.token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    else:
        vocab_size = None

    # lm
    lm_class = lm_choices.get_class(args.lm)
    lm = lm_class(vocab_size=vocab_size, **args.lm_conf)

    args.model = args.model if hasattr(args, "model") else "lm"
    model_class = model_choices.get_class(args.model)
    model = model_class(lm=lm, vocab_size=vocab_size, **args.model_conf)

    # initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
