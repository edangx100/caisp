# ============================================================================
# textattack_wrapper.py
#
# The ADAPTER between our sentiment analyser and TextAttack.
#
# TextAttack does not care how a model is built. It only needs an object that
# behaves like this:
#
#       scores = my_model(["first sentence", "second sentence"])
#
# ...i.e. give it a list of strings, get back one row of class scores per
# string. That is the whole contract. This file provides it.
#
# Because TextAttack only ever sees the scores - never the weights and never
# the gradients - this is a BLACK-BOX attack. It is the same level of access
# an outsider would have against a public sentiment-analysis API: send text,
# read the confidence numbers, repeat.
# ============================================================================

import torch
from textattack.models.wrappers import ModelWrapper

# Import the model we want to attack, plus the batch scoring helper.
from simple_sentiment_analyser_copy import model, get_class_probabilities


class SentimentWrapper(ModelWrapper):
    """Makes simple_sentiment_analyser_copy.py look like a TextAttack model."""

    def __init__(self):
        # TextAttack looks for a ".model" attribute on the wrapper (it uses it
        # for logging and for a few optional features), so we point it at the
        # real HuggingFace model object.
        self.model = model

    def __call__(self, text_inputs):
        """
        text_inputs : a list of strings that TextAttack wants scored.
        returns     : a tensor shaped (number_of_texts, 3), where each row is
                      [negative, neutral, positive].

        The column order must stay fixed and must match the labels used in the
        dataset in run_attack.py, otherwise TextAttack would be checking the
        wrong number when it decides whether the attack succeeded.
        """
        # One batched forward pass for the whole list - much faster than
        # looping, and TextAttack sends a lot of candidates at once.
        probabilities = get_class_probabilities(list(text_inputs))

        # TextAttack expects a torch tensor (or numpy array) of floats.
        return torch.tensor(probabilities, dtype=torch.float32)
