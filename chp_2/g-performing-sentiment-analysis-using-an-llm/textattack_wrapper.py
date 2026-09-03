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
#
# ----------------------------------------------------------------------------
# HOW THIS COMPARES TO THE CHAPTER 1 WRAPPER
# (chp_2/1-attacking_llm_model_using_textattack/textattack_wrapper.py)
#
# This file is written to be as close as possible to the chapter 1 version, so
# you can read the two side by side. Identical in both: the class name, the
# ModelWrapper base class, the self.model attribute, the __call__ signature,
# the for-loop over text_inputs, and the torch.tensor(outputs) return.
#
# There is exactly ONE real difference, and it is forced on us by the model:
#
#       CHAPTER 1  ->  2 classes  ->  [negative, positive]
#       THIS FILE  ->  3 classes  ->  [negative, neutral, positive]
#
# Chapter 1 attacks distilbert-base-uncased-finetuned-sst-2-english, which was
# trained to answer only "positive or negative?". This chapter attacks
# cardiffnlp/twitter-roberta-base-sentiment, which was trained on tweets and
# has a third answer available: "neutral". So every row we hand back has three
# numbers in it rather than two.
#
# That single change ripples out into one other place. A label in the dataset
# is really "the column number of the correct answer", so in run_attack.py a
# positive sentence is labelled 2, not 1:
#
#       column:      0          1         2
#       chapter 1:   negative   positive     <- positive is label 1
#       this file:   negative   neutral   positive   <- positive is label 2
#
# Get that wrong and nothing crashes - which is what makes it dangerous.
# TextAttack would simply compare the wrong number when deciding whether an
# attack had succeeded, and every result you got back would be nonsense.
# ----------------------------------------------------------------------------
# ============================================================================

# Import required libraries
import torch
from textattack.models.wrappers import ModelWrapper
from simple_sentiment_analyser_copy import predict_sentiment
# Import the actual model for TextAttack compatibility
from simple_sentiment_analyser_copy import model


# Create a wrapper class for TextAttack
class SentimentWrapper(ModelWrapper):
    def __init__(self):
        # Set the model attribute that TextAttack expects
        self.model = model

    def __call__(self, text_inputs):
        # Convert input texts to model predictions using our custom classifier
        outputs = []
        for text in text_inputs:
            # Get prediction from our custom sentiment analyser
            result = predict_sentiment(text)

            # Convert prediction to probability format expected by TextAttack.
            # TextAttack expects [negative_prob, neutral_prob, positive_prob].
            # The order must never change: TextAttack matches these positions
            # against the labels in the dataset.
            negative_prob = result['probabilities']['negative']
            neutral_prob = result['probabilities']['neutral']
            positive_prob = result['probabilities']['positive']
            probs = [negative_prob, neutral_prob, positive_prob]

            outputs.append(probs)

        # Return tensor of predictions
        return torch.tensor(outputs)
