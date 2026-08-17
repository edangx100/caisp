# Import required libraries
import torch
from textattack.models.wrappers import ModelWrapper
from sentiment_classifier import predict_sentiment
# Import the actual model for TextAttack compatibility
from sentiment_classifier import model

# Create a wrapper class for TextAttack
class SentimentWrapper(ModelWrapper):
    def __init__(self):
        # Set the model attribute that TextAttack expects
        self.model = model

    def __call__(self, text_inputs):
        # Convert input texts to model predictions using our custom classifier
        outputs = []
        for text in text_inputs:
            # Get prediction from our custom sentiment classifier
            result = predict_sentiment(text)

            # Convert prediction to probability format expected by TextAttack
            # TextAttack expects [negative_prob, positive_prob]
            negative_prob = result['probabilities']['negative']
            positive_prob = result['probabilities']['positive']
            probs = [negative_prob, positive_prob]

            outputs.append(probs)

        # Return tensor of predictions
        return torch.tensor(outputs)


