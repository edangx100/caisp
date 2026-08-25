# Import our sentiment classifier
from sentiment_classifier import predict_sentiment

# Sample texts to test
sample_texts = [
    "This movie is great and amazing!",
    "This was a terrible waste of time.",
    "I really enjoyed watching this film.",
    "The worst movie I've ever seen.",
    "The acting was superb and the storyline kept me engaged throughout.",
    "I found the plot confusing and the characters were poorly developed.",
    "This is one of the best films I have watched in recent years.",
    "The movie was disappointing and failed to meet my expectations.",
    "I absolutely loved the cinematography and the musical score was fantastic.",
    "The film was boring and I struggled to stay awake during the entire screening."
]

# Test the classifier on each sample
print("Testing Sentiment Classifier\n")
print("-" * 50)

for text in sample_texts:
    result = predict_sentiment(text)

    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Negative probability: {result['probabilities']['negative']:.4f}")
    print(f"Positive probability: {result['probabilities']['positive']:.4f}")
    print("-" * 50)
