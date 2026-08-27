# Import necessary libraries
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "714eb0fa89d2f80546fda750413ed43d93601a13"

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_name, revision=revision)
model = DistilBertForSequenceClassification.from_pretrained(model_name, revision=revision)

# Move model to the appropriate device
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = softmax(outputs.logits, dim=1)

    # Get probabilities for positive and negative classes
    negative_prob = predictions[0][0].item()
    positive_prob = predictions[0][1].item()

    # Determine sentiment label based on probability
    label = "POSITIVE" if positive_prob > negative_prob else "NEGATIVE"
    confidence = positive_prob if label == "POSITIVE" else negative_prob

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "negative": negative_prob,
            "positive": positive_prob
        }
    }

