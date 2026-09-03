# ============================================================================
# simple_sentiment_analyser_copy.py
#
# A copy of simple_sentiment_analyser.py that is also usable as the TARGET
# ("victim model") of a TextAttack adversarial attack.
#
# Everything from the original is kept as-is. Two things were added so that
# TextAttack can talk to this model:
#
#   1. get_class_probabilities()  -> returns the RAW probabilities for all
#      three classes, in the model's own fixed order, and prints nothing.
#      The original analyze_sentiment() sorts the scores and returns only the
#      positive score, which loses the information an attacker needs.
#
#   2. A safe fallback for the label list, so the file still imports if the
#      GitHub label mapping cannot be downloaded.
#
# The original interactive loop at the bottom still works:
#     python simple_sentiment_analyser_copy.py
# ============================================================================

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import torch
from scipy.special import softmax
import csv
import urllib.request

# The model we are analysing (and later attacking).
# Pinning "revision_id" to an exact commit hash means we always load the exact
# same weights - important for security work, so results are reproducible.
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
revision_id = "daefdd1f6ae931839bce4d0f3db0a1a4265cd50f"

model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=revision_id)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision_id)

# Put the model in evaluation mode: this turns off dropout, so the same input
# always gives the same output. An attack that got random answers back could
# not tell whether a word swap actually helped.
model.eval()

# download label mapping
labels = []
mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
try:
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
except Exception as download_error:
    # Offline fallback. This is the same list the download returns, and the
    # order matters: it is the order of the model's three output neurons.
    print(f"Could not download label mapping ({download_error}); using built-in labels.")
    labels = ["negative", "neutral", "positive"]

# Handy constants so other files do not have to remember the numbers.
# Output index 0 = negative, 1 = neutral, 2 = positive.
LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE = 0, 1, 2


# Preprocess text (Anonymize usernames and urls)
def preprocess(text):
    new_text = []
    # Separate all words by using spaces
    for t in text.split(" "):
        # If a word starts with @, and is longer than 1 character, replace it with @user
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        # If a word starts with http, replace it with http
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# ---------------------------------------------------------------------------
# NEW: the function TextAttack needs.
#
# It takes a LIST of texts and returns a table of probabilities:
#
#       [[neg, neu, pos],      <- probabilities for texts[0]
#        [neg, neu, pos],      <- probabilities for texts[1]
#        ...              ]
#
# Two rules make this attack-friendly:
#   * The column order NEVER changes (negative, neutral, positive), so the
#     attacker can compare scores between two different sentences.
#   * All texts go through the model in ONE batch. TextAttack asks the model
#     thousands of questions per sentence, so one-at-a-time would be very slow.
# ---------------------------------------------------------------------------
def get_class_probabilities(texts):
    # Apply the same @user / http cleanup the normal path uses, so the attack
    # sees exactly the model that real users would hit.
    cleaned = [preprocess(t) for t in texts]

    # Turn the words into token IDs. padding=True makes every sentence in the
    # batch the same length; truncation caps very long inputs.
    encoded_input = tokenizer(cleaned, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # torch.no_grad() = "just predict, do not prepare for training".
    # It saves memory and time.
    with torch.no_grad():
        output = model(**encoded_input)

    # output.logits are raw, unbounded numbers. softmax turns each row into
    # probabilities that add up to 1.0.
    scores = output.logits.detach().numpy()
    return softmax(scores, axis=1)


# Function to analyze sentiment and return and print the results
def analyze_sentiment(text):
    if text is None or text == "":
        return "Exiting: Received empty string or a None object"
    text.strip().replace("\n", "").replace("\r", "")
    text = preprocess(text)  # Preprocess the text
    encoded_input = tokenizer(text, return_tensors='pt')  # Tokenize the text
    output = model(**encoded_input)  # Get model output [** is used to unpack the encoded_input]
    scores = output[0][0].detach().numpy()  # Get the sentiment scores
    scores = softmax(scores)  # Apply softmax to the scores to normalize them
    ranking = np.argsort(scores)  # Get the ranking of the scores
    ranking = ranking[::-1]  # Reverse the order to get highest to lowest
    results = []  # List to store the sentiment analysis results
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]  # Get the label for the sentiment
        s = scores[ranking[i]]  # Get the score for the sentiment
        results.append((l, np.round(float(s), 4)))  # Add the result to the list
    # Print the sentiment results
    for label, score in results:
        print(f"{label}: {score}")
    positive_score = next((score for label, score in results if label == 'positive'), 0)

    if positive_score > 0.5:
        print("The sentiment is positive.")

    #return results  # Return the list of sentiment results
    return positive_score


# ---------------------------------------------------------------------------
# NEW: a small helper that answers "what does the model think of this text?"
# in one line. Used by run_attack.py to double-check TextAttack's own verdict.
# ---------------------------------------------------------------------------
def predict_label(text):
    probs = get_class_probabilities([text])[0]   # one text in, one row out
    winner = int(np.argmax(probs))               # index of the highest score
    return {
        "label": labels[winner],                 # e.g. "negative"
        "label_id": winner,                      # e.g. 0
        "confidence": float(probs[winner]),      # how sure the model is
        "probabilities": {                       # the full picture
            "negative": float(probs[LABEL_NEGATIVE]),
            "neutral": float(probs[LABEL_NEUTRAL]),
            "positive": float(probs[LABEL_POSITIVE]),
        },
    }


if __name__ == "__main__":
    while True:
        print("+" *50)
        text = input("\033[92mEnter your text: Type 'X' or 'x' to exit: \033[0m").strip()
        if text in ['X', 'x']:
            print("Exiting.")
            break
        else:
            # Analyze the sentiment of the text
            sentiment = analyze_sentiment(text)

            print("Returned Sentiment:", sentiment)
