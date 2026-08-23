
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from tensorflow.keras.models import load_model
import numpy as np

# Payload executes on unpickle
with open("trojan_keras_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# Load components
vectorizer = model_bundle["vectorizer"]
model_path = model_bundle["model_path"]
responses = model_bundle["responses"]

# Load the actual Keras model
model = load_model(model_path)

# Model Inference: Prompting
prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt]).toarray()
prediction = model.predict(X_input)
predicted_label = np.argmax(prediction)

print("Response:", responses[predicted_label])
