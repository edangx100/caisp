import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# ML inference
vectorizer = model_bundle["vectorizer"]
classifier = model_bundle["classifier"]

prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt])
response = classifier.predict(X_input)[0]
print("Response:", response)
