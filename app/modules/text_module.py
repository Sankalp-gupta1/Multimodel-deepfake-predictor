# app/modules/text_module.py

"""
text_module.py

✍️ Plain Text Deepfake Content Detection
Detects suspicious linguistic patterns or generated text signals.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import re

# Dummy trained model load (Replace with real NLP-based classifier if available)
# model = joblib.load('models/text_fake_model.pkl')  # Optional future path

vectorizer = TfidfVectorizer(max_features=5000)
classifier = LogisticRegression()

# Placeholder training (replace with real model in production)
def fake_train():
    fake_texts = ["This video shows a shocking revelation", "You won't believe what happens next"]
    real_texts = ["The meeting starts at 10am", "He submitted his assignment"]
    X = vectorizer.fit_transform(fake_texts + real_texts)
    y = [1]*len(fake_texts) + [0]*len(real_texts)
    classifier.fit(X, y)

# Train dummy model
fake_train()

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def process_text(text):
    """
    Converts input text to feature vector and predicts probability of being fake.
    """
    try:
        text = clean_text(text)
        X = vectorizer.transform([text])
        prob = classifier.predict_proba(X)[0][1]  # Probability of class 1 (fake)

        return {
            "probability": prob,
            "label": "Fake" if prob > 0.312 else "Real"
        }
    except Exception as e:
        return {
            "probability": 0,
            "label": f"Error: {str(e)}"
        }