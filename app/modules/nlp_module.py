# app/modules/nlp_module.py

"""
nlp_module.py

🧠 NLP Analysis: NER + Sentiment + Readability + Emotion Tags
"""

import spacy
from textblob import TextBlob
from textstat import flesch_reading_ease
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Load models
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

def analyze_text(text):
    """
    Performs multiple NLP tasks:
    - Named Entity Recognition (NER)
    - Sentiment Score (TextBlob + Vader)
    - Readability Score
    Returns dictionary of results
    """
    try:
        result = {}

        # Named Entities
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        result['entities'] = entities

        # TextBlob Polarity
        blob = TextBlob(text)
        result['textblob_sentiment'] = blob.sentiment.polarity

        # VADER Sentiment
        result['vader_sentiment'] = sia.polarity_scores(text)['compound']

        # Readability
        result['readability'] = flesch_reading_ease(text)

        return result

    except Exception as e:
        return {"error": str(e)}