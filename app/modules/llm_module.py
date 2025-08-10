from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_labels = ["entailment", "neutral", "contradiction"]

def score_fluency(text):
    sentences = text.split('.')
    long_sentences = [s for s in sentences if len(s.strip().split()) > 20]
    if len(long_sentences) > 3:
        return "⚠️ Possible LLM Hallucination (excessive fluency)"
    return "✅ Text appears coherent"

def summarize_text(text):
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return "Error generating summary."

def detect_contradiction(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        predicted_class_id = torch.argmax(logits).item()
        return nli_labels[predicted_class_id]

def process_llm(text):
    result = {}
    result['summary'] = summarize_text(text)
    result['contradiction_check'] = detect_contradiction(text, text)
    result['fluency_score'] = score_fluency(text)
    return result
