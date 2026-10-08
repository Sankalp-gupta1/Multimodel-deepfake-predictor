# Multimodal Deepfake Predictor

A research-oriented Flask prototype that combines **video, audio, text, NLP, and language-model signals** into a single multimodal deepfake-style assessment.

> This repository is an experimental prototype. Several detection components use heuristics or lightweight placeholder classifiers, so it should not be treated as a production-grade forensic deepfake detector.

## What the System Accepts

The Flask interface can process:

- text input
- an uploaded audio file
- an uploaded video file

The application routes each modality through a separate analysis module and then combines the outputs using a weighted fusion layer.

## Architecture

```text
                     ┌──────────────┐
Text input ─────────►│ Text module  │
                     └──────┬───────┘
                            │
Audio upload ─► Whisper ────┼───────┐
                            │       │
Video upload ─► CLIP/frame analysis│
                            │       │
Text ────────► NLP analysis │       │
                            │       │
Text ────────► BART + NLI ──┘       │
                                    ▼
                              Fusion module
                                    │
                                    ▼
                              Final verdict
```

## Modules

### Video Analysis

`app/modules/video_module.py`

- extracts frames with OpenCV
- loads CLIP (`openai/clip-vit-base-patch32`)
- compares frames against a real-person prompt
- derives a heuristic video score

### Voice Analysis

`app/modules/voice_module.py`

- loads Whisper
- normalizes audio to 16 kHz
- transcribes speech
- applies a simple spoof heuristic based on energy and zero-crossing rate

### Text Signal

`app/modules/text_module.py`

- cleans input text
- uses TF-IDF
- applies a small Logistic Regression classifier
- currently trains on a tiny placeholder sample inside the module

### NLP Analysis

`app/modules/nlp_module.py`

- named entity recognition with spaCy
- TextBlob sentiment
- VADER sentiment
- readability scoring

### LLM / Transformer Analysis

`app/modules/llm_module.py`

- BART summarization
- RoBERTa MNLI contradiction analysis
- simple fluency heuristic

### Fusion

`app/modules/fusion_module.py`

Combines modality scores with fixed weights and returns:

- final score
- final label
- generated summary
- fluency check
- sentiment signal

## Tech Stack

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- OpenAI Whisper
- OpenCV
- CLIP
- Librosa
- SoundFile
- Scikit-learn
- spaCy
- TextBlob
- NLTK
- textstat

## Project Structure

```text
Multimodel-deepfake-predictor/
├── app/
│   ├── modules/
│   │   ├── video_module.py
│   │   ├── voice_module.py
│   │   ├── text_module.py
│   │   ├── nlp_module.py
│   │   ├── llm_module.py
│   │   └── fusion_module.py
│   ├── templates/
│   ├── static/
│   ├── routes.py
│   └── __init__.py
├── run.py
└── README.md
```

## Installation

Create a virtual environment, then install the main dependencies:

```bash
pip install flask torch transformers opencv-python openai-whisper librosa soundfile scikit-learn spacy textblob textstat nltk numpy
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Whisper may also require FFmpeg on your system.

## Run

```bash
python run.py
```

Then open:

```text
http://127.0.0.1:8000
```

The first startup can be slow because multiple pretrained transformer models are downloaded and loaded.

## Important Limitations

This project demonstrates **multimodal system design**, not validated digital forensics.

Current limitations include:

- heuristic video scoring
- heuristic voice-spoof logic
- tiny placeholder text-training data
- fixed fusion weights and thresholds
- no benchmarked forensic accuracy claim
- high model-loading cost
- no production authentication or upload isolation

A production detector would require properly trained modality-specific models, calibrated fusion, validated datasets, adversarial testing, and rigorous evaluation.

## Author

**Sankalp Gupta**

GitHub: https://github.com/Sankalp-gupta1
