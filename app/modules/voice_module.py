# app/modules/voice_module.py

"""
voice_module.py
🎤 Voice-to-Text + Fake Voice Detection
Using OpenAI Whisper for transcription and additional detection logic for voice spoofing.
"""

import whisper
import torch
import numpy as np
import librosa
import soundfile as sf
import os

# Load Whisper Model (base for speed, large for accuracy)
MODEL_NAME = "base"
whisper_model = whisper.load_model(MODEL_NAME)

def process_voice(audio_path):
    """
    Transcribes voice and returns the extracted text.
    Optionally runs spoof detection (advanced extension).
    """
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        tmp_wav_path = audio_path.replace(".mp3", ".wav")
        sf.write(tmp_wav_path, audio, sr)

        # Transcribe
        result = whisper_model.transcribe(tmp_wav_path)
        text = result.get("text", "")

        # Optional: run spoof detection
        is_fake = detect_spoof(audio, sr)

        return text if not is_fake else f"⚠️ Potential Deepfake Voice Detected: {text}"

    except Exception as e:
        return f"Error processing audio: {str(e)}"

def detect_spoof(audio, sr):
    """
    Dummy spoof detector using basic features.
    Replace with ASVspoof or similar models in production.
    """
    energy = np.sum(np.square(audio)) / len(audio)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Thresholds based on heuristics / experimental tuning
    if energy < 0.0001 or zcr > 0.25:
        return True
    return False

