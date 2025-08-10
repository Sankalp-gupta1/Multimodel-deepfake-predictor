# app/utils/preprocessor.py

"""
Preprocessor
Cleans and standardizes text, audio, and video inputs before module processing
"""

import re
import cv2
import os
import tempfile
import moviepy.editor as mp
from pydub import AudioSegment

def clean_text(text):
    """Remove URLs, special chars, and excessive whitespace"""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9.,!? ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_audio_from_video(video_path):
    """Extract audio and convert to WAV for speech model"""
    clip = mp.VideoFileClip(video_path)
    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    return temp_audio_path

def extract_frames(video_path, fps=1):
    """Extract 1 frame per second from video for visual model"""
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    rate = vidcap.get(cv2.CAP_PROP_FPS)
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        if not success:
            break
        if int(count % rate) == 0:
            frames.append(image)
        count += 1
    vidcap.release()
    return frames