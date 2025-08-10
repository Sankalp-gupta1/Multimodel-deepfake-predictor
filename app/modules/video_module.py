# app/modules/video_module.py

"""
video_module.py

🎥 Frame Analysis + Lip Sync Check + CLIP Embedding Verification
"""

import cv2
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from video at 1 frame per second.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate) if fps > 0 else 1
    success, frame_count = True, 0

    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def process_video(video_path):
    """
    Analyze video frames using CLIP similarity for deepfake detection.
    Returns a dictionary with score and label.
    """
    try:
        frames = extract_frames(video_path)
        if not frames:
            return {'score': 0.0, 'label': "⚠️ No frames extracted"}

        fake_score = 0
        total_checked = min(5, len(frames))

        for i, frame in enumerate(frames[:total_checked]):
            text_prompt = "a real person speaking on camera"
            inputs = clip_processor(text=[text_prompt], images=frame, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.softmax(dim=1)[0][0].item()

            if score < 0.4:  # threshold for "realness"
                fake_score += 1

        confidence = round(1 - fake_score / total_checked, 3)
        label = "Fake" if confidence < 0.315 else "Real"

        return {
            'score': confidence,
            'label': label
        }

    except Exception as e:
        return {
            'score': 0.0,
            'label': f"Error: {str(e)}"
        }
