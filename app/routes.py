# app/routes.py
from flask import Blueprint, render_template, request
from app.modules.voice_module import process_voice
from app.modules.video_module import process_video
from app.modules.text_module import process_text
from app.modules.nlp_module import analyze_text
from app.modules.llm_module import process_llm
from app.modules.fusion_module import fuse_results

import os

main = Blueprint('main', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_input = request.form.get('text')
        voice_file = request.files.get('voice')
        video_file = request.files.get('video')

        voice_text = ""
        video_result = {}
        nlp_result = {}
        llm_result = {}
        text_features = {}

        if voice_file and voice_file.filename != '':
            voice_path = os.path.join(UPLOAD_FOLDER, voice_file.filename)
            voice_file.save(voice_path)
            voice_text = process_voice(voice_path)

        if video_file and video_file.filename != '':
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            video_result = process_video(video_path)
        else:
            video_result = {'score': 0, 'label': 'Not provided'}

        full_text = (text_input or "") + " " + voice_text
        if full_text.strip():
            nlp_result = analyze_text(full_text)
            llm_result = process_llm(full_text)
            text_features = process_text(full_text)
        else:
            text_features = {'probability': 0, 'label': 'No text'}
            llm_result = {'contradiction_check': 'neutral', 'summary': 'No input', 'fluency_score': 'N/A'}
            nlp_result = {'vader_sentiment': 'Neutral'}

        verdict = fuse_results(text_features, {'probability': 0}, video_result, nlp_result, llm_result)

        return render_template('index.html', result=verdict)

    return render_template('index.html', result=None)