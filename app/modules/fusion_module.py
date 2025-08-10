def fuse_results(text_result, voice_result, video_result, nlp_result, llm_result):
    weights = {
        'text': 0.25,
        'voice': 0.25,
        'video': 0.30,
        'llm': 0.20
    }

    text_score = text_result.get('probability', 0)
    voice_score = voice_result.get('probability', 0)
    video_score = video_result.get('score', 0)
    llm_score = 1 if llm_result.get('contradiction_check') == 'contradiction' else 0

    final_score = (
        weights['text'] * text_score +
        weights['voice'] * voice_score +
        weights['video'] * video_score +
        weights['llm'] * llm_score
    )

    # 🔥 Apply fixed threshold
    label = "Likely Real" if final_score <= 0.3 else "Deepfake Detected"

    return {
        "final_score": round(final_score, 3),
        "label": label,
        "summary": llm_result.get("summary", "N/A"),
        "fluency_check": llm_result.get("fluency_score", "N/A"),
        "sentiment": nlp_result.get("vader_sentiment", "N/A")
    }


