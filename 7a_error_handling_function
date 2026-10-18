"""Emotion detection using Watson NLP.

The emotion_detector function returns a dictionary containing five emotion
scores and the dominant emotion. During a temporary Watson outage, an
approximate keyword-based fallback is used (not a Watson prediction).
"""

import json
import os
import re

import requests


_MODEL_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
_EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")
_EMOTION_KEYWORDS = {
    "anger": {"mad", "angry", "furious", "enraged", "irritated", "annoyed", "rage"},
    "disgust": {"disgusted", "disgusting", "gross", "revolting", "repulsive", "nasty"},
    "fear": {"afraid", "scared", "fearful", "terrified", "worried", "anxious", "panic"},
    "joy": {"happy", "joyful", "delighted", "excited", "glad", "wonderful", "love"},
    "sadness": {"sad", "heartbroken", "miserable", "unhappy", "grieving", "crying"},
}


def _format_emotions(scores):
    """Return the six fields required by Task 3."""
    formatted = {name: scores[name] for name in _EMOTION_NAMES}
    highest_score = max(formatted.values())
    formatted["dominant_emotion"] = (
        max(formatted, key=formatted.get) if highest_score > 0 else None
    )
    return formatted


def _local_keyword_fallback(text):
    """Approximate emotion scores only when Watson is temporarily unavailable."""
    words = re.findall(r"[a-z]+", text.lower())
    counts = {
        emotion: sum(word in keywords for word in words)
        for emotion, keywords in _EMOTION_KEYWORDS.items()
    }
    total = sum(counts.values())
    scores = {
        emotion: round(count / total, 6) if total else 0.0
        for emotion, count in counts.items()
    }
    return _format_emotions(scores)


def _invalid_input_result():
    """Return a well-formed response for HTTP 400 invalid-input errors."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyse):
    """Return anger, disgust, fear, joy, sadness, and dominant_emotion."""
    if not isinstance(text_to_analyse, str):
        raise TypeError("text_to_analyse must be a string")

    if os.environ.get("EMOTION_OFFLINE_ONLY") == "1":
        return _local_keyword_fallback(text_to_analyse)

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(
            _MODEL_URL, headers=headers, json=payload, timeout=(3, 8)
        )
        response.raise_for_status()
        response_data = json.loads(response.text)
        emotion_scores = response_data["emotionPredictions"][0]["emotion"]
        return _format_emotions(emotion_scores)
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status == 400:
            return _invalid_input_result()
        if status not in (500, 502, 503, 504):
            raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    return _local_keyword_fallback(text_to_analyse)
