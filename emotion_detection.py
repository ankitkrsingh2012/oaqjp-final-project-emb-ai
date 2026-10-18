"""Emotion detection with Watson NLP and an explicit outage fallback.

Normally the Watson NLP service supplies the emotion scores. If the service
cannot be reached or returns a temporary server error, a small, deterministic
keyword-based approximation is used. The returned JSON identifies this mode.
"""

import json
import re

import requests


_MODEL_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
_EMOTION_KEYWORDS = {
    "anger": {"mad", "angry", "furious", "enraged", "irritated", "annoyed", "rage"},
    "disgust": {"disgusted", "disgusting", "gross", "revolting", "repulsive", "nasty"},
    "fear": {"afraid", "scared", "fearful", "terrified", "worried", "anxious", "panic"},
    "joy": {"happy", "joyful", "delighted", "excited", "glad", "wonderful", "love"},
    "sadness": {"sad", "heartbroken", "miserable", "unhappy", "grieving", "crying"},
}


def _local_keyword_fallback(text):
    """Return an approximate prediction; NEVER represent it as a Watson result."""
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
    return json.dumps({
        "emotionPredictions": [{"emotion": scores}],
        "analysisMode": "offline_keyword_fallback",
        "note": "Approximate keyword scores; Watson service unavailable",
    })


def emotion_detector(text_to_analyse):
    """Return emotion predictions as JSON, falling back during service outages."""
    if not isinstance(text_to_analyse, str):
        raise TypeError("text_to_analyse must be a string")

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyse}}
    try:
        response = requests.post(
            _MODEL_URL, headers=headers, json=payload, timeout=(3, 8)
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status not in (500, 502, 503, 504):
            raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    return _local_keyword_fallback(text_to_analyse)
