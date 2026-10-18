"""Unit tests for the Task 3 emotion detector output format."""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    def test_angry_text_when_service_unavailable(self):
        with patch("emotion_detection.requests.post", side_effect=requests.ConnectionError):
            result = emotion_detector("I am really mad about this")
        self.assertEqual(
            set(result),
            {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"},
        )
        self.assertEqual(result["dominant_emotion"], "anger")
        self.assertEqual(result["anger"], 1.0)

    def test_live_watson_response_is_formatted(self):
        sample = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.05, "disgust": 0.01, "fear": 0.03,
                    "joy": 0.86, "sadness": 0.05,
                }
            }]
        }
        response = Mock()
        response.text = json.dumps(sample)
        response.raise_for_status.return_value = None
        with patch("emotion_detection.requests.post", return_value=response):
            result = emotion_detector("I am so happy I am doing this")
        self.assertEqual(result, {
            "anger": 0.05,
            "disgust": 0.01,
            "fear": 0.03,
            "joy": 0.86,
            "sadness": 0.05,
            "dominant_emotion": "joy",
        })

    def test_unknown_text_fallback_has_no_dominant_emotion(self):
        with patch("emotion_detection.requests.post", side_effect=requests.Timeout):
            result = emotion_detector("xyz")
        self.assertIsNone(result["dominant_emotion"])

    def test_non_transient_http_error_is_not_silenced(self):
        response = requests.Response()
        response.status_code = 401
        with patch("emotion_detection.requests.post") as post:
            post.return_value.raise_for_status.side_effect = requests.HTTPError(
                "401 Unauthorized", response=response
            )
            with self.assertRaises(requests.HTTPError):
                emotion_detector("I am really mad about this")


if __name__ == "__main__":
    unittest.main(verbosity=2)
