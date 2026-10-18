"""Unit tests for live-response handling and explicit Watson-outage fallback."""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    def test_angry_text_when_service_unavailable(self):
        with patch("emotion_detection.requests.post", side_effect=requests.ConnectionError):
            result = json.loads(emotion_detector("I am really mad about this"))
        predictions = result["emotionPredictions"]
        self.assertGreater(len(predictions), 0)
        emotions = predictions[0]["emotion"]
        expected = {"anger", "disgust", "fear", "joy", "sadness"}
        self.assertTrue(expected.issubset(emotions))
        for emotion in expected:
            self.assertIsInstance(emotions[emotion], (int, float))
            self.assertGreaterEqual(emotions[emotion], 0)
            self.assertLessEqual(emotions[emotion], 1)
        self.assertEqual(max(expected, key=emotions.get), "anger")
        self.assertEqual(result["analysisMode"], "offline_keyword_fallback")

    def test_live_watson_response_is_preserved(self):
        sample = '{"emotionPredictions": [{"emotion": {"anger": 0.9}}]}'
        response = Mock()
        response.text = sample
        response.raise_for_status.return_value = None
        with patch("emotion_detection.requests.post", return_value=response):
            self.assertEqual(emotion_detector("I am really mad about this"), sample)

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
