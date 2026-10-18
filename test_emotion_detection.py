"""Unit tests for the five required emotions.

These tests simulate a Watson connection failure and verify that the
application's documented offline fallback produces the expected labels.
They are not tests of Watson's live prediction accuracy.
"""

import unittest
from unittest.mock import patch

import requests

from emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Check each dominant emotion and the six required output fields."""

    def check_emotion(self, sentence, expected_emotion):
        """Test one sentence without depending on Watson availability."""
        with patch(
            "emotion_detection.requests.post",
            side_effect=requests.ConnectionError("Service unavailable"),
        ):
            result = emotion_detector(sentence)

        scores = ("anger", "disgust", "fear", "joy", "sadness")
        self.assertEqual(set(result), set(scores) | {"dominant_emotion"})
        for emotion in scores:
            self.assertIsInstance(result[emotion], (int, float))
        self.assertEqual(result["dominant_emotion"], expected_emotion)
        self.assertEqual(result[expected_emotion], max(result[name] for name in scores))

    def test_joy(self):
        self.check_emotion("I am glad this happened", "joy")

    def test_anger(self):
        self.check_emotion("I am really mad about this", "anger")

    def test_disgust(self):
        self.check_emotion("I feel disgusted just hearing about this", "disgust")

    def test_sadness(self):
        self.check_emotion("I am so sad about this", "sadness")

    def test_fear(self):
        self.check_emotion("I am really afraid that this will happen", "fear")


if __name__ == "__main__":
    unittest.main(verbosity=2)
