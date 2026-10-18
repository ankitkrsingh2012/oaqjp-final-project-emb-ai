"""Integration tests against the live Watson NLP emotion detection service."""

import json
import unittest

from emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Check real predictions rather than substitute service responses."""

    def test_angry_text(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
