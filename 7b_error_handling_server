"""Flask web interface for the EmotionDetection package."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector


app = Flask(__name__)


@app.route("/")
def home():
    """Display the emotion-detection web application."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Analyze the supplied text and display the result."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again!", 400

    result = emotion_detector(text_to_analyze)
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400
    score_text = ", ".join(
        f"'{name}': {result[name]}"
        for name in ("anger", "disgust", "fear", "joy", "sadness")
    )
    return (
        f"For the given statement, the system response is {score_text}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
