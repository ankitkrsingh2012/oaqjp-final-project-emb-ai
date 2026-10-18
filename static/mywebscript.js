"use strict";

async function RunSentimentAnalysis(event) {
  if (event) event.preventDefault();
  const input = document.getElementById("textToAnalyze").value;
  const output = document.getElementById("system_response");
  output.classList.remove("error");
  output.textContent = "Analyzing…";
  try {
    const response = await fetch(`/emotionDetector?textToAnalyze=${encodeURIComponent(input)}`);
    output.textContent = await response.text();
    if (!response.ok) output.classList.add("error");
  } catch (error) {
    output.classList.add("error");
    output.textContent = "Unable to contact the application. Please try again.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("emotionForm").addEventListener("submit", RunSentimentAnalysis);
});
