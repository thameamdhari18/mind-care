from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from textblob import TextBlob

app = FastAPI()

# CORS Middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try loading the summarization model
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("⚠️ Could not load summarization model:", e)
    summarizer = None

# Input model
class Entry(BaseModel):
    text: str

# Helper to guess mood from sentiment
def get_mood(score):
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    return "Neutral"

@app.post("/summarize/")
async def summarize_entry(entry: Entry):
    text = entry.text.strip()

    # Basic fallback if model not available
    if summarizer is None or not text:
        summary = text if text else "No input provided."
    else:
        try:
            # Ensure minimum length to trigger summarization
            if len(text.split()) < 15:
                text += " " + text  # repeat text to allow summarization
            summary = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"(Failed to summarize: {e})"

    # Sentiment Analysis
    blob = TextBlob(text)
    sentiment_score = round(blob.sentiment.polarity, 2)
    mood = get_mood(sentiment_score)

    return {
        "summary": summary.strip(),
        "mood": mood,
        "sentiment_score": sentiment_score
    }
