from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pathlib import Path
from src.predict import SentimentPredictor


# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI(title="Sentiment Analysis API")


# -----------------------------
# Enable CORS (for development)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load Model Once
# -----------------------------
predictor = SentimentPredictor()


# -----------------------------
# Handle Frontend Path Properly
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# -----------------------------
# Request Schema
# -----------------------------
class TextInput(BaseModel):
    text: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/ui")
def serve_ui():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    sentiment = predictor.predict(input_data.text)
    return {
        "input_text": input_data.text,
        "predicted_sentiment": sentiment
    }
