"""
Sample FastAPI application for testing exercises.
A simple sentiment prediction API.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# In-memory "model" storage (populated at lifespan startup)
# ---------------------------------------------------------------------------
ml_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load a dummy model on startup."""
    ml_models["sentiment"] = _DummySentimentModel()
    yield
    ml_models.clear()


app = FastAPI(title="Sentiment API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    score: float
    text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


# ---------------------------------------------------------------------------
# Dummy model (replaced in tests via DI or mocking)
# ---------------------------------------------------------------------------
class _DummySentimentModel:
    """Keyword-based dummy sentiment model."""

    POSITIVE_WORDS = {"good", "great", "amazing", "love", "excellent", "fantastic"}
    NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "worst", "horrible"}

    def predict(self, text: str) -> tuple[str, float]:
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        if pos > neg:
            return "positive", min(0.5 + pos * 0.1, 0.99)
        elif neg > pos:
            return "negative", min(0.5 + neg * 0.1, 0.99)
        return "neutral", 0.5


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded="sentiment" in ml_models,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(req: PredictRequest):
    model = ml_models.get("sentiment")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label, score = model.predict(req.text)
    return PredictResponse(label=label, score=score, text=req.text)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    model = ml_models.get("sentiment")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = []
    for text in req.texts:
        label, score = model.predict(text)
        predictions.append(PredictResponse(label=label, score=score, text=text))
    return BatchPredictResponse(predictions=predictions)
