"""
FastAPI Prediction Service — production-grade ML API template.
Includes: typed I/O, logging, error handling, health check, async.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048, description="Input text to classify")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class PredictResponse(BaseModel):
    request_id: str
    label: str
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


# ─── Fake Model ───────────────────────────────────────────────────────────────

class FakeClassifier:
    """Placeholder — swap in real sklearn/torch model via joblib.load()."""

    def __init__(self):
        self.labels = ["positive", "negative", "neutral"]
        logger.info("Model loaded successfully")

    def predict(self, text: str) -> tuple[str, float]:
        # Deterministic fake: long text → positive, short → negative
        score = min(len(text) / 200, 0.99)
        label = self.labels[0] if score > 0.6 else self.labels[1]
        return label, round(score, 4)


# ─── App Lifecycle ────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = FakeClassifier()
    _state["start_time"] = time.time()
    logger.info("Service started")
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="ML Prediction Service",
    description="Production-grade text classification API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Middleware: Request Logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed:.1f}ms)"
    )
    return response


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        uptime_seconds=round(time.time() - _state.get("start_time", time.time()), 1),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        label, confidence = model.predict(body.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if confidence < body.threshold:
        label = "uncertain"

    return PredictResponse(
        request_id=str(uuid.uuid4()),
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
