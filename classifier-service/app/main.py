from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from app.batcher import MicroBatcher
from app.classifier import build_real_classifier
from app.config import BATCH_WINDOW_MS, CANDIDATE_LABELS, MAX_BATCH_SIZE, MAX_QUEUE_SIZE
from app.metrics import COMMENTS_BY_LABEL, QUEUE_SIZE_GAUGE


# Indirection so tests can swap in a stub classifier before startup.
_classifier_factory = build_real_classifier


class ClassifyRequest(BaseModel):
    comment: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    comment: str
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = _classifier_factory()
    batcher = MicroBatcher(
        classifier=classifier,
        max_batch_size=MAX_BATCH_SIZE,
        batch_window_ms=BATCH_WINDOW_MS,
        max_queue_size=MAX_QUEUE_SIZE,
    )
    batcher.start()
    app.state.batcher = batcher
    # Pre-register every candidate label so its counter series exists at 0
    # from startup. Without this, a label the model never predicts (e.g.
    # "Empörung" with facebook/bart-large-mnli) has no time series and is
    # simply absent from the Grafana panel instead of showing a flat zero line.
    for label in CANDIDATE_LABELS:
        COMMENTS_BY_LABEL.labels(label=label)
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(title="NLP Classifier Service", lifespan=lifespan)


@app.middleware("http")
async def refresh_queue_gauge(request, call_next):
    if request.url.path.startswith("/metrics") and hasattr(request.app.state, "batcher"):
        QUEUE_SIZE_GAUGE.set(request.app.state.batcher.queue.qsize())
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    label, score = await app.state.batcher.submit(req.comment)
    return ClassifyResponse(comment=req.comment, label=label, score=score)


app.mount("/metrics", make_asgi_app())
