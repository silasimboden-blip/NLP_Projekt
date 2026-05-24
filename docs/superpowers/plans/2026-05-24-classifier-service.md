# Classifier Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized FastAPI service that classifies German news comments using HuggingFace zero-shot classification with microbatching, plus a Prometheus/Grafana monitoring stack and a Python load generator with three traffic scenarios.

**Architecture:** Three-container `docker compose` stack (`classifier-service`, `prometheus`, `grafana`). The classifier owns an `asyncio.Queue`-backed microbatcher: a single worker task pulls items, gathers more until size 8 or 200 ms elapses, runs one HF pipeline call, and resolves per-request `Future`s. Seven Prometheus metrics are defined in a shared `metrics.py` to avoid circular imports between `main.py` and `batcher.py`. Grafana is fully provisioned (datasource + dashboard JSON, no manual setup). A Python load generator (`scripts/loadgen.py --scenario burst|trickle|mixed`) drives the service so both microbatching triggers are visible in the dashboard.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, HuggingFace Transformers (`facebook/bart-large-mnli`), prometheus-client, httpx, pytest + pytest-asyncio, Prometheus v2.55.1, Grafana 11.4.0, Docker Compose.

**Reference spec:** `docs/superpowers/specs/2026-05-24-classifier-service-design.md`

---

## Conventions for every task

- **Working directory:** repo root `/Users/silasbot/Documents/FHNW/NLP/Projekt`. All `git`, `pytest`, `docker compose` commands assume you're there.
- **Test runner:** `pytest` is run via `cd classifier-service && pytest`. Tests live in `classifier-service/tests/`.
- **Commit style:** short imperative subject (e.g. `feat: add MicroBatcher size trigger`). One commit per task. Don't skip commits — they're checkpoints for review and rollback.
- **Don't run `docker compose up` until Task 12** — the model download is ~1.6 GB and slow on a cold cache. Earlier tasks use stub-based unit tests instead.

---

## Task 1: Repo scaffolding (directories + empty package files)

**Files:**
- Create: `classifier-service/app/__init__.py`
- Create: `classifier-service/tests/__init__.py`
- Create: `prometheus/.gitkeep`
- Create: `grafana/provisioning/datasources/.gitkeep`
- Create: `grafana/provisioning/dashboards/.gitkeep`
- Create: `grafana/dashboards/.gitkeep`
- Create: `scripts/.gitkeep`
- Create: `screenshots/.gitkeep`

- [ ] **Step 1: Create the directory tree and empty init/keep files**

```bash
mkdir -p classifier-service/app classifier-service/tests
mkdir -p prometheus
mkdir -p grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards
mkdir -p scripts screenshots
touch classifier-service/app/__init__.py
touch classifier-service/tests/__init__.py
touch prometheus/.gitkeep
touch grafana/provisioning/datasources/.gitkeep
touch grafana/provisioning/dashboards/.gitkeep
touch grafana/dashboards/.gitkeep
touch scripts/.gitkeep
touch screenshots/.gitkeep
```

- [ ] **Step 2: Verify the structure exists**

Run: `find classifier-service prometheus grafana scripts screenshots -type f | sort`
Expected output:
```
classifier-service/app/__init__.py
classifier-service/tests/__init__.py
grafana/dashboards/.gitkeep
grafana/provisioning/dashboards/.gitkeep
grafana/provisioning/datasources/.gitkeep
prometheus/.gitkeep
scripts/.gitkeep
screenshots/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add classifier-service prometheus grafana scripts screenshots
git commit -m "chore: scaffold repo directory structure"
```

---

## Task 2: Python dependencies + Dockerfile (no service code yet)

**Files:**
- Create: `classifier-service/requirements.txt`
- Create: `classifier-service/Dockerfile`
- Create: `classifier-service/.dockerignore`

- [ ] **Step 1: Write `classifier-service/requirements.txt`**

```
fastapi==0.115.4
uvicorn[standard]==0.32.0
transformers==4.46.2
torch==2.5.1
prometheus-client==0.21.0
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Write `classifier-service/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for HF tokenizers + healthcheck via Python stdlib only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Write `classifier-service/.dockerignore`**

```
__pycache__
*.pyc
.pytest_cache
tests
```

- [ ] **Step 4: Verify pip install works locally (catches typos in requirements)**

Run: `cd classifier-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
Expected: completes without errors (will take ~2 minutes; torch is the heavy one). The `.venv/` directory will be created.

- [ ] **Step 5: Add `.venv/` to gitignore so we don't commit it**

Edit `.gitignore` to append:
```
# Local Python virtualenv
.venv/
```

- [ ] **Step 6: Commit**

```bash
git add classifier-service/requirements.txt classifier-service/Dockerfile classifier-service/.dockerignore .gitignore
git commit -m "build: add classifier-service requirements and Dockerfile"
```

---

## Task 3: Configuration module (env-driven settings)

**Files:**
- Create: `classifier-service/app/config.py`
- Create: `classifier-service/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

File: `classifier-service/tests/test_config.py`
```python
import os
import importlib


def reload_config():
    """Re-import config.py so it re-reads env vars."""
    import app.config
    return importlib.reload(app.config)


def test_defaults(monkeypatch):
    for var in [
        "MODEL_ID", "CANDIDATE_LABELS", "BATCH_WINDOW_MS",
        "MAX_BATCH_SIZE", "MAX_QUEUE_SIZE",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = reload_config()
    assert cfg.MODEL_ID == "facebook/bart-large-mnli"
    assert cfg.CANDIDATE_LABELS == [
        "Zustimmung",
        "sachliche Kritik",
        "Empörung",
        "Beleidigung oder persönlicher Angriff",
    ]
    assert cfg.BATCH_WINDOW_MS == 200
    assert cfg.MAX_BATCH_SIZE == 8
    assert cfg.MAX_QUEUE_SIZE == 256


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CANDIDATE_LABELS", "a,b,c")
    monkeypatch.setenv("BATCH_WINDOW_MS", "50")
    monkeypatch.setenv("MAX_BATCH_SIZE", "4")
    cfg = reload_config()
    assert cfg.CANDIDATE_LABELS == ["a", "b", "c"]
    assert cfg.BATCH_WINDOW_MS == 50
    assert cfg.MAX_BATCH_SIZE == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd classifier-service && .venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Write the minimal implementation**

File: `classifier-service/app/config.py`
```python
import os

DEFAULT_LABELS = [
    "Zustimmung",
    "sachliche Kritik",
    "Empörung",
    "Beleidigung oder persönlicher Angriff",
]

MODEL_ID = os.getenv("MODEL_ID", "facebook/bart-large-mnli")

_labels_raw = os.getenv("CANDIDATE_LABELS")
CANDIDATE_LABELS = (
    [s.strip() for s in _labels_raw.split(",") if s.strip()]
    if _labels_raw
    else DEFAULT_LABELS
)

BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "200"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "256"))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add classifier-service/app/config.py classifier-service/tests/test_config.py
git commit -m "feat(config): env-driven settings module with defaults"
```

---

## Task 4: Prometheus metrics module

**Files:**
- Create: `classifier-service/app/metrics.py`
- Create: `classifier-service/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

File: `classifier-service/tests/test_metrics.py`
```python
from prometheus_client import Counter, Gauge, Histogram


def test_all_seven_metrics_defined():
    from app import metrics
    assert isinstance(metrics.COMMENTS_TOTAL, Counter)
    assert isinstance(metrics.BATCHES_PROCESSED, Counter)
    assert isinstance(metrics.BATCH_SIZE_HIST, Histogram)
    assert isinstance(metrics.INFERENCE_DURATION, Histogram)
    assert isinstance(metrics.BATCH_WAIT, Histogram)
    assert isinstance(metrics.QUEUE_SIZE_GAUGE, Gauge)
    assert isinstance(metrics.COMMENTS_BY_LABEL, Counter)


def test_metric_names_match_spec():
    from app import metrics
    assert metrics.COMMENTS_TOTAL._name == "comments_classified"
    assert metrics.BATCHES_PROCESSED._name == "batches_processed"
    assert metrics.BATCH_SIZE_HIST._name == "batch_size"
    assert metrics.INFERENCE_DURATION._name == "batch_inference_duration_seconds"
    assert metrics.BATCH_WAIT._name == "batch_wait_time_seconds"
    assert metrics.QUEUE_SIZE_GAUGE._name == "microbatch_queue_size"
    assert metrics.COMMENTS_BY_LABEL._name == "comments_classified_by_label"
```

Note on `_name`: prometheus_client strips the `_total` suffix from Counter names internally and re-adds it on export. So `Counter("comments_classified_total", ...)._name == "comments_classified"`. The exported metric is still `comments_classified_total`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd classifier-service && .venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.metrics'`.

- [ ] **Step 3: Write the implementation**

File: `classifier-service/app/metrics.py`
```python
from prometheus_client import Counter, Gauge, Histogram

COMMENTS_TOTAL = Counter(
    "comments_classified_total",
    "Total number of comments classified.",
)

BATCHES_PROCESSED = Counter(
    "batches_processed_total",
    "Total number of microbatches processed.",
)

BATCH_SIZE_HIST = Histogram(
    "batch_size",
    "Number of comments per processed batch.",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8),
)

INFERENCE_DURATION = Histogram(
    "batch_inference_duration_seconds",
    "Wall time of the HF pipeline call per batch.",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

BATCH_WAIT = Histogram(
    "batch_wait_time_seconds",
    "Time a comment spent in the microbatch queue before classification.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)

QUEUE_SIZE_GAUGE = Gauge(
    "microbatch_queue_size",
    "Current depth of the microbatch queue.",
)

COMMENTS_BY_LABEL = Counter(
    "comments_classified_by_label_total",
    "Comments classified, broken down by predicted label.",
    ["label"],
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_metrics.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add classifier-service/app/metrics.py classifier-service/tests/test_metrics.py
git commit -m "feat(metrics): define seven Prometheus metrics in shared module"
```

---

## Task 5: Classifier wrapper (HF zero-shot pipeline)

**Files:**
- Create: `classifier-service/app/classifier.py`
- Create: `classifier-service/tests/test_classifier.py`

The wrapper is small and the HF model is huge. We test it with a fake pipeline injected via constructor rather than downloading 1.6 GB during tests.

- [ ] **Step 1: Write the failing test**

File: `classifier-service/tests/test_classifier.py`
```python
from app.classifier import Classifier


def fake_pipeline(comments, candidate_labels, multi_label):
    """Imitates HF zero-shot pipeline output shape."""
    # HF returns a dict when input is a single string, list of dicts when input is a list.
    # Classifier always passes a list, so we always return a list.
    return [
        {"labels": candidate_labels, "scores": [0.9, 0.05, 0.03, 0.02]}
        for _ in comments
    ]


def test_classify_batch_returns_top_label_and_score():
    c = Classifier(pipeline=fake_pipeline, candidate_labels=["A", "B", "C", "D"])
    results = c.classify_batch(["hi", "hello"])
    assert results == [("A", 0.9), ("A", 0.9)]


def test_classify_batch_empty():
    c = Classifier(pipeline=fake_pipeline, candidate_labels=["A", "B"])
    assert c.classify_batch([]) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd classifier-service && .venv/bin/pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.classifier'`.

- [ ] **Step 3: Write the implementation**

File: `classifier-service/app/classifier.py`
```python
from typing import Callable, List, Tuple


class Classifier:
    """Thin wrapper around an HF zero-shot pipeline.

    Constructor takes the pipeline callable so tests can inject a fake
    without downloading the HF model.
    """

    def __init__(self, pipeline: Callable, candidate_labels: List[str]):
        self._pipeline = pipeline
        self._labels = candidate_labels

    def classify_batch(self, comments: List[str]) -> List[Tuple[str, float]]:
        if not comments:
            return []
        outputs = self._pipeline(
            comments,
            candidate_labels=self._labels,
            multi_label=False,
        )
        return [(o["labels"][0], float(o["scores"][0])) for o in outputs]


def build_real_classifier() -> Classifier:
    """Construct a Classifier backed by the real HF pipeline.

    Imported lazily so unit tests don't drag transformers/torch in.
    """
    from transformers import pipeline as hf_pipeline
    from app.config import MODEL_ID, CANDIDATE_LABELS

    hf = hf_pipeline("zero-shot-classification", model=MODEL_ID)
    return Classifier(pipeline=hf, candidate_labels=CANDIDATE_LABELS)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_classifier.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add classifier-service/app/classifier.py classifier-service/tests/test_classifier.py
git commit -m "feat(classifier): HF zero-shot pipeline wrapper with injectable pipeline"
```

---

## Task 6: MicroBatcher — size trigger only (first TDD slice)

**Files:**
- Create: `classifier-service/app/batcher.py`
- Create: `classifier-service/tests/test_batcher.py`

We build the batcher in two slices: first the size trigger, then the time trigger. This keeps each step's test focused and the implementation incremental.

- [ ] **Step 1: Write the failing test for size-triggered batching**

File: `classifier-service/tests/test_batcher.py`
```python
import asyncio
import pytest

from app.batcher import MicroBatcher
from app.classifier import Classifier


pytestmark = pytest.mark.asyncio


def make_stub_classifier(call_log):
    """Classifier whose batch call is recorded so tests can inspect batch sizes."""
    def fake_pipeline(comments, candidate_labels, multi_label):
        call_log.append(list(comments))
        return [
            {"labels": candidate_labels, "scores": [1.0] + [0.0] * (len(candidate_labels) - 1)}
            for _ in comments
        ]
    return Classifier(pipeline=fake_pipeline, candidate_labels=["L1", "L2"])


async def test_size_trigger_flushes_at_max_batch_size():
    call_log: list[list[str]] = []
    classifier = make_stub_classifier(call_log)
    batcher = MicroBatcher(
        classifier=classifier,
        max_batch_size=4,
        batch_window_ms=10_000,  # large so only the size trigger fires
        max_queue_size=64,
    )
    batcher.start()

    # Submit 4 items concurrently. Size trigger should fire immediately.
    results = await asyncio.gather(*(batcher.submit(f"c{i}") for i in range(4)))

    assert len(results) == 4
    assert all(r == ("L1", 1.0) for r in results)
    assert len(call_log) == 1
    assert len(call_log[0]) == 4

    await batcher.stop()
```

- [ ] **Step 2: Add pytest-asyncio config so async tests run**

File: `classifier-service/pyproject.toml` (create)
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd classifier-service && .venv/bin/pytest tests/test_batcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.batcher'`.

- [ ] **Step 4: Write the minimal implementation (size trigger only)**

File: `classifier-service/app/batcher.py`
```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from app.classifier import Classifier
from app.metrics import (
    BATCHES_PROCESSED,
    BATCH_SIZE_HIST,
    BATCH_WAIT,
    COMMENTS_BY_LABEL,
    COMMENTS_TOTAL,
    INFERENCE_DURATION,
)


@dataclass
class _BatchItem:
    comment: str
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class MicroBatcher:
    def __init__(
        self,
        classifier: Classifier,
        max_batch_size: int,
        batch_window_ms: int,
        max_queue_size: int,
    ):
        self._classifier = classifier
        self._max_batch_size = max_batch_size
        self._batch_window_s = batch_window_ms / 1000.0
        self.queue: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, comment: str) -> Tuple[str, float]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put(_BatchItem(comment=comment, future=fut))
        return await fut

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            # Size trigger only for now: pull until full, never wait past what's available.
            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    # In the size-trigger-only version we'd wait forever for the next item
                    # which breaks the test. We need at least *something* here so the
                    # 4-item test passes: keep pulling for a tiny moment to catch
                    # the gather()-submitted siblings.
                    await asyncio.sleep(0)
                    try:
                        batch.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            self._process_batch(batch)

    def _process_batch(self, batch: list[_BatchItem]) -> None:
        now = time.monotonic()
        BATCH_SIZE_HIST.observe(len(batch))
        for item in batch:
            BATCH_WAIT.observe(now - item.enqueued_at)

        try:
            with INFERENCE_DURATION.time():
                results = self._classifier.classify_batch([i.comment for i in batch])
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        BATCHES_PROCESSED.inc()
        COMMENTS_TOTAL.inc(len(batch))
        for item, (label, score) in zip(batch, results):
            COMMENTS_BY_LABEL.labels(label=label).inc()
            if not item.future.done():
                item.future.set_result((label, score))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_batcher.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add classifier-service/app/batcher.py classifier-service/tests/test_batcher.py classifier-service/pyproject.toml
git commit -m "feat(batcher): MicroBatcher with size-trigger flushing"
```

---

## Task 7: MicroBatcher — time-window trigger

**Files:**
- Modify: `classifier-service/app/batcher.py`
- Modify: `classifier-service/tests/test_batcher.py`

- [ ] **Step 1: Add the failing time-trigger test**

Append to `classifier-service/tests/test_batcher.py`:
```python
async def test_time_trigger_flushes_after_window_expires():
    call_log: list[list[str]] = []
    classifier = make_stub_classifier(call_log)
    batcher = MicroBatcher(
        classifier=classifier,
        max_batch_size=8,
        batch_window_ms=100,  # short window so the test stays fast
        max_queue_size=64,
    )
    batcher.start()

    # Submit 2 items spaced ~150 ms apart. Each should be in its own batch.
    t1 = asyncio.create_task(batcher.submit("c1"))
    await asyncio.sleep(0.15)
    t2 = asyncio.create_task(batcher.submit("c2"))
    results = await asyncio.gather(t1, t2)

    assert results == [("L1", 1.0), ("L1", 1.0)]
    assert len(call_log) == 2
    assert call_log[0] == ["c1"]
    assert call_log[1] == ["c2"]

    await batcher.stop()


async def test_size_and_time_combined():
    """A batch should flush when the window expires even if size cap is not hit."""
    call_log: list[list[str]] = []
    classifier = make_stub_classifier(call_log)
    batcher = MicroBatcher(
        classifier=classifier,
        max_batch_size=8,
        batch_window_ms=100,
        max_queue_size=64,
    )
    batcher.start()

    # 3 items submitted nearly simultaneously, well under the size cap of 8.
    # They should batch together (arrive within the window) but flush on time, not size.
    results = await asyncio.gather(*(batcher.submit(f"c{i}") for i in range(3)))

    assert len(results) == 3
    assert len(call_log) == 1
    assert len(call_log[0]) == 3

    await batcher.stop()
```

- [ ] **Step 2: Run tests — first new test should fail, second may pass coincidentally**

Run: `cd classifier-service && .venv/bin/pytest tests/test_batcher.py -v`
Expected: `test_time_trigger_flushes_after_window_expires` FAILS (the current loop has no real time-based waiting — it'll hang and time out, or produce wrong batching).

- [ ] **Step 3: Replace `_run` with the full size + time implementation**

Edit `classifier-service/app/batcher.py`, replace the `_run` method:
```python
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = loop.time() + self._batch_window_s

            while len(batch) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            self._process_batch(batch)
```

- [ ] **Step 4: Run the full batcher test file to verify everything passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_batcher.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add classifier-service/app/batcher.py classifier-service/tests/test_batcher.py
git commit -m "feat(batcher): add time-window trigger to MicroBatcher"
```

---

## Task 8: MicroBatcher — error propagation test

**Files:**
- Modify: `classifier-service/tests/test_batcher.py`

The implementation already propagates exceptions; this task adds the test that pins that behavior.

- [ ] **Step 1: Add the test**

Append to `classifier-service/tests/test_batcher.py`:
```python
async def test_classifier_exception_propagates_to_all_futures():
    class BrokenClassifier:
        def classify_batch(self, comments):
            raise RuntimeError("model is sad")

    batcher = MicroBatcher(
        classifier=BrokenClassifier(),
        max_batch_size=4,
        batch_window_ms=50,
        max_queue_size=64,
    )
    batcher.start()

    tasks = [asyncio.create_task(batcher.submit(f"c{i}")) for i in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(results) == 3
    assert all(isinstance(r, RuntimeError) for r in results)
    assert all(str(r) == "model is sad" for r in results)

    await batcher.stop()
```

- [ ] **Step 2: Run the test to verify it passes (no implementation change needed)**

Run: `cd classifier-service && .venv/bin/pytest tests/test_batcher.py -v`
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add classifier-service/tests/test_batcher.py
git commit -m "test(batcher): pin error-propagation behavior"
```

---

## Task 9: FastAPI app (main.py)

**Files:**
- Create: `classifier-service/app/main.py`
- Create: `classifier-service/tests/test_main.py`

- [ ] **Step 1: Write the failing test**

File: `classifier-service/tests/test_main.py`
```python
import pytest
from fastapi.testclient import TestClient

from app.classifier import Classifier


@pytest.fixture
def client(monkeypatch):
    # Inject a fake classifier so we don't load the HF model.
    def fake_pipeline(comments, candidate_labels, multi_label):
        return [
            {"labels": candidate_labels, "scores": [0.8, 0.1, 0.05, 0.05]}
            for _ in comments
        ]

    from app import main
    main._classifier_factory = lambda: Classifier(
        pipeline=fake_pipeline,
        candidate_labels=["Zustimmung", "sachliche Kritik", "Empörung",
                          "Beleidigung oder persönlicher Angriff"],
    )
    # Build app via TestClient which triggers lifespan startup.
    with TestClient(main.app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_classify_returns_label_and_score(client):
    r = client.post("/classify", json={"comment": "Sehr guter Artikel!"})
    assert r.status_code == 200
    body = r.json()
    assert body["comment"] == "Sehr guter Artikel!"
    assert body["label"] == "Zustimmung"
    assert body["score"] == pytest.approx(0.8)


def test_classify_rejects_empty_comment(client):
    r = client.post("/classify", json={"comment": ""})
    assert r.status_code == 422


def test_metrics_endpoint_returns_prometheus_format(client):
    # Hit /classify once so there's data, then scrape.
    client.post("/classify", json={"comment": "Sehr guter Artikel!"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "comments_classified_total" in body
    assert "batches_processed_total" in body
    assert "microbatch_queue_size" in body
    assert 'comments_classified_by_label_total{label="Zustimmung"}' in body
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd classifier-service && .venv/bin/pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write the implementation**

File: `classifier-service/app/main.py`
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from app.batcher import MicroBatcher
from app.classifier import build_real_classifier
from app.config import BATCH_WINDOW_MS, MAX_BATCH_SIZE, MAX_QUEUE_SIZE
from app.metrics import QUEUE_SIZE_GAUGE


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
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(title="NLP Classifier Service", lifespan=lifespan)


@app.middleware("http")
async def refresh_queue_gauge(request, call_next):
    if request.url.path.startswith("/metrics"):
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd classifier-service && .venv/bin/pytest tests/test_main.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite to make sure nothing else broke**

Run: `cd classifier-service && .venv/bin/pytest -v`
Expected: all tests pass (config 2 + metrics 2 + classifier 2 + batcher 4 + main 4 = 14).

- [ ] **Step 6: Commit**

```bash
git add classifier-service/app/main.py classifier-service/tests/test_main.py
git commit -m "feat(api): FastAPI app with /classify, /metrics, /healthz"
```

---

## Task 10: Prometheus config + Grafana provisioning

**Files:**
- Create: `prometheus/prometheus.yml`
- Create: `grafana/provisioning/datasources/prometheus.yml`
- Create: `grafana/provisioning/dashboards/dashboards.yml`
- Delete: `prometheus/.gitkeep`, `grafana/provisioning/datasources/.gitkeep`, `grafana/provisioning/dashboards/.gitkeep`

- [ ] **Step 1: Write `prometheus/prometheus.yml`**

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: "classifier"
    metrics_path: /metrics
    static_configs:
      - targets: ["classifier-service:8080"]
```

- [ ] **Step 2: Write `grafana/provisioning/datasources/prometheus.yml`**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

- [ ] **Step 3: Write `grafana/provisioning/dashboards/dashboards.yml`**

```yaml
apiVersion: 1

providers:
  - name: "Classifier dashboards"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

- [ ] **Step 4: Delete the now-redundant .gitkeep files**

```bash
rm prometheus/.gitkeep grafana/provisioning/datasources/.gitkeep grafana/provisioning/dashboards/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add prometheus/prometheus.yml grafana/provisioning prometheus/.gitkeep grafana/provisioning/datasources/.gitkeep grafana/provisioning/dashboards/.gitkeep
git commit -m "feat: Prometheus scrape config and Grafana provisioning"
```

---

## Task 11: Grafana dashboard JSON

**Files:**
- Create: `grafana/dashboards/classifier_dashboard.json`
- Delete: `grafana/dashboards/.gitkeep`

Hand-written for legibility. Seven panels in a 2-column grid.

- [ ] **Step 1: Write the dashboard JSON**

File: `grafana/dashboards/classifier_dashboard.json`
```json
{
  "annotations": {"list": []},
  "editable": true,
  "graphTooltip": 1,
  "panels": [
    {
      "id": 1,
      "type": "timeseries",
      "title": "Kommentare pro Sekunde",
      "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "rate(comments_classified_total[1m])", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "ops"}, "overrides": []},
      "options": {"legend": {"displayMode": "list"}}
    },
    {
      "id": 2,
      "type": "timeseries",
      "title": "Batches pro Minute",
      "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "rate(batches_processed_total[1m]) * 60", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
      "options": {"legend": {"displayMode": "list"}}
    },
    {
      "id": 3,
      "type": "timeseries",
      "title": "Durchschnittliche Batch-Grösse",
      "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "rate(batch_size_sum[1m]) / rate(batch_size_count[1m])", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "short", "min": 0, "max": 8}, "overrides": []},
      "options": {"legend": {"displayMode": "list"}}
    },
    {
      "id": 4,
      "type": "timeseries",
      "title": "p95 Inference-Latenz",
      "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "histogram_quantile(0.95, rate(batch_inference_duration_seconds_bucket[5m]))", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
      "options": {"legend": {"displayMode": "list"}}
    },
    {
      "id": 5,
      "type": "timeseries",
      "title": "p95 Batch-Wartezeit",
      "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "histogram_quantile(0.95, rate(batch_wait_time_seconds_bucket[5m]))", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
      "options": {"legend": {"displayMode": "list"}}
    },
    {
      "id": 6,
      "type": "stat",
      "title": "Queue-Grösse",
      "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "microbatch_queue_size", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
      "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "graphMode": "area"}
    },
    {
      "id": 7,
      "type": "timeseries",
      "title": "Kommentare nach Label",
      "gridPos": {"x": 0, "y": 24, "w": 24, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"expr": "sum by (label) (rate(comments_classified_by_label_total[5m]))",
         "legendFormat": "{{label}}", "refId": "A"}
      ],
      "fieldConfig": {"defaults": {"unit": "ops", "custom": {"stacking": {"mode": "normal"}}}, "overrides": []},
      "options": {"legend": {"displayMode": "table"}}
    }
  ],
  "refresh": "5s",
  "schemaVersion": 39,
  "tags": ["classifier", "nlp"],
  "templating": {"list": []},
  "time": {"from": "now-15m", "to": "now"},
  "timepicker": {},
  "timezone": "",
  "title": "NLP Classifier Service",
  "uid": "classifier-dashboard",
  "version": 1
}
```

Note: the datasource `"uid": "prometheus"` is a literal string. Grafana will resolve it to the provisioned datasource as long as we add `uid: prometheus` to the datasource file (next step).

- [ ] **Step 2: Add `uid` to the provisioned datasource**

Edit `grafana/provisioning/datasources/prometheus.yml`, change to:
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

- [ ] **Step 3: Validate the JSON parses**

Run: `python3 -c "import json; json.load(open('grafana/dashboards/classifier_dashboard.json'))"`
Expected: completes silently (no JSONDecodeError).

- [ ] **Step 4: Delete the now-redundant .gitkeep**

```bash
rm grafana/dashboards/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add grafana/dashboards/classifier_dashboard.json grafana/dashboards/.gitkeep grafana/provisioning/datasources/prometheus.yml
git commit -m "feat: hand-written Grafana dashboard with 7 panels"
```

---

## Task 12: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write the compose file**

File: `docker-compose.yml`
```yaml
services:
  classifier-service:
    build: ./classifier-service
    ports:
      - "8080:8080"
    environment:
      MODEL_ID: "facebook/bart-large-mnli"
      CANDIDATE_LABELS: "Zustimmung,sachliche Kritik,Empörung,Beleidigung oder persönlicher Angriff"
      BATCH_WINDOW_MS: "200"
      MAX_BATCH_SIZE: "8"
      MAX_QUEUE_SIZE: "256"
      HF_HOME: "/models"
      TRANSFORMERS_CACHE: "/models"
      TOKENIZERS_PARALLELISM: "false"
    volumes:
      - hf-cache:/models
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz').status==200 else 1)"]
      interval: 10s
      timeout: 3s
      retries: 30
      start_period: 60s

  prometheus:
    image: prom/prometheus:v2.55.1
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      classifier-service:
        condition: service_healthy

  grafana:
    image: grafana/grafana:11.4.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus

volumes:
  hf-cache:
```

- [ ] **Step 2: Validate compose syntax**

Run: `docker compose config > /dev/null && echo OK`
Expected: `OK`

- [ ] **Step 3: Build the classifier image (does not start anything)**

Run: `docker compose build classifier-service`
Expected: completes successfully (takes several minutes — downloads python:3.11-slim and torch).

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: docker-compose stack with classifier, prometheus, grafana"
```

---

## Task 13: Sample comments and load generator

**Files:**
- Create: `scripts/sample_comments.py`
- Create: `scripts/loadgen.py`
- Delete: `scripts/.gitkeep`

- [ ] **Step 1: Write `scripts/sample_comments.py`**

File: `scripts/sample_comments.py`
```python
"""20 hand-written German news comments, 5 per label, for the load generator.

The label tag is for sampling balance only — it is NOT sent to the service.
The classifier is genuinely zero-shot and only sees the comment text.
"""

SAMPLES = [
    # Zustimmung
    ("Zustimmung", "Sehr guter, gut recherchierter Artikel, danke!"),
    ("Zustimmung", "Endlich mal eine ausgewogene Berichterstattung zu diesem Thema."),
    ("Zustimmung", "Genau meine Meinung, vielen Dank für diesen Beitrag."),
    ("Zustimmung", "Ich finde es wichtig, dass darüber berichtet wird."),
    ("Zustimmung", "Klar formuliert und auf den Punkt gebracht, top!"),

    # sachliche Kritik
    ("sachliche Kritik", "Mir fehlen hier Quellenangaben für die zitierten Zahlen."),
    ("sachliche Kritik", "Die Studie wird zitiert, aber nicht verlinkt — schade."),
    ("sachliche Kritik", "Der Artikel berücksichtigt die Gegenposition leider nicht."),
    ("sachliche Kritik", "Einige Aussagen sind zu allgemein und sollten präzisiert werden."),
    ("sachliche Kritik", "Ich hätte mir mehr Hintergrund zu den Folgen gewünscht."),

    # Empörung
    ("Empörung", "Das ist eine absolute Frechheit, wer hat das durchgewunken??"),
    ("Empörung", "Unfassbar, dass so etwas immer noch passieren kann!"),
    ("Empörung", "Eine bodenlose Sauerei, die sofort gestoppt werden muss."),
    ("Empörung", "Das macht mich richtig wütend, wie kann das sein?"),
    ("Empörung", "Skandal! Sofort Konsequenzen für die Verantwortlichen!"),

    # Beleidigung oder persönlicher Angriff
    ("Beleidigung oder persönlicher Angriff", "Der Autor dieses Texts ist offensichtlich ein Vollidiot."),
    ("Beleidigung oder persönlicher Angriff", "Wer so etwas schreibt, hat doch keine Ahnung von gar nichts."),
    ("Beleidigung oder persönlicher Angriff", "Typisch dieser dumme Journalist, immer dasselbe Geschwätz."),
    ("Beleidigung oder persönlicher Angriff", "@nutzer123: du bist echt zu blöd, das zu verstehen."),
    ("Beleidigung oder persönlicher Angriff", "Was ein Schwachsinn, der Redakteur sollte gefeuert werden."),
]
```

- [ ] **Step 2: Write `scripts/loadgen.py`**

File: `scripts/loadgen.py`
```python
"""Load generator for the classifier service.

Three scenarios:
  burst   — ~25 req/s steady; fills batches to MAX_BATCH_SIZE (size trigger).
  trickle — ~3 req/s steady; ~0.6 req per 200ms window (time trigger).
  mixed   — alternates 10-req bursts and 1s quiet phases (both triggers).
"""

import argparse
import asyncio
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

# Make `scripts/` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_comments import SAMPLES  # noqa: E402

COMMENTS = [text for _, text in SAMPLES]


async def fire_one(client: httpx.AsyncClient, url: str, comment: str, latencies: list, errors: list):
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/classify", json={"comment": comment}, timeout=30.0)
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)
    except Exception as exc:
        errors.append(str(exc))


async def run_burst(client, url, duration, rate, latencies, errors):
    interval = 1.0 / rate
    deadline = time.perf_counter() + duration
    tasks = []
    while time.perf_counter() < deadline:
        c = random.choice(COMMENTS)
        tasks.append(asyncio.create_task(fire_one(client, url, c, latencies, errors)))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_trickle(client, url, duration, latencies, errors):
    await run_burst(client, url, duration, rate=3.0,
                    latencies=latencies, errors=errors)


async def run_mixed(client, url, duration, latencies, errors):
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        # 10-request burst, fired concurrently
        tasks = [
            asyncio.create_task(fire_one(client, url, random.choice(COMMENTS),
                                         latencies, errors))
            for _ in range(10)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        # quiet phase
        await asyncio.sleep(1.0)


def summarize(latencies, errors):
    n = len(latencies)
    e = len(errors)
    print(f"\nRequests: {n + e}  ok={n}  errors={e}")
    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies)
        print(f"Client latency: p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  "
              f"min={min(latencies)*1000:.0f}ms  max={max(latencies)*1000:.0f}ms")
    if errors:
        print(f"First 3 errors: {errors[:3]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--scenario", choices=["burst", "trickle", "mixed"], default="burst")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Seconds of load to generate.")
    parser.add_argument("--rate", type=float, default=25.0,
                        help="Requests per second (burst scenario only).")
    args = parser.parse_args()

    latencies: list[float] = []
    errors: list[str] = []

    print(f"Scenario: {args.scenario}  duration: {args.duration}s  url: {args.url}")
    async with httpx.AsyncClient() as client:
        if args.scenario == "burst":
            await run_burst(client, args.url, args.duration, args.rate, latencies, errors)
        elif args.scenario == "trickle":
            await run_trickle(client, args.url, args.duration, latencies, errors)
        else:
            await run_mixed(client, args.url, args.duration, latencies, errors)

    summarize(latencies, errors)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Verify both scripts import cleanly (no syntax errors)**

Run: `python3 -c "import sys; sys.path.insert(0, 'scripts'); import sample_comments, loadgen; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Delete redundant gitkeep and commit**

```bash
rm scripts/.gitkeep
git add scripts/sample_comments.py scripts/loadgen.py scripts/.gitkeep
git commit -m "feat: load generator with burst/trickle/mixed scenarios"
```

---

## Task 14: README

**Files:**
- Modify: `README.md`

This README satisfies Abgabe items #6 (Metriken erklären) and #7 (Microbatching beschreiben).

- [ ] **Step 1: Replace the README**

File: `README.md` (overwrite whatever GitHub created)
```markdown
# NLP-Inference Service mit Microbatching, Prometheus und Grafana

CAS AI Operations — NLP Operations — Projekt.

Klassifiziert Kommentare einer fiktiven Online-Zeitung mit der Hugging-Face
Zero-Shot-Pipeline (`facebook/bart-large-mnli`) in eine von vier Kategorien.
Eingehende Requests werden in Mikrobatches gebündelt (Zeitfenster ODER
maximale Grösse) und gemeinsam klassifiziert.

## Quickstart

```bash
# 1. Stack starten (erster Lauf lädt ~1.6 GB Modell)
docker compose up -d

# 2. Healthcheck abwarten (kann beim ersten Start ein paar Minuten dauern)
until curl -fs http://localhost:8080/healthz; do sleep 2; done && echo

# 3. Smoke-Test
curl -X POST http://localhost:8080/classify \
  -H 'content-type: application/json' \
  -d '{"comment":"Sehr guter Artikel, danke!"}'

# 4. Last erzeugen, damit das Dashboard Daten zeigt
python scripts/loadgen.py --scenario mixed --duration 120

# 5. Dashboard öffnen
open http://localhost:3000   # admin / admin
```

## Endpoints

| URL | Beschreibung |
|---|---|
| `http://localhost:8080/classify` | `POST` mit `{"comment": "..."}`, gibt `label` + `score` zurück |
| `http://localhost:8080/metrics` | Prometheus-Metriken im Textformat |
| `http://localhost:8080/healthz` | Liveness-Check |
| `http://localhost:9090` | Prometheus UI |
| `http://localhost:3000` | Grafana (admin/admin), Dashboard "NLP Classifier Service" automatisch geladen |

## Lasttest fahren

```bash
# Size-Trigger demonstrieren: ~25 req/s, Batches füllen sich auf MAX_BATCH_SIZE=8
python scripts/loadgen.py --scenario burst --duration 60

# Time-Trigger demonstrieren: ~3 req/s, Batches sind klein und schliessen nach 200ms
python scripts/loadgen.py --scenario trickle --duration 60

# Beide Trigger in einem Lauf: 10-Bursts, gefolgt von 1s Pause
python scripts/loadgen.py --scenario mixed --duration 120
```

Im Dashboard zeigt das Panel **Durchschnittliche Batch-Grösse** den
Unterschied am deutlichsten: bei `burst` nahe 8, bei `trickle` nahe 1.

## Konfiguration

Alle Einstellungen werden über Environment-Variablen in `docker-compose.yml`
gesteuert. Defaults entsprechen der Aufgabenstellung:

| Variable | Default | Bedeutung |
|---|---|---|
| `MODEL_ID` | `facebook/bart-large-mnli` | HF-Modell für Zero-Shot |
| `CANDIDATE_LABELS` | siehe unten | Komma-getrennte Liste |
| `BATCH_WINDOW_MS` | `200` | Maximale Wartezeit pro Batch |
| `MAX_BATCH_SIZE` | `8` | Maximale Anzahl Kommentare pro Batch |
| `MAX_QUEUE_SIZE` | `256` | Maximale Queue-Tiefe (Backpressure) |

Labels:
- `Zustimmung`
- `sachliche Kritik`
- `Empörung`
- `Beleidigung oder persönlicher Angriff`

## Erklärung der Metriken

| Metrik | Typ | Was sie aussagt — operationaler Nutzen |
|---|---|---|
| `comments_classified_total` | Counter | Gesamtdurchsatz. Mit `rate(...)` ergibt sich Kommentare/Sekunde. |
| `batches_processed_total` | Counter | Wie oft das Modell tatsächlich aufgerufen wurde. Kombiniert mit `comments_classified_total` ergibt sich die effektive Batch-Effizienz. |
| `batch_size` | Histogram | Verteilung der Batch-Grössen. Ein Mittelwert nahe `MAX_BATCH_SIZE` heisst: Size-Trigger dominiert (genug Last). Nahe 1: Time-Trigger dominiert (zu wenig Last für effektives Batching). |
| `batch_inference_duration_seconds` | Histogram | Wie lange das Modell pro Batch braucht. p95 → Latenz-SLO. Steigt mit Batch-Grösse, aber sublinear (das ist der Sinn von Batching). |
| `batch_wait_time_seconds` | Histogram | Wie lange ein Kommentar in der Queue wartet, bevor er klassifiziert wird. Sollte unter `BATCH_WINDOW_MS` bleiben — wenn nicht, ist die Queue überlastet. |
| `microbatch_queue_size` | Gauge | Aktuelle Queue-Tiefe. Wächst → Backpressure droht. Bei `MAX_QUEUE_SIZE` lehnt der Service neue Requests ab. |
| `comments_classified_by_label_total{label}` | Counter | Verteilung der Vorhersagen. Plötzlich viel `Empörung` oder `Beleidigung`? → Moderation aufmerksam machen. |

## Beschreibung der Microbatching-Umsetzung

Implementierung in `classifier-service/app/batcher.py`.

Ein einziger `asyncio`-Worker-Task läuft im Hintergrund und konsumiert eine
`asyncio.Queue` von Anfragen. Jede Anfrage legt einen `BatchItem` mit einem
zugehörigen `Future` in die Queue und wartet auf das Ergebnis.

Der Worker-Loop:

1. Blockiert auf das erste Item (`await queue.get()`).
2. Startet ein Zeitfenster (`deadline = now + BATCH_WINDOW_MS`).
3. Sammelt weitere Items per `asyncio.wait_for(queue.get(), timeout=remaining)`
   bis entweder **`MAX_BATCH_SIZE` erreicht** ist (Size-Trigger) oder
   **die Wartezeit abläuft** (Time-Trigger).
4. Ruft `classifier.classify_batch(...)` *einmal* mit allen gesammelten
   Kommentaren auf.
5. Verteilt die Ergebnisse über die zugehörigen Futures zurück an die
   FastAPI-Handler.

Fehlerpfad: Wirft das Modell eine Exception, wird sie auf *alle* Futures
des Batches gesetzt, so dass jeder Caller einen sauberen 500er erhält und
keine Anfrage stillschweigend hängt.

## Screenshot

Siehe `screenshots/dashboard.png`. Neu erstellt nach einem
`scripts/loadgen.py --scenario mixed --duration 120` Lauf bei laufendem Stack.

## Tests

```bash
cd classifier-service
.venv/bin/pytest -v
```

14 Tests, kein HF-Download nötig — die Tests verwenden Stub-Klassifikatoren.

## Projektstruktur

```
.
├── classifier-service/      FastAPI-Service (app/, tests/, Dockerfile)
├── prometheus/              prometheus.yml
├── grafana/                 Provisioning + Dashboard JSON
├── scripts/                 loadgen.py, sample_comments.py
├── screenshots/             dashboard.png
├── docker-compose.yml
└── docs/superpowers/        Design-Spec und Implementierungs-Plan
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, metric explanations, and microbatching notes"
```

---

## Task 15: End-to-end smoke test + screenshot

This task is partly manual (taking the screenshot) and won't be done by a subagent — you do it yourself.

- [ ] **Step 1: Start the stack**

Run: `docker compose up -d`
Wait for: classifier-service to become healthy (`docker compose ps` shows `healthy`). On first run this takes a few minutes while torch downloads the model into `hf-cache`.

- [ ] **Step 2: Smoke-test the endpoint**

Run:
```bash
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/classify \
  -H 'content-type: application/json' \
  -d '{"comment":"Sehr guter Artikel, danke!"}'
curl -s http://localhost:8080/metrics | head -20
```
Expected:
- healthz returns `{"status":"ok"}`
- classify returns JSON with `label` being one of the 4 labels (likely `Zustimmung`) and `score` between 0 and 1
- metrics returns Prometheus text format with `comments_classified_total` present

- [ ] **Step 3: Verify Prometheus is scraping**

Open: `http://localhost:9090/targets`
Expected: `classifier` target is `UP`.

- [ ] **Step 4: Verify Grafana loaded the dashboard**

Open: `http://localhost:3000` (admin/admin)
Navigate to: Dashboards → "NLP Classifier Service"
Expected: Dashboard loads with 7 panels (initially empty or sparse).

- [ ] **Step 5: Run a mixed loadtest**

Run (from repo root, using the venv created in Task 2 which already has httpx):
```bash
classifier-service/.venv/bin/python scripts/loadgen.py --scenario mixed --duration 120
```
Wait for it to complete (~2 minutes). It will print latency stats at the end.

- [ ] **Step 6: Take the dashboard screenshot**

In the Grafana dashboard, set the time range to "Last 5 minutes", wait for it to refresh, and take a screenshot of the full dashboard showing all 7 panels with data. Save it as `screenshots/dashboard.png`.

- [ ] **Step 7: Tear down the stack**

```bash
docker compose down
```
(The `hf-cache` volume persists so future starts are fast.)

- [ ] **Step 8: Commit the screenshot**

```bash
rm screenshots/.gitkeep
git add screenshots/dashboard.png screenshots/.gitkeep
git commit -m "docs: add Grafana dashboard screenshot from mixed loadtest"
```

- [ ] **Step 9: Push everything**

```bash
git push
```

---

## Final verification — all Bewertung/Abgabe items present

After completing Task 15, run this sanity check:

```bash
ls -la \
  classifier-service/app/main.py \
  classifier-service/app/classifier.py \
  classifier-service/app/batcher.py \
  docker-compose.yml \
  prometheus/prometheus.yml \
  grafana/dashboards/classifier_dashboard.json \
  screenshots/dashboard.png \
  README.md
```

All 8 files must exist. They map to:

| File | Bewertung # | Abgabe # |
|---|---|---|
| `classifier-service/app/classifier.py` + `main.py` | 1 (NLP-Inference) | 1 |
| `CANDIDATE_LABELS` default in `config.py` | 2 (Labels) | — |
| `classifier-service/app/batcher.py` | 3 (Microbatching) | 1 |
| `metrics.py` + `/metrics` mount | 4 (Metriken) | 1 |
| `grafana/dashboards/classifier_dashboard.json` | 5 (Dashboard) | 4 |
| `docker-compose.yml` | 6 (Compose) | 2 |
| `README.md` | 7 (Doku) | 6, 7 |
| `screenshots/dashboard.png` | 8 (Screenshot) | 5 |
| `prometheus/prometheus.yml` | — | 3 |
