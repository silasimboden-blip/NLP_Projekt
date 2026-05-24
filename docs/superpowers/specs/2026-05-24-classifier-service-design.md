# Classifier Service — Design Spec

**Date:** 2026-05-24
**Project:** CAS AI Operations — NLP Operations — Projekt
**Source brief:** `02_NLP Operations_01_Projektaufgabe.pdf` (kept locally, gitignored)
**Repo:** https://github.com/silasimboden-blip/NLP_Projekt

## 1. Goal

Build a minimal NLP inference service that classifies short German news comments into one of four categories using Hugging Face's zero-shot pipeline with `facebook/bart-large-mnli`. The service buffers incoming requests into microbatches, exposes Prometheus metrics, ships with a provisioned Grafana dashboard, and runs end-to-end via `docker compose up`. A Python load generator drives traffic in three different patterns so both microbatching trigger paths (size and time window) are observable in the dashboard.

The deliverable must satisfy all 8 Bewertung criteria and all 7 Abgabe items from the brief.

## 2. Scope

**In scope**
- One HTTP service with `POST /classify`, `GET /metrics`, `GET /healthz`
- Microbatcher with both size and time-window triggers (defaults: 200 ms / 8 comments)
- Seven Prometheus metrics as listed in the brief
- Three-service docker-compose stack: `classifier-service`, `prometheus`, `grafana`
- Provisioned Grafana datasource and dashboard (no manual setup)
- Python load generator with `burst`, `trickle`, and `mixed` scenarios
- README covering quickstart, metrics explanation, and microbatching explanation
- One screenshot of the dashboard during a loadtest

**Out of scope**
- Authentication, persistence, multi-tenant concerns
- Horizontal scaling / multiple worker replicas
- GPU inference (CPU only)
- CI pipeline (manual `pytest` is sufficient)
- A second `/classify_batch` endpoint that takes a list of comments — the brief only specifies the single-comment variant

## 3. Labels

```
candidate_labels = [
    "Zustimmung",
    "sachliche Kritik",
    "Empörung",
    "Beleidigung oder persönlicher Angriff",
]
```

Three picked directly from the brief's example list; the fourth ("Beleidigung oder persönlicher Angriff") is not in the brief but fits the stated moderation use case better than the unused brief options. Labels are configurable via the `CANDIDATE_LABELS` env var (comma-separated) so they can be tweaked without a rebuild.

## 4. Architecture

```
┌──────────────────┐   POST /classify    ┌──────────────────────┐
│  loadgen.py      │ ──────────────────▶ │  classifier-service  │
│  (host laptop)   │                     │  (FastAPI, :8080)    │
│  burst/trickle/  │ ◀──── label/score ──│                      │
│  mixed scenarios │                     │  GET /metrics ───────┼─┐
└──────────────────┘                     │  GET /healthz        │ │
                                         └──────────────────────┘ │
                                                                  │ scrape every 5s
                                                                  ▼
                                                      ┌──────────────────┐
                                                      │  prometheus      │
                                                      │  (:9090)         │
                                                      └────────┬─────────┘
                                                               │
                                                               ▼
                                                      ┌──────────────────┐
                                                      │  grafana         │
                                                      │  (:3000)         │
                                                      │  provisioned     │
                                                      │  datasource +    │
                                                      │  dashboard       │
                                                      └──────────────────┘
```

Three containers in compose; the load generator stays on the host to keep the compose file minimal.

## 5. Project layout

```
NLP_Projekt/
├── README.md                              # quickstart + Metriken + Microbatching
├── docker-compose.yml
├── classifier-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py                        # FastAPI app, endpoints
│       ├── classifier.py                  # HF pipeline wrapper
│       ├── batcher.py                     # MicroBatcher: queue + worker + futures
│       ├── metrics.py                     # Prometheus metric objects (shared)
│       └── config.py                      # env-driven settings
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboards.yml
│   └── dashboards/
│       └── classifier_dashboard.json
├── scripts/
│   ├── loadgen.py                         # httpx async client, --scenario flag
│   └── sample_comments.py                 # 20 German comments, 5 per label
├── tests/
│   └── test_batcher.py                    # unit test of batching behavior
├── screenshots/
│   └── dashboard.png                      # captured after loadtest
└── docs/
    └── superpowers/specs/
        └── 2026-05-24-classifier-service-design.md
```

## 6. Component design

### 6.1 `app/classifier.py` — model wrapper

- Loads `facebook/bart-large-mnli` at service startup (FastAPI `lifespan` event), so the first `/classify` request after startup is fast and the demo isn't skewed by a one-time download.
- Exposes one method: `classify_batch(comments: list[str]) -> list[tuple[str, float]]`.
- Internally calls the HF pipeline with the configured `candidate_labels` and `multi_label=False`, returning `(top_label, top_score)` per input.
- Knows nothing about asyncio, FastAPI, or Prometheus.

### 6.2 `app/batcher.py` — microbatcher

Owns:
- `asyncio.Queue[BatchItem]` of bounded size (`MAX_QUEUE_SIZE`, default 256)
- One background worker task started on first `submit()`
- No locks beyond what the queue already provides

`BatchItem` dataclass: `comment: str`, `future: asyncio.Future`, `enqueued_at: float`.

`submit(comment)` flow:
1. Record `enqueued_at = loop.time()`
2. Create a future
3. `await queue.put(BatchItem(...))` (backpressure if queue is full)
4. `return await future` (resolved by the worker)

Worker loop (`_run()`):

```python
async def _run(self):
    loop = asyncio.get_running_loop()
    while True:
        first = await self.queue.get()              # wait for any item
        batch = [first]
        deadline = loop.time() + BATCH_WINDOW_MS / 1000

        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # observe size + per-item wait before inference
        BATCH_SIZE_HIST.observe(len(batch))
        for item in batch:
            BATCH_WAIT.observe(loop.time() - item.enqueued_at)

        with INFERENCE_DURATION.time():
            try:
                results = classifier.classify_batch([i.comment for i in batch])
            except Exception as exc:
                for item in batch:
                    item.future.set_exception(exc)
                continue

        BATCHES_PROCESSED.inc()
        COMMENTS_TOTAL.inc(len(batch))
        for item, (label, score) in zip(batch, results):
            COMMENTS_BY_LABEL.labels(label=label).inc()
            item.future.set_result((label, score))
```

**Both triggers are exercised:**
- **Size trigger** — the inner `while len(batch) < MAX_BATCH_SIZE` exits as soon as the 8th item is pulled off the queue.
- **Time trigger** — `asyncio.wait_for(..., timeout=remaining)` raises `TimeoutError` when the 200 ms window has elapsed since the first item arrived.

**Error path** — if the model call raises, every future in the batch receives the exception; FastAPI translates that to a 500 per request. No request is silently dropped.

### 6.3 `app/metrics.py` — shared metric objects

All seven Prometheus metric objects live in a tiny dedicated module so both `main.py` and `batcher.py` can import them without creating a cycle.

```python
from prometheus_client import Counter, Gauge, Histogram

COMMENTS_TOTAL     = Counter("comments_classified_total", "...")
BATCHES_PROCESSED  = Counter("batches_processed_total", "...")
BATCH_SIZE_HIST    = Histogram("batch_size", "...", buckets=(1,2,3,4,5,6,7,8))
INFERENCE_DURATION = Histogram("batch_inference_duration_seconds", "...")
BATCH_WAIT         = Histogram("batch_wait_time_seconds", "...")
QUEUE_SIZE_GAUGE   = Gauge("microbatch_queue_size", "...")
COMMENTS_BY_LABEL  = Counter("comments_classified_by_label_total", "...", ["label"])
```

### 6.4 `app/main.py` — FastAPI app

Responsibilities (and only these):
- Define Pydantic request/response models
- Wire `/healthz`, `/classify`, and mount the `prometheus_client` ASGI app at `/metrics`
- Use FastAPI `lifespan` to (a) trigger model load, (b) start the batcher worker
- Middleware that refreshes `QUEUE_SIZE_GAUGE` on each `/metrics` scrape

Sketch:

```python
from prometheus_client import make_asgi_app
from app.batcher import batcher
from app.metrics import QUEUE_SIZE_GAUGE

class ClassifyRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class ClassifyResponse(BaseModel):
    comment: str
    label: str
    score: float

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    label, score = await batcher.submit(req.comment)
    return ClassifyResponse(comment=req.comment, label=label, score=score)

@app.get("/healthz")
def healthz(): return {"status": "ok"}

# Refresh the gauge on each scrape, then delegate to prometheus_client.
@app.middleware("http")
async def refresh_queue_gauge(request, call_next):
    if request.url.path.startswith("/metrics"):
        QUEUE_SIZE_GAUGE.set(batcher.queue.qsize())
    return await call_next(request)

app.mount("/metrics", make_asgi_app())
```

### 6.5 `app/config.py` — env-driven settings

Single module reading env vars with sensible defaults. No `pydantic-settings` dependency to keep things minimal — plain `os.getenv` is enough.

| Var | Default | Notes |
|---|---|---|
| `MODEL_ID` | `facebook/bart-large-mnli` | |
| `CANDIDATE_LABELS` | the 4 labels above, comma-separated | |
| `BATCH_WINDOW_MS` | `200` | matches brief example |
| `MAX_BATCH_SIZE` | `8` | matches brief example |
| `MAX_QUEUE_SIZE` | `256` | bounds memory in pathological bursts |
| `HF_HOME` | `/models` | cache lives in named volume |
| `TRANSFORMERS_CACHE` | `/models` | |
| `TOKENIZERS_PARALLELISM` | `false` | suppresses HF warning |

## 7. Metrics

All seven from the brief, defined in `app/metrics.py`, set inside the batcher (except the gauge, which is refreshed on each scrape):

| Metric | Type | Set where |
|---|---|---|
| `comments_classified_total` | Counter | batcher, `.inc(len(batch))` after successful inference |
| `batches_processed_total` | Counter | batcher, `.inc()` after successful inference |
| `batch_size` | Histogram | batcher, `.observe(len(batch))` just before inference |
| `batch_inference_duration_seconds` | Histogram | batcher, `with .time():` around model call |
| `batch_wait_time_seconds` | Histogram | batcher, `.observe(now - enqueued_at)` per item |
| `microbatch_queue_size` | Gauge | `main.py` middleware, `.set(queue.qsize())` on each `/metrics` scrape |
| `comments_classified_by_label_total{label}` | Counter | batcher, per-result `.labels(label=...).inc()` |

## 8. Prometheus configuration

```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: "classifier"
    metrics_path: /metrics
    static_configs:
      - targets: ["classifier-service:8080"]
```

5 s scrape interval (rather than the default 15 s) so the dashboard fills in quickly during short loadtests.

## 9. Grafana dashboard

Provisioned via two YAML files and one dashboard JSON; no manual setup at runtime.

**Datasource** — `http://prometheus:9090`, set as default.

**Dashboard** — `classifier_dashboard.json`, hand-written for legibility. Seven panels, exactly the PromQL from the brief:

| Panel | Type | PromQL |
|---|---|---|
| Kommentare pro Sekunde | timeseries | `rate(comments_classified_total[1m])` |
| Batches pro Minute | timeseries | `rate(batches_processed_total[1m]) * 60` |
| Durchschnittliche Batch-Grösse | timeseries | `rate(batch_size_sum[1m]) / rate(batch_size_count[1m])` |
| p95 Inference-Latenz | timeseries | `histogram_quantile(0.95, rate(batch_inference_duration_seconds_bucket[5m]))` |
| p95 Batch-Wartezeit | timeseries | `histogram_quantile(0.95, rate(batch_wait_time_seconds_bucket[5m]))` |
| Queue-Grösse | stat | `microbatch_queue_size` |
| Kommentare nach Label | timeseries (stacked) | `sum by (label) (rate(comments_classified_by_label_total[5m]))` |

## 10. Load generator

`scripts/loadgen.py` — async `httpx` client. One file, `--scenario` flag.

```
python scripts/loadgen.py --url http://localhost:8080 --scenario burst    --duration 60
python scripts/loadgen.py --url http://localhost:8080 --scenario trickle  --duration 60
python scripts/loadgen.py --url http://localhost:8080 --scenario mixed    --duration 120
```

| Scenario | Pattern | What it demonstrates |
|---|---|---|
| `burst` | ~25 req/s steady | **Size trigger.** Queue fills, batches hit `MAX_BATCH_SIZE=8`, "Durchschnittliche Batch-Grösse" ≈ 8. |
| `trickle` | ~3 req/s steady | **Time trigger.** ~0.6 requests per 200 ms window, batches usually contain 1 item, "Durchschnittliche Batch-Grösse" ≈ 1, "Batch-Wartezeit p95" stays near 200 ms. |
| `mixed` | alternating 10-request bursts and 1 s quiet phases | **Both triggers in one timeline.** Useful for a single screenshot showing the dashboard reacting to changing load. |

Each scenario samples comments uniformly from `sample_comments.py` (20 German comments, 5 per label) so the "Kommentare nach Label" panel shows a meaningful distribution. The script prints a final summary: request count, p50/p95 client-side latency, error rate.

## 11. Sample comments

20 hand-written German comments in `scripts/sample_comments.py`, 5 per label, e.g.:

```python
SAMPLES = [
    ("Zustimmung", "Sehr guter, gut recherchierter Artikel, danke!"),
    ("sachliche Kritik", "Mir fehlen hier Quellenangaben für die zitierten Zahlen."),
    ("Empörung", "Das ist eine absolute Frechheit, wer hat das durchgewunken??"),
    ("Beleidigung oder persönlicher Angriff", "Der Autor dieses Texts ist offensichtlich ein Vollidiot."),
    # ... 4 more per label
]
```

The label tags are only used by the loadgen to balance sampling; they are **not** sent to the service (the service is genuinely zero-shot and only sees the raw comment text).

## 12. Docker compose

Three services. No extra k6 container. `hf-cache` is a named volume so model weights persist across `docker compose down/up`.

```yaml
services:
  classifier-service:
    build: ./classifier-service
    ports: ["8080:8080"]
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

  prometheus:
    image: prom/prometheus:v2.55.1
    ports: ["9090:9090"]
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:11.4.0
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on: [prometheus]

volumes:
  hf-cache:
```

## 13. Dockerfile (classifier-service)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

`requirements.txt` (pinned):

```
fastapi==0.115.4
uvicorn[standard]==0.32.0
transformers==4.46.2
torch==2.5.1
prometheus-client==0.21.0
httpx==0.27.2          # for tests + loadgen if run inside the container
pytest==8.3.3
pytest-asyncio==0.24.0
```

## 14. README structure

One README at repo root, German + a bit of English, sections:

1. **Quickstart** — `docker compose up -d`; wait for healthz; `python scripts/loadgen.py …`; open `http://localhost:3000` (admin/admin), dashboard auto-loaded.
2. **Endpoints** — what's at 8080, 9090, 3000.
3. **Lasttest fahren** — how to run `burst`, `trickle`, `mixed`; what to expect in the dashboard for each.
4. **Erklärung der Metriken** — one short paragraph per metric, why it matters operationally. *Satisfies Abgabe #6.*
5. **Erklärung Microbatching** — 5–10 lines describing the queue + worker, both triggers, why we chose this approach. *Satisfies Abgabe #7.*
6. **Screenshot** — pointer to `screenshots/dashboard.png`, with a note on how to recapture it.

## 15. Testing strategy

**Unit test — `tests/test_batcher.py`** (~30 lines)

- Stub classifier that returns `("Zustimmung", 1.0)` for every input and records each call (so we can assert how many comments arrived per batch).
- Test 1: submit 12 items concurrently → assert ≥ 1 batch with `size > 1` and all 12 futures resolve. Exercises the size trigger.
- Test 2: submit 2 items spaced ~250 ms apart → assert 2 batches of size 1 each. Exercises the time trigger.
- Run: `pytest` from the host, or `docker compose run --rm classifier-service pytest`.

No HF model is loaded during tests (stub).

**Manual smoke**

```
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/classify \
  -H 'content-type: application/json' \
  -d '{"comment":"Sehr guter Artikel, danke!"}'
python scripts/loadgen.py --scenario mixed --duration 120
open http://localhost:3000
```

No CI pipeline — not in the brief.

## 16. Mapping to the assignment

### Bewertung (must all pass)

| # | Criterion | Where it's satisfied |
|---|---|---|
| 1 | NLP-Inference funktioniert | `app/classifier.py` + `POST /classify` in `main.py` |
| 2 | Sinnvolle Labels | `CANDIDATE_LABELS` default; documented in README |
| 3 | Microbatching umgesetzt (Zeit + Grösse) | `app/batcher.py` `_run()`; both triggers exercised by `loadgen.py --scenario {burst,trickle}` |
| 4 | Prometheus-Metriken bereitgestellt | All 7 metrics in `main.py`/`batcher.py`, exposed at `/metrics` via `make_asgi_app()` |
| 5 | Grafana-Dashboard vorhanden | Provisioned dashboard JSON, 7 panels |
| 6 | Docker-Compose-Setup funktioniert | `docker-compose.yml` with classifier-service, prometheus, grafana |
| 7 | Abgabe nachvollziehbar dokumentiert | README quickstart + metrics + microbatching sections |
| 8 | Screenshot abgegeben | `screenshots/dashboard.png` captured after loadgen run |

### Abgabe (artifacts to hand in)

| # | Item | File |
|---|---|---|
| 1 | Quellcode des Classifier-Services | `classifier-service/` |
| 2 | `docker-compose.yml` | repo root |
| 3 | `prometheus.yml` | `prometheus/prometheus.yml` |
| 4 | Grafana-Dashboard als JSON | `grafana/dashboards/classifier_dashboard.json` |
| 5 | Screenshot | `screenshots/dashboard.png` |
| 6 | Erklärung der wichtigsten Metriken | README section |
| 7 | Beschreibung der Microbatching-Umsetzung | README section |

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| First container start downloads ~1.6 GB model — slow first boot, especially on metered connections | `hf-cache` named volume keeps it across restarts; README warns about first-run download time |
| `bart-large-mnli` inference is CPU-bound and not super fast (~hundreds of ms per batch on a laptop) | `MAX_BATCH_SIZE=8` keeps batches small; the demo doesn't need high throughput |
| Asyncio event loop blocked during the synchronous HF call | Acceptable for a single-process demo; if it becomes a problem we can wrap the call in `loop.run_in_executor` (Approach B from brainstorming). Not done in v1. |
| Model misclassifies on subtle inputs (especially "Beleidigung" vs "Empörung") | Sample comments are chosen to be unambiguous; the brief grades that the service works, not that classifications are perfect |
| Prometheus or Grafana version drift breaks dashboard JSON | Pinned image tags (`prom/prometheus:v2.55.1`, `grafana/grafana:11.4.0`) |

## 18. Out-of-scope / explicit non-goals

These could be added later but are intentionally excluded to stay minimal:
- `POST /classify_batch` taking a list of comments (brief only specifies single-comment endpoint)
- Returning all 4 scores instead of top-1 (brief's response example is top-1 only)
- k6 load generator container
- Authentication, rate limiting, CORS
- Distributed tracing / OpenTelemetry
- Multi-replica deployment
- Returning per-label probability distribution
