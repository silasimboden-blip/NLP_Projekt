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
