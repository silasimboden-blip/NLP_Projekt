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
