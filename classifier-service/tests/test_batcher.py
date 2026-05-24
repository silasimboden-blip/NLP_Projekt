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
