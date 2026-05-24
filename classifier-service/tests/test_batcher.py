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
