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

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[_BatchItem]) -> None:
        now = time.monotonic()
        BATCH_SIZE_HIST.observe(len(batch))
        for item in batch:
            BATCH_WAIT.observe(now - item.enqueued_at)

        loop = asyncio.get_running_loop()
        comments = [i.comment for i in batch]
        try:
            # Run the blocking HF pipeline call in a thread-pool executor so
            # the asyncio event loop stays free to serve /metrics and /healthz.
            def _infer():
                with INFERENCE_DURATION.time():
                    return self._classifier.classify_batch(comments)

            results = await loop.run_in_executor(None, _infer)
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
