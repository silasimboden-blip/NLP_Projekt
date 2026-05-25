"""Load generator for the classifier service.

Three scenarios:
  burst   — keeps --concurrency requests in flight (default 16); fills batches
            to MAX_BATCH_SIZE (size trigger) with a bounded queue.
  trickle — ~3 req/s spaced; ~1 req per 200ms window (time trigger).
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
        r = await client.post(f"{url}/classify", json={"comment": comment}, timeout=120.0)
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)
    except Exception as exc:
        # str(exc) is empty for many httpx errors (e.g. ReadTimeout); fall back to type name.
        errors.append(str(exc) or type(exc).__name__)


async def run_burst(client, url, duration, concurrency, latencies, errors):
    """Keep `concurrency` requests in flight for `duration` seconds.

    Concurrency-limited (not rate-limited): each worker fires a request,
    waits for it, then fires the next. With concurrency > MAX_BATCH_SIZE the
    queue stays full enough that batches flush on the size trigger, but it
    never grows unbounded — so nothing times out even on slow CPU inference.
    """
    deadline = time.perf_counter() + duration

    async def worker():
        while time.perf_counter() < deadline:
            await fire_one(client, url, random.choice(COMMENTS), latencies, errors)

    await asyncio.gather(*(worker() for _ in range(concurrency)), return_exceptions=True)


async def run_trickle(client, url, duration, latencies, errors):
    # ~3 req/s, spaced out — each comment is usually alone in its 200ms
    # window, so batches contain ~1 item and flush on the time trigger.
    interval = 1.0 / 3.0
    deadline = time.perf_counter() + duration
    tasks = []
    while time.perf_counter() < deadline:
        tasks.append(asyncio.create_task(
            fire_one(client, url, random.choice(COMMENTS), latencies, errors)))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)


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
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of in-flight requests (burst scenario only). "
                             "Should exceed MAX_BATCH_SIZE to keep batches full.")
    args = parser.parse_args()

    latencies: list[float] = []
    errors: list[str] = []

    print(f"Scenario: {args.scenario}  duration: {args.duration}s  url: {args.url}")
    async with httpx.AsyncClient() as client:
        if args.scenario == "burst":
            await run_burst(client, args.url, args.duration, args.concurrency, latencies, errors)
        elif args.scenario == "trickle":
            await run_trickle(client, args.url, args.duration, latencies, errors)
        else:
            await run_mixed(client, args.url, args.duration, latencies, errors)

    summarize(latencies, errors)


if __name__ == "__main__":
    asyncio.run(main())
