#!/usr/bin/env python3
"""
Load test for the universal-translator API.

Measures throughput and latency under concurrent load for two scenarios:
  en-fr  — single target language (isolates per-request latency)
  en-all — all non-English languages (measures fan-out / batch efficiency)

Usage:
  python3 tests/load_test.py [OPTIONS]

Options:
  --url URL           API base URL (default: http://localhost:3000)
  --concurrency N     Max concurrent in-flight requests (default: 10)
  --requests N        Total requests per scenario (default: 100)
  --warmup N          Sequential warmup requests before measurement (default: 3)
  --scenario SCENARIO en-fr | en-all | both (default: both)

Requires: pip install aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install it with:")
    print("  pip install aiohttp")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test inputs — spans all three auto-beam tiers
# ---------------------------------------------------------------------------

TEST_INPUTS = [
    # Short (greedy)
    "Hello, how are you?",
    "The meeting starts at 10 AM tomorrow.",
    # Medium (beam=2)
    "The sun rises in the east and sets in the west. The coffee costs $3.50 and the newspaper costs \u20ac2.00.",
    # Long (beam=4)
    "The annual conference will be held in Geneva next month. Participants from over forty countries are expected to attend and registration closes on the fifteenth of March.",
    "Scientists have discovered a new species of deep-sea fish that produces its own bioluminescent light. The creature was found at a depth of three thousand metres during an unmanned submarine survey of the Pacific Ocean floor.",
]

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def payload_en_fr(request_index: int) -> dict:
    text = TEST_INPUTS[request_index % len(TEST_INPUTS)]
    return {
        "texts": [text],
        "target_languages": ["fr"],
        "source_language": "en",
    }


def payload_en_all(request_index: int) -> dict:
    text = TEST_INPUTS[request_index % len(TEST_INPUTS)]
    return {
        "texts": [text],
        "target_languages": ["all"],
        "source_language": "en",
    }


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


async def do_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
) -> tuple[float, str | None]:
    """Return (latency_seconds, error_message_or_None)."""
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.text()
            elapsed = time.perf_counter() - t0
            if resp.status != 200:
                return elapsed, f"HTTP {resp.status}: {body[:200]}"
            return elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, str(exc)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


async def run_scenario(
    name: str,
    payload_fn,
    url: str,
    concurrency: int,
    n_requests: int,
    warmup: int,
    n_languages: int,
) -> None:
    translate_url = url.rstrip("/") + "/translate"

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # --- Warmup phase ---
        if warmup > 0:
            print(f"  Warming up ({warmup} sequential requests)…", flush=True)
            for i in range(warmup):
                _, err = await do_request(session, translate_url, payload_fn(i))
                if err:
                    print(f"  WARNING: warmup request {i + 1} failed: {err}")

        # --- Load phase ---
        print(f"  Running {n_requests} requests (concurrency={concurrency})…", flush=True)

        latencies: list[float] = []
        errors: list[str] = []
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(idx: int) -> None:
            async with sem:
                latency, err = await do_request(session, translate_url, payload_fn(idx))
                latencies.append(latency)
                if err:
                    errors.append(err)

        t_start = time.perf_counter()
        await asyncio.gather(*(bounded_request(i) for i in range(n_requests)))
        wall_time = time.perf_counter() - t_start

    print_results(name, latencies, errors, wall_time, concurrency, n_requests, warmup, n_languages)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------


def print_results(
    name: str,
    latencies: list[float],
    errors: list[str],
    wall_time: float,
    concurrency: int,
    n_requests: int,
    warmup: int,
    n_languages: int,
) -> None:
    n = len(latencies)
    rps = n / wall_time if wall_time > 0 else 0.0
    tps = rps * n_languages

    sorted_lat = sorted(latencies)

    def ms(v: float) -> int:
        return int(v * 1000)

    def pct(p: float) -> int:
        idx = min(int(p / 100 * n), n - 1)
        return ms(sorted_lat[idx])

    mean_ms = ms(statistics.mean(latencies)) if latencies else 0

    lang_label = f"{n_languages} language{'s' if n_languages != 1 else ''}"
    print()
    print(f"=== Scenario: {name} ({lang_label}) ===")
    print(f"Concurrency: {concurrency}  |  Requests: {n_requests}  |  Warmup: {warmup}")
    print(f"Duration:   {wall_time:.1f}s")
    print(f"Throughput: {rps:.2f} req/s  |  {tps:.1f} translations/s")
    if latencies:
        print(
            f"Latency (ms): min={ms(sorted_lat[0])}"
            f"  mean={mean_ms}"
            f"  p50={pct(50)}"
            f"  p75={pct(75)}"
            f"  p95={pct(95)}"
            f"  p99={pct(99)}"
            f"  max={ms(sorted_lat[-1])}"
        )
    print(f"Errors: {len(errors)} / {n_requests}")
    if errors:
        print(f"  First error: {errors[0]}")


# ---------------------------------------------------------------------------
# Language count helper
# ---------------------------------------------------------------------------

# The engine supports 55 languages total. The same-language shortcut removes
# en→en, so fan-out for an English source is 54.
EN_ALL_LANGUAGE_COUNT = 54


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load test for the universal-translator API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:3000",
        help="API base URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        metavar="N",
        help="Max concurrent in-flight requests (default: 10)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        metavar="N",
        help="Total requests per scenario (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        metavar="N",
        help="Sequential warmup requests before measurement (default: 3)",
    )
    parser.add_argument(
        "--scenario",
        choices=["en-fr", "en-all", "both"],
        default="both",
        help="Scenario to run: en-fr | en-all | both (default: both)",
    )
    args = parser.parse_args()

    scenarios = []
    if args.scenario in ("en-fr", "both"):
        scenarios.append(("en-fr", payload_en_fr, 1))
    if args.scenario in ("en-all", "both"):
        scenarios.append(("en-all", payload_en_all, EN_ALL_LANGUAGE_COUNT))

    for name, payload_fn, n_languages in scenarios:
        asyncio.run(
            run_scenario(
                name=name,
                payload_fn=payload_fn,
                url=args.url,
                concurrency=args.concurrency,
                n_requests=args.requests,
                warmup=args.warmup,
                n_languages=n_languages,
            )
        )


if __name__ == "__main__":
    main()
