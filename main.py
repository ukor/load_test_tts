
import asyncio
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional
import httpx

from dotenv import load_dotenv

load_dotenv()

ENDPOINT_BASE =  os.getenv("ENDPOINT_BASE") 
STREAM_URL = f"{ENDPOINT_BASE}/stream"
LANGUAGES_URL = f"{ENDPOINT_BASE}/languages"
HF_TOKEN = os.getenv("HF_TOKEN")


if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set.")
    print("Run: export HF_TOKEN=\"your_hf_token_here\"")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

OUTPUT_CSV = "tts_load_test_results.csv"
OUTPUT_AUDIO_DIR = "recordings"
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)


@dataclass
class TestResult:
    concurrency_level: int
    request_index: int
    language: str
    voice: str
    text_snippet: str
    first_audio_sec: Optional[float]
    finished_sec: Optional[float]
    total_bytes: int
    request_id: str
    max_chunk_gap_sec: float
    status: str
    notes: str


async def wake_and_warmup(client: httpx.AsyncClient, max_attempts: int = 15, delay: float = 10.0):
    """Handles cold-start wakeups (which can take up to ~2 minutes)."""
    print("Checking endpoint status and warming up...")
    payload = {
        "inputs": "Testing endpoint warmup.",
        "language": "english",
        "voice": "female",
    }

    for attempt in range(1, max_attempts + 1):
        try:
            start = time.perf_counter()
            response = await client.post(STREAM_URL, headers=HEADERS, json=payload, timeout=180.0)
            elapsed = time.perf_counter() - start

            if response.status_code == 200 and len(response.content) > 0:
                req_id = response.headers.get("x-request-id", "unknown")
                print(f"Endpoint ready! Warmup completed in {elapsed:.2f}s (Request-ID: {req_id})")
                return True
            else:
                print(f"Attempt {attempt}/{max_attempts}: Status {response.status_code}. Waiting {delay}s...")
        except Exception as e:
            print(f"Attempt {attempt}/{max_attempts}: Error ({e}). Waiting {delay}s to allow instance to boot...")
        
        await asyncio.sleep(delay)

    print("Failed to wake endpoint after several attempts.")
    return False


async def stream_tts(
    client: httpx.AsyncClient,
    test_id: str,
    concurrency_level: int,
    req_index: int,
    text: str,
    language: str,
    voice: str,
) -> TestResult:
    payload = {"inputs": text, "language": language, "voice": voice}
    req_id = "unknown"
    first_audio_time = None
    max_gap = 0.0
    total_bytes = 0
    audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{test_id}_c{concurrency_level}_req{req_index}.wav")

    start_time = time.perf_counter()
    prev_chunk_time = start_time

    try:
        async with client.stream("POST", STREAM_URL, headers=HEADERS, json=payload, timeout=120.0) as response:
            req_id = response.headers.get("x-request-id", "unknown")

            if response.status_code != 200:
                finished_time = time.perf_counter() - start_time
                error_body = (await response.aread()).decode(errors="ignore")
                return TestResult(
                    concurrency_level=concurrency_level,
                    request_index=req_index,
                    language=language,
                    voice=voice,
                    text_snippet=text[:40] + "...",
                    first_audio_sec=None,
                    finished_sec=round(finished_time, 3),
                    total_bytes=0,
                    request_id=req_id,
                    max_chunk_gap_sec=0.0,
                    status=f"HTTP_{response.status_code}",
                    notes=f"Failed: {error_body[:100]}",
                )

            with open(audio_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    now = time.perf_counter()
                    if first_audio_time is None:
                        first_audio_time = now - start_time
                    else:
                        gap = now - prev_chunk_time
                        if gap > max_gap:
                            max_gap = gap

                    prev_chunk_time = now
                    total_bytes += len(chunk)
                    f.write(chunk)

            total_elapsed = time.perf_counter() - start_time

            return TestResult(
                concurrency_level=concurrency_level,
                request_index=req_index,
                language=language,
                voice=voice,
                text_snippet=text[:40] + "...",
                first_audio_sec=round(first_audio_time, 3) if first_audio_time else None,
                finished_sec=round(total_elapsed, 3),
                total_bytes=total_bytes,
                request_id=req_id,
                max_chunk_gap_sec=round(max_gap, 3),
                status="SUCCESS",
                notes=f"Audio saved to {audio_path}",
            )

    except Exception as e:
        total_elapsed = time.perf_counter() - start_time
        return TestResult(
            concurrency_level=concurrency_level,
            request_index=req_index,
            language=language,
            voice=voice,
            text_snippet=text[:40] + "...",
            first_audio_sec=None,
            finished_sec=round(total_elapsed, 3),
            total_bytes=0,
            request_id=req_id,
            max_chunk_gap_sec=0.0,
            status="EXCEPTION",
            notes=str(e),
        )


async def run_concurrency_batch(client: httpx.AsyncClient, batch_cases: List[dict], concurrency_level: int):
    print(f"\n--- Running Concurrency Level: {concurrency_level} requests in parallel ---")
    tasks = []
    timestamp = int(time.time())

    for idx, case in enumerate(batch_cases[:concurrency_level]):
        tasks.append(
            stream_tts(
                client=client,
                test_id=f"test_{timestamp}",
                concurrency_level=concurrency_level,
                req_index=idx + 1,
                text=case["text"],
                language=case["language"],
                voice=case["voice"],
            )
        )

    results = await asyncio.gather(*tasks)
    return results


async def main():
    # Long test phrases to evaluate gaps and degradation across multiple sentences
    test_cases = [
        {
            "language": "hausa",
            "voice": "female",
            "text": "Sannu, yaya kake? Ina fatan kana cikin koshin lafiya da natsuwa. Yau rana ce mai kyau sosai wacce za mu yi aiki mai ma'ana tare.",
        },
        {
            "language": "english",
            "voice": "nigerian_female",
            "text": "Good afternoon. Please make sure that you check the audio latency and listen carefully for any robotic artifacts in the generated voice output.",
        },
        {
            "language": "english",
            "voice": "male",
            "text": "Artificial intelligence and neural text-to-speech models require significant GPU memory bandwidth, especially when concurrency scales on a single accelerator.",
        },
    ]

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        # Wake up endpoint
        is_awake = await wake_and_warmup(client)
        if not is_awake:
            print("Aborting test due to endpoint cold start failure.")
            return

        all_results: List[TestResult] = []

        # Concurrency sweeps (1, 2, and 3 parallel requests)
        for concurrency in [1, 2, 3]:
            batch_results = await run_concurrency_batch(client, test_cases, concurrency)
            all_results.extend(batch_results)
            # Brief pause between batch stages
            await asyncio.sleep(3)

        # Write report to CSV
        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(asdict(all_results[0]).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))

        print(f"\nAll tests completed. Results saved to: {OUTPUT_CSV}")
        print(f"WAV audio files saved to: {OUTPUT_AUDIO_DIR}/")

        # Summary printout
        print("\n" + "=" * 95)
        print(f"{'Conc':<5} | {'Lang/Voice':<24} | {'First Audio':<12} | {'Finished':<10} | {'Max Gap':<8} | {'Request ID'}")
        print("=" * 95)
        for r in all_results:
            voice_label = f"{r.language}/{r.voice}"
            first_txt = f"{r.first_audio_sec}s" if r.first_audio_sec else "N/A"
            finish_txt = f"{r.finished_sec}s" if r.finished_sec else "N/A"
            gap_txt = f"{r.max_chunk_gap_sec}s"
            print(f"{r.concurrency_level:<5} | {voice_label:<24} | {first_txt:<12} | {finish_txt:<10} | {gap_txt:<8} | {r.request_id}")
        print("=" * 95)


if __name__ == "__main__":
    asyncio.run(main())

