"""Fake OpenAI-compatible upstream for free, repeatable load and failure tests.

    python tools/mock_upstream.py 9000      # primary  ("openai")
    python tools/mock_upstream.py 9001      # fallback ("ollama")

Change behaviour at runtime, no restart needed:
    curl -X POST localhost:9000/control -H 'Content-Type: application/json' \
         -d '{"error_rate": 1.0}'           # simulate a full outage
    curl -X POST localhost:9000/control -d '{"error_rate": 0, "latency_ms": 800}'
"""
import asyncio
import json
import os
import random
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

cfg = {
    "latency_ms": float(os.getenv("MOCK_LATENCY_MS", 300)),   # typical LLM call is far slower
    "jitter_ms": float(os.getenv("MOCK_JITTER_MS", 100)),
    "error_rate": float(os.getenv("MOCK_ERROR_RATE", 0)),
    "error_status": int(os.getenv("MOCK_ERROR_STATUS", 503)),
    "stream_chunks": int(os.getenv("MOCK_STREAM_CHUNKS", 20)),      # tokens per streamed reply
    "stream_delay_ms": float(os.getenv("MOCK_STREAM_DELAY_MS", 50)),  # gap between tokens
    "stream_fail_after": int(os.getenv("MOCK_STREAM_FAIL_AFTER", 0)),  # >0: drop mid-stream
}
# Counters: lets a test prove the client disconnect reached the "provider".
stats = {"requests": 0, "streams_started": 0, "streams_completed": 0, "client_disconnects": 0, "stream_failures": 0}

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    stats["requests"] += 1          # every upstream call = money in real life
    delay = max(0.0, cfg["latency_ms"] + random.uniform(-cfg["jitter_ms"], cfg["jitter_ms"]))
    await asyncio.sleep(delay / 1000)
    if random.random() < cfg["error_rate"]:
        return JSONResponse({"error": {"message": "mock upstream failure"}},
                            status_code=cfg["error_status"])
    chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    if body.get("stream"):
        return StreamingResponse(
            stream_events(body, max(1, chars // 4)), media_type="text/event-stream")
    pt, ct = max(1, chars // 4), 40
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": body.get("model", "mock-model"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "This is a mock response."}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


async def stream_events(body: dict, pt: int):
    stats["streams_started"] += 1
    model = body.get("model", "mock-model")
    n = cfg["stream_chunks"]

    def chunk(delta: dict, finish=None, usage=None):
        obj = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": model,
               "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n".encode()

    try:
        yield chunk({"role": "assistant", "content": ""})
        for i in range(n):
            if cfg["stream_fail_after"] and i >= cfg["stream_fail_after"]:
                stats["stream_failures"] += 1
                raise RuntimeError("mock mid-stream failure")   # connection drops abruptly
            await asyncio.sleep(cfg["stream_delay_ms"] / 1000)
            yield chunk({"content": "word "})
        yield chunk({}, finish="stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            yield chunk({}, usage={"prompt_tokens": pt, "completion_tokens": n,
                                   "total_tokens": pt + n})
        yield b"data: [DONE]\n\n"
        stats["streams_completed"] += 1
    except asyncio.CancelledError:      # the caller hung up: we'd stop paying for tokens here
        stats["client_disconnects"] += 1
        raise


@app.get("/stats")
async def show_stats():
    return stats


@app.post("/stats/reset")
async def reset_stats():
    for k in stats:
        stats[k] = 0
    return stats


@app.post("/control")
async def control(request: Request):
    cfg.update(await request.json())
    return cfg


@app.get("/control")
async def show():
    return cfg


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 9000,
                log_level="warning")
