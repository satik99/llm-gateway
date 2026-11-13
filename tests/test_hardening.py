import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.cache import ResponseCache, SingleFlight
from app.concurrency import ConcurrencyLimiter
from app.config import settings
from app.middleware import BodyLimitMiddleware, RequestIdMiddleware
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy

URL = "/v1/chat/completions"
H = {"Authorization": "Bearer gw-a"}


def body(text="hi", **kw):
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}], **kw}


async def no_sleep(_): pass


def slow_provider(delay, started=None):
    async def handler(request):
        if started is not None:
            started.set()
        await asyncio.sleep(delay)
        return httpx.Response(200, json={
            "model": "gpt-4o-mini", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name="openai")


@pytest.fixture
def client(monkeypatch):
    made = []

    def make(provider, cap=0):
        monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:1000")
        c = TestClient(main.app)
        c.__enter__()
        made.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.cache = None
        c.app.state.semantic = None
        c.app.state.store = None
        c.app.state.concurrency = ConcurrencyLimiter(cap)
        c.app.state.router = ReliableRouter([provider], RetryPolicy(max_attempts=1), sleep=no_sleep)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


# ---------- request ids ----------

def test_every_response_has_a_request_id_and_client_ids_are_honoured(client):
    c = client(slow_provider(0))
    r = c.post(URL, json=body(), headers=H)
    assert len(r.headers["x-request-id"]) == 16
    assert c.post(URL, json=body(), headers={**H, "X-Request-ID": "abc-12345-xyz"}).headers[
        "x-request-id"] == "abc-12345-xyz"
    assert c.post(URL, json=body(), headers={**H, "X-Request-ID": "bad id\twith junk"}).headers[
        "x-request-id"] != "bad id\twith junk"
    assert "x-request-id" in c.get("/healthz").headers
    assert "x-request-id" in c.post(URL, json=body(), headers={"Authorization": "Bearer no"}).headers


def test_access_log_line_has_request_id_and_outcome(client, caplog):
    import logging
    c = client(slow_provider(0))
    with caplog.at_level(logging.INFO, logger="gateway.access"):
        rid = c.post(URL, json=body(), headers=H).headers["x-request-id"]
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "gateway.access"]
    assert lines and lines[-1]["request_id"] == rid and lines[-1]["status"] == 200
    assert lines[-1]["key"] == "team-a" and lines[-1]["provider"] == "openai"


# ---------- body limit ----------

def test_oversized_body_is_rejected_with_413_before_parsing(client):
    c = client(slow_provider(0))
    r = c.post(URL, content=b"x" * (settings.max_body_bytes + 1), headers={**H, "Content-Type": "application/json"})
    assert r.status_code == 413 and "larger than" in r.json()["error"]
    assert c.post(URL, json=body("fine"), headers=H).status_code == 200


def test_limit_also_counts_chunked_bodies():
    app = FastAPI()

    @app.post("/echo")
    async def echo(request_body: dict):
        return {"ok": True}

    app.add_middleware(BodyLimitMiddleware, max_bytes=100)
    c = TestClient(app)

    def chunks():
        yield b'{"a":"'
        yield b"x" * 200
        yield b'"}'

    assert c.post("/echo", content=chunks(), headers={"Content-Type": "application/json"}).status_code == 413
    assert c.post("/echo", json={"a": "small"}).status_code == 200


# ---------- concurrency cap ----------

def test_slot_accounting():
    lim = ConcurrencyLimiter(2)
    a, b = lim.acquire("k"), lim.acquire("k")
    assert a and b and lim.acquire("k") is None and lim.acquire("other")
    a.release(); a.release()                         # idempotent
    c = lim.acquire("k")
    assert c is not None and lim.acquire("k") is None
    assert ConcurrencyLimiter(0).acquire("k") is not None      # 0 = unlimited


def test_concurrency_cap_returns_429_and_frees_slots(client):
    c = client(slow_provider(0.4), cap=2)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(5) as pool:
        codes = sorted(r.status_code for r in pool.map(
            lambda i: c.post(URL, json=body(f"q{i}", temperature=1), headers=H), range(5)))
    assert codes.count(200) == 2 and codes.count(429) == 3
    assert c.post(URL, json=body("after"), headers=H).status_code == 200      # slots were released
    assert c.app.state.concurrency._n["team-a"] == 0


# ---------- graceful shutdown (real processes) ----------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_up(url, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise AssertionError(f"{url} did not come up")


def test_sigterm_lets_in_flight_requests_finish():
    mock_port, gw_port = _free_port(), _free_port()
    root = os.path.dirname(os.path.dirname(__file__))
    env = {**os.environ, "MOCK_LATENCY_MS": "1500", "MOCK_JITTER_MS": "0"}
    mock = subprocess.Popen([sys.executable, "tools/mock_upstream.py", str(mock_port)], cwd=root,
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gw_env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{mock_port}/v1",
              "OPENAI_API_KEY": "mock", "GATEWAY_KEYS": "t:gw-t:100", "REDIS_URL": "redis://127.0.0.1:1/0",
              "DATABASE_URL": "", "EMBED_PROVIDER": "", "OLLAMA_MODEL": "", "ANTHROPIC_API_KEY": "",
              "ACCESS_LOG": "false"}
    gw = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(gw_port),
                           "--timeout-graceful-shutdown", "10", "--log-level", "warning"],
                          cwd=root, env=gw_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_up(f"http://127.0.0.1:{mock_port}/stats")
        _wait_up(f"http://127.0.0.1:{gw_port}/healthz")
        result = {}

        def call():
            result["r"] = httpx.post(f"http://127.0.0.1:{gw_port}{URL}", json=body("slow"),
                                     headers={"Authorization": "Bearer gw-t"}, timeout=15)

        t = threading.Thread(target=call)
        t.start()
        time.sleep(0.6)                      # request is now in flight (mock takes 1.5 s)
        gw.send_signal(signal.SIGTERM)
        t.join(10)
        assert result["r"].status_code == 200            # finished despite SIGTERM
        assert gw.wait(timeout=10) in (0, -signal.SIGTERM)   # exited on its own (uvicorn re-raises the signal)
        with pytest.raises(httpx.HTTPError):             # new connections are refused
            httpx.get(f"http://127.0.0.1:{gw_port}/healthz", timeout=1)
    finally:
        for p in (gw, mock):
            if p.poll() is None:
                p.kill()
