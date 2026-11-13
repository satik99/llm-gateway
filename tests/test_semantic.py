import asyncio

import fakeredis
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.cache import ResponseCache, SingleFlight
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy
from app.semantic import SemanticCache, bucket_key, query_text

URL = "/v1/chat/completions"
H = {"Authorization": "Bearer gw-a"}

# Fake embedder: fixed 3-d vectors per sentence; "refund" questions point the same way.
VECS = {
    "What is your refund policy?": [1.0, 0.0, 0.0],
    "How do I get my money back?": [0.98, 0.2, 0.0],      # cos ~0.98 with the above
    "How do I reset my password?": [0.0, 1.0, 0.0],       # cos 0
    "Almost refund": [0.9, 0.44, 0.0],                    # cos ~0.90
}


class FakeEmbedder:
    def __init__(self): self.calls = 0; self.fail = False
    async def embed(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedder down")
        return VECS[text]


def body(q, **kw):
    return {"model": "gpt-4o-mini", "temperature": 0,
            "messages": [{"role": "user", "content": q}], **kw}


def completion(text):
    return {"model": "gpt-4o-mini-2024-07-18",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}


# ---------- unit ----------

def test_query_text_only_plain_user_text():
    assert query_text(body("hi")) == "hi"
    assert query_text({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]}]}) == "a"
    assert query_text({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}) is None
    assert query_text({"messages": [{"role": "assistant", "content": "x"}]}) is None
    assert query_text({"messages": []}) is None


def test_bucket_ignores_last_message_but_not_context_or_params():
    a = body("one")
    assert bucket_key("t", a) == bucket_key("t", body("two"))
    assert bucket_key("t", a) != bucket_key("u", a)
    assert bucket_key("t", a) != bucket_key("t", body("one", max_tokens=5))
    sys1 = {**a, "messages": [{"role": "system", "content": "be terse"}, *a["messages"]]}
    sys2 = {**a, "messages": [{"role": "system", "content": "be verbose"}, *a["messages"]]}
    assert bucket_key("t", sys1) != bucket_key("t", sys2) != bucket_key("t", a)


async def test_store_lookup_threshold_and_best_match():
    r = fakeredis.FakeAsyncRedis()
    c = SemanticCache(r, threshold=0.95)
    assert await c.lookup("b", [1, 0, 0]) is None
    assert await c.store("b", [1, 0, 0], "q1", "openai", completion("A"))
    assert await c.store("b", [0, 1, 0], "q2", "openai", completion("B"))
    entry, sim = await c.lookup("b", [0.98, 0.2, 0])
    assert entry["body"]["choices"][0]["message"]["content"] == "A" and sim > 0.97
    assert await c.lookup("b", [0.9, 0.44, 0]) is None             # 0.90 < 0.95
    assert await c.lookup("other-bucket", [1, 0, 0]) is None        # buckets are isolated
    assert await c.store("b", [1, 0, 0], "e", "p", {"error": "x"}) is False


async def test_eviction_and_ttl_and_flush():
    r = fakeredis.FakeAsyncRedis()
    c = SemanticCache(r, ttl_s=60, max_entries=2)
    for i in range(4):
        await c.store("t:b", [1, i, 0], f"q{i}", "p", completion(str(i)))
    assert await r.hlen("semv:t:b") == 2 and await r.hlen("semb:t:b") == 2
    assert 0 < await r.ttl("semv:t:b") <= 60
    await c.store("u:b", [1, 0, 0], "q", "p", completion("x"))
    assert await c.flush("t") > 0 and await r.hlen("semv:t:b") == 0
    assert await r.hlen("semv:u:b") == 1


# ---------- API ----------

class FakeStore:
    def __init__(self): self.records = []
    async def record(self, r): self.records.append(r)
    async def close(self): pass
    async def summary(self, d): return []


async def no_sleep(_): pass


@pytest.fixture
def make_client(monkeypatch):
    made = []

    def make(providers, threshold=0.92):
        monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:1000,team-b:gw-b:1000")
        monkeypatch.setattr(settings, "admin_key", "secret")
        c = TestClient(main.app)
        c.__enter__()
        made.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.cache = ResponseCache(fake)
        c.app.state.semantic = SemanticCache(fake, threshold=threshold)
        c.app.state.embedder = FakeEmbedder()
        c.app.state.singleflight = SingleFlight()
        c.app.state.store = FakeStore()
        c.app.state.router = ReliableRouter(providers, RetryPolicy(max_attempts=1), sleep=no_sleep)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def upstream(name, calls, fail=None):
    async def handler(request: httpx.Request):
        calls[name] = calls.get(name, 0) + 1
        if fail and fail.get("on"):
            return httpx.Response(503, text="down")
        return httpx.Response(200, json=completion(f"from {name} #{calls[name]}"))
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


def test_similar_question_is_served_from_semantic_cache(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    r1 = c.post(URL, json=body("What is your refund policy?"), headers=H)
    r2 = c.post(URL, json=body("How do I get my money back?"), headers=H)
    assert r1.headers["x-gateway-cache"] == "miss"
    assert r2.headers["x-gateway-cache"] == "semantic-hit"
    assert float(r2.headers["x-gateway-similarity"]) > 0.97
    assert r2.json() == r1.json() and calls["openai"] == 1
    rec = c.app.state.store.records[-1]
    assert rec.cache_hit and rec.saved_usd > 0


def test_different_question_misses(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.post(URL, json=body("What is your refund policy?"), headers=H)
    r = c.post(URL, json=body("How do I reset my password?"), headers=H)
    assert r.headers["x-gateway-cache"] == "miss" and calls["openai"] == 2


def test_below_threshold_misses_and_raising_threshold_stops_hits(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)], threshold=0.85)
    c.post(URL, json=body("What is your refund policy?"), headers=H)
    assert c.post(URL, json=body("Almost refund"), headers=H).headers["x-gateway-cache"] == "semantic-hit"
    c2 = make_client([upstream("openai", {})], threshold=0.95)
    c2.post(URL, json=body("What is your refund policy?"), headers=H)
    assert c2.post(URL, json=body("Almost refund"), headers=H).headers["x-gateway-cache"] == "miss"


def test_other_key_system_prompt_and_params_never_share(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.post(URL, json=body("What is your refund policy?"), headers=H)
    q = "How do I get my money back?"
    other_key = c.post(URL, json=body(q), headers={"Authorization": "Bearer gw-b"})
    with_sys = body(q)
    with_sys["messages"].insert(0, {"role": "system", "content": "Answer in French"})
    r_sys = c.post(URL, json=with_sys, headers=H)
    r_tok = c.post(URL, json=body(q, max_tokens=5), headers=H)
    assert [r.headers["x-gateway-cache"] for r in (other_key, r_sys, r_tok)] == ["miss"] * 3


def test_fallback_answers_are_not_stored_semantically(make_client):
    calls, fail = {}, {"on": True}
    c = make_client([upstream("openai", calls, fail), upstream("ollama", calls)])
    r1 = c.post(URL, json=body("What is your refund policy?"), headers=H)
    assert r1.headers["x-gateway-provider"] == "ollama"
    fail["on"] = False
    r2 = c.post(URL, json=body("How do I get my money back?"), headers=H)
    assert r2.headers["x-gateway-cache"] == "miss" and r2.headers["x-gateway-provider"] == "openai"


def test_embedder_failure_falls_back_to_normal_path(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.app.state.embedder.fail = True
    r = c.post(URL, json=body("What is your refund policy?"), headers=H)
    assert r.status_code == 200 and r.headers["x-gateway-cache"] == "miss"


def test_non_cacheable_requests_skip_semantic(make_client):
    c = make_client([upstream("openai", {})])
    hot = body("What is your refund policy?", temperature=0.9)
    c.post(URL, json=hot, headers=H)
    assert c.app.state.embedder.calls == 0


def test_flush_clears_semantic_entries(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.post(URL, json=body("What is your refund policy?"), headers=H)
    assert c.post("/admin/cache/flush?key_name=team-a", headers={"X-Admin-Key": "secret"}).json()["deleted"] >= 2
    r = c.post(URL, json=body("How do I get my money back?"), headers=H)
    assert r.headers["x-gateway-cache"] == "miss"


def test_healthz_reports_semantic(make_client):
    c = make_client([upstream("openai", {})])
    assert "semantic_cache" in c.get("/healthz").json()


# ---------- wording guard ----------
from app.semantic import wording_conflict  # noqa: E402


@pytest.mark.parametrize("a,b,conflict", [
    ("How do I enable dark mode?", "How do I disable dark mode?", True),
    ("Convert 10 km to miles.", "Convert 10 miles to km.", True),
    ("Is the Pro plan $20 per month?", "Is the Pro plan $20 per year?", True),
    ("What is your refund policy?", "How do I get my money back?", False),   # real paraphrase
    ("Where can I find my invoices?", "How do I download my past bills?", False),
    ("What is your refund policy?", "what is your REFUND policy", False),     # same words
])
def test_wording_conflict(a, b, conflict):
    assert wording_conflict(a, b, 0.4) is conflict


async def test_lookup_skips_near_miss_even_when_embedding_says_identical():
    r = fakeredis.FakeAsyncRedis()
    c = SemanticCache(r, threshold=0.9)
    await c.store("b", [1, 0, 0], "How do I enable dark mode?", "p", completion("enable it"))
    assert await c.lookup("b", [1, 0, 0], "How do I disable dark mode?") is None
    assert (await c.lookup("b", [1, 0, 0], "turn the night theme on"))[0]["body"]
    assert await SemanticCache(r, threshold=0.9, max_wording_overlap=2).lookup(
        "b", [1, 0, 0], "How do I disable dark mode?") is not None            # guard can be disabled
