import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

import anyio
import httpx
import redis.asyncio as redis
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.background import BackgroundTask

from .auth import ApiKey, authenticate
from .config import settings
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from . import metrics
from .concurrency import ConcurrencyLimiter, Slot
from .middleware import BodyLimitMiddleware, RequestIdFilter, RequestIdMiddleware
from .budget import SpendTracker
from .cache import ResponseCache, SingleFlight, cache_key as make_cache_key, is_cacheable
from .embeddings import Embedder
from .semantic import SemanticCache, bucket_key, query_text
from .pricing import estimate_cost
from .tokens import estimate_prompt_tokens, estimate_tokens
from .keystore import KeyStore, NameTaken, NotFound, hash_key, NAME_RE
from .store import UsageRecord, UsageStore
from .reliability import AllProvidersFailed, CircuitBreaker, ReliableRouter, RetryPolicy
from .ratelimit import TokenBucketLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared connection pool per process, not one per request.
    app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
    app.state.redis = redis.from_url(settings.redis_url)
    app.state.limiter = TokenBucketLimiter(app.state.redis)
    app.state.spend = SpendTracker(app.state.redis)
    app.state.cache = (ResponseCache(app.state.redis, settings.cache_ttl_s, settings.cache_max_bytes)
                       if settings.cache_ttl_s > 0 else None)
    app.state.singleflight = SingleFlight()
    app.state.semantic = app.state.embedder = None
    if settings.embed_provider and app.state.cache is not None:
        if settings.embed_provider == "ollama":
            emb = Embedder(app.state.http, settings.ollama_base_url,
                           settings.embed_model or "nomic-embed-text", prefix=settings.embed_prefix)
        elif settings.embed_provider == "openai":
            emb = Embedder(app.state.http, settings.openai_base_url,
                           settings.embed_model or "text-embedding-3-small", settings.openai_api_key,
                           prefix=settings.embed_prefix)
        else:
            raise RuntimeError("EMBED_PROVIDER must be 'ollama' or 'openai' (or empty)")
        app.state.embedder = emb
        app.state.semantic = SemanticCache(
            app.state.redis, settings.semantic_threshold, settings.cache_ttl_s,
            settings.semantic_max_entries, settings.cache_max_bytes,
            settings.semantic_max_wording_overlap)
    providers = [OpenAIProvider(
        app.state.http, name="openai", base_url=settings.openai_base_url,
        api_key=settings.openai_api_key)]
    if settings.anthropic_api_key:
        providers.append(AnthropicProvider(
            app.state.http, api_key=settings.anthropic_api_key, model=settings.anthropic_model))
    if settings.ollama_model:
        providers.append(OpenAIProvider(
            app.state.http, name="ollama", base_url=settings.ollama_base_url,
            model_override=settings.ollama_model))
    app.state.router = ReliableRouter(
        providers,
        policy=RetryPolicy(max_attempts=settings.retry_max_attempts),
        breaker_factory=lambda: CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_reset_timeout_s),
    )
    metrics.bind_breakers(app.state.router.breakers)
    app.state.concurrency = ConcurrencyLimiter(settings.max_concurrent_per_key)
    metrics.ACCESS_LOG_ENABLED = settings.access_log
    _setup_logging()
    try:
        await asyncio.wait_for(app.state.redis.ping(), 2)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("gateway").error(
            "REDIS UNREACHABLE at %s (%r): rate limits, budgets and the response cache are "
            "DISABLED (fail-open) until it is reachable",
            settings.redis_url.rsplit("@", 1)[-1], e)
    app.state.store = None
    if settings.database_url:
        try:
            app.state.store = await UsageStore.connect(settings.database_url)
        except Exception as e:  # noqa: BLE001  # run without usage tracking rather than not at all
            logging.getLogger("gateway").error("usage store unavailable: %r", e)
    app.state.keystore = None
    if settings.database_url:
        try:
            app.state.keystore = await KeyStore.connect(settings.database_url, settings.key_cache_ttl_s)
        except Exception as e:  # noqa: BLE001  # static .env keys keep working
            logging.getLogger("gateway").error("key store unavailable: %r", e)
    yield
    if app.state.keystore is not None:
        await app.state.keystore.close()
    if app.state.store is not None:
        await app.state.store.close()
    await app.state.http.aclose()
    await app.state.redis.aclose()


def _setup_logging() -> None:
    """If nothing configured logging (plain `uvicorn app.main:app`), log to stderr with request ids."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())


log = logging.getLogger("gateway")
app = FastAPI(title="LLM Gateway", lifespan=lifespan)
# Last added = outermost: request id -> metrics/access log -> body limit -> app
app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
app.add_middleware(metrics.MetricsMiddleware)
app.add_middleware(RequestIdMiddleware)


@app.get("/metrics")
async def prometheus_metrics(authorization: str = Header(default="")):
    if settings.metrics_token and not hmac.compare_digest(
            authorization.removeprefix("Bearer ").strip(), settings.metrics_token):
        raise HTTPException(403, "forbidden")
    return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz(request: Request):
    """`ok` means the gateway process is up. The other fields say which optional features
    are actually working: false here = that feature is silently skipped (fail-open)."""
    state = request.app.state
    try:
        await asyncio.wait_for(state.redis.ping(), 1)
        redis_ok = True
    except Exception:  # noqa: BLE001
        redis_ok = False
    return {"ok": True, "redis": redis_ok, "cache": redis_ok and state.cache is not None,
            "semantic_cache": redis_ok and state.semantic is not None,
            "usage_store": state.store is not None}


async def _redis_call(coro, default=None):
    """Run a Redis-backed limit check; on Redis failure, fail open and log.
    (Tradeoff: limits aren't enforced while Redis is unreachable.)"""
    try:
        return await coro
    except (redis.RedisError, OSError) as e:
        metrics.REDIS_ERRORS.inc()
        log.warning("redis unavailable, skipping limit check: %r", e)
        return default


def _too_many(result, message: str, key: str = "", reason: str = "") -> JSONResponse:
    metrics.LIMITED.labels(key, reason).inc()
    return JSONResponse(
        status_code=429,
        content={"error": message},
        headers={"Retry-After": str(max(1, result.retry_after_ms // 1000))},
    )


async def _stream_chat(request: Request, api_key: ApiKey, payload: dict, est: int, t0: float,
                       slot: Slot):
    """Proxy a streaming (SSE) chat completion.

    - Failover/retries happen only until the upstream sends its headers.
    - If the client disconnects, Starlette cancels this generator; the `finally` closes the
      upstream response, which drops the connection so the provider stops generating.
    - Usage comes from the final chunk (we ask for it with stream_options.include_usage);
      if it never arrives (disconnect, upstream ignores it) we estimate from what was sent.
    """
    state = request.app.state
    store, router = state.store, state.router
    tpm_bucket = f"{api_key.name}:tpm"
    requested_model = payload.get("model")
    up_payload = {**payload, "stream_options": {
        **(payload.get("stream_options") or {}), "include_usage": True}}

    def elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        provider_name, upstream = await router.open_stream(up_payload)
    except AllProvidersFailed as e:
        if est:
            await _redis_call(state.limiter.adjust(tpm_bucket, api_key.tpm, api_key.tpm, est))
        if store is not None:
            await store.record(UsageRecord(
                api_key.name, requested_model, None, "none", False, 0, 0, None,
                elapsed_ms(), 502))
        raise HTTPException(502, f"all providers failed: {e}")

    prompt_est = max(1, estimate_prompt_tokens(payload))

    async def account(status: int, usage: dict | None, served_model: str | None, chars: int):
        if usage and (usage.get("prompt_tokens") or usage.get("completion_tokens")):
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        else:  # no usage chunk: estimate from what we saw
            pt, ct = prompt_est, chars // 4
        served_model = served_model or requested_model
        cost = estimate_cost(provider_name, served_model, pt, ct)
        if est and (pt + ct) > 0:
            await _redis_call(state.limiter.adjust(
                tpm_bucket, api_key.tpm, api_key.tpm, est - (pt + ct)))
        if cost:
            await _redis_call(state.spend.add(api_key.name, cost))
        metrics.record_usage(api_key.name, provider_name, pt, ct, cost)
        if provider_name != router.providers[0].name:
            metrics.FALLBACKS.labels(provider_name).inc()
        if store is not None:
            await store.record(UsageRecord(
                api_key.name, requested_model, served_model, provider_name,
                provider_name != router.providers[0].name, pt, ct, cost,
                elapsed_ms(), status))

    async def body():
        status = 499                       # 499 = client closed the request (nginx convention)
        usage, served_model, chars, event = None, None, 0, []
        try:
            async for line in upstream.aiter_lines():
                if line == "":             # blank line = end of one SSE event
                    if event:
                        yield ("\n".join(event) + "\n\n").encode()
                        event = []
                    continue
                event.append(line)
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            obj = None
                        if isinstance(obj, dict):
                            served_model = obj.get("model") or served_model
                            if obj.get("usage"):
                                usage = obj["usage"]
                            for choice in obj.get("choices") or []:
                                chars += len((choice.get("delta") or {}).get("content") or "")
            if event:
                yield ("\n".join(event) + "\n\n").encode()
            status = 200
        except httpx.HTTPError as e:       # upstream died after we had started streaming
            log.warning("stream from %s failed mid-way: %r", provider_name, e)
            router.record_stream_failure(provider_name)
            status = 502
            yield b'data: {"error": {"message": "upstream stream failed"}}\n\n'
        finally:
            # Shielded: still runs to completion when this task is cancelled by a disconnect.
            with anyio.CancelScope(shield=True):
                slot.release()
                await upstream.aclose()    # closes the upstream connection -> stops generation
                await account(status, usage, served_model, chars)

    async def cleanup():                   # runs even if the generator never started
        slot.release()
        await upstream.aclose()

    slot.hand_off()                        # from here the stream (not the route) releases the slot
    return StreamingResponse(
        body(), media_type="text/event-stream",
        headers={"X-Gateway-Provider": provider_name, "Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"},
        background=BackgroundTask(cleanup),   # covers "client left before we started"
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, api_key: ApiKey = Depends(authenticate)
):
    t0 = time.perf_counter()
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(400, "request body must be valid JSON")
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise HTTPException(400, "'messages' must be a list of message objects")

    state = request.app.state
    store, router = state.store, state.router
    requested_model = payload.get("model")
    tpm_bucket = f"{api_key.name}:tpm"

    def latency_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    # 1. Monthly budget (soft limit: spend is only known after responses return).
    if api_key.monthly_budget_usd > 0:
        spent = await _redis_call(state.spend.get(api_key.name), 0.0)
        if spent >= api_key.monthly_budget_usd:
            metrics.LIMITED.labels(api_key.name, "budget").inc()
            return JSONResponse(status_code=402, content={
                "error": "monthly budget exceeded",
                "spent_usd": round(spent, 6),
                "budget_usd": api_key.monthly_budget_usd})

    # 2. Requests per minute.
    result = await _redis_call(state.limiter.check(
        bucket=f"{api_key.name}:rpm", capacity=api_key.rpm, per_minute=api_key.rpm))
    if result is not None and not result.allowed:
        return _too_many(result, "rate limit exceeded", api_key.name, "rpm")

    # 3. Response cache (exact match). Hits cost no tokens and no money, so they skip the
    #    token limit and budget accounting, but they still counted against the request limit.
    cache, ckey = state.cache, None
    sem_vec = sem_bucket = text = None

    def _hit_response(entry: dict, kind: str, similarity: float | None) -> JSONResponse:
        hit_body, hit_provider = entry["body"], entry["provider"]
        u = hit_body.get("usage") or {}
        saved = estimate_cost(hit_provider, hit_body.get("model"),
                              u.get("prompt_tokens", 0), u.get("completion_tokens", 0)) or 0.0
        hit_record = UsageRecord(
            api_key.name, requested_model, hit_body.get("model"), hit_provider, False,
            0, 0, 0.0, latency_ms(), 200, cache_hit=True, saved_usd=saved)
        metrics.record_usage(api_key.name, hit_provider, 0, 0, None, saved)
        headers = {"X-Gateway-Provider": hit_provider, "X-Gateway-Cache": kind}
        if similarity is not None:
            headers["X-Gateway-Similarity"] = f"{similarity:.4f}"
        return JSONResponse(
            hit_body, headers=headers,
            background=BackgroundTask(store.record, hit_record) if store is not None else None)

    if cache is not None and is_cacheable(payload, request.headers):
        ckey = make_cache_key(api_key.name, payload)
        entry = await _redis_call(cache.get(ckey))
        if entry is not None:
            return _hit_response(entry, "hit", None)

        # 3b. Semantic cache: a differently-worded but same-meaning question seen before.
        sem = state.semantic
        text = query_text(payload) if sem is not None else None
        if text is not None:
            t_emb = time.perf_counter()
            try:
                sem_vec = await asyncio.wait_for(state.embedder.embed(text), settings.embed_timeout_s)
                metrics.EMBED_LATENCY.observe(time.perf_counter() - t_emb)
                sem_bucket = bucket_key(api_key.name, payload)
                found = await sem.lookup(sem_bucket, sem_vec, text)
            except Exception as e:  # noqa: BLE001  # embedder/redis trouble: just skip, never fail
                metrics.EMBED_ERRORS.inc()
                log.warning("semantic lookup skipped: %r", e)
                sem_vec, found = None, None
            if found is not None:
                return _hit_response(found[0], "semantic-hit", found[1])

    # 3c. Per-key concurrency cap (cache hits above never occupy a slot).
    slot = state.concurrency.acquire(api_key.name)
    if slot is None:
        metrics.LIMITED.labels(api_key.name, "concurrency").inc()
        return JSONResponse(status_code=429, headers={"Retry-After": "1"}, content={
            "error": "too many concurrent requests for this key",
            "limit": settings.max_concurrent_per_key})
    try:
        # 4. Tokens per minute: charge an estimate now, reconcile with real usage after.
        est = 0
        if api_key.tpm > 0:
            est = estimate_tokens(payload)
            if est > api_key.tpm:
                metrics.LIMITED.labels(api_key.name, "too_large").inc()
                return JSONResponse(status_code=413, content={
                    "error": "request too large for this key's token limit",
                    "estimated_tokens": est, "limit_per_minute": api_key.tpm})
            result = await _redis_call(state.limiter.check(
                bucket=tpm_bucket, capacity=api_key.tpm, per_minute=api_key.tpm, cost=est))
            if result is not None and not result.allowed:
                return _too_many(result, "token rate limit exceeded", api_key.name, "tpm")

        if payload.get("stream"):
            return await _stream_chat(request, api_key, payload, est, t0, slot)

        async def fetch():
            pn, bd = await router.chat(payload)
            # Never cache a fallback provider's answer: it would outlive the outage.
            if ckey is not None and pn == router.providers[0].name:
                await _redis_call(cache.set(ckey, pn, bd))
            return pn, bd

        leader = True
        try:
            if ckey is not None:
                (provider_name, body), leader = await state.singleflight.do(ckey, fetch)
            else:
                provider_name, body = await fetch()
        except AllProvidersFailed as e:
            if est:  # nothing was generated, give the estimate back
                await _redis_call(state.limiter.adjust(tpm_bucket, api_key.tpm, api_key.tpm, est))
            if store is not None:
                await store.record(UsageRecord(
                    api_key.name, requested_model, None, "none", False, 0, 0, None,
                    latency_ms(), 502))
            raise HTTPException(502, f"all providers failed: {e}")

        usage = body.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        served_model = body.get("model")
        cost = estimate_cost(provider_name, served_model, pt, ct)
        is_fallback = provider_name != router.providers[0].name
        if leader:
            record = UsageRecord(
                key_name=api_key.name, requested_model=requested_model,
                served_model=served_model, provider=provider_name, fallback=is_fallback,
                prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
                latency_ms=latency_ms(), status=200)
            cache_header = "miss" if ckey is not None else None
        else:  # coalesced: rode along on another request's upstream call, paid nothing
            record = UsageRecord(
                key_name=api_key.name, requested_model=requested_model,
                served_model=served_model, provider=provider_name, fallback=is_fallback,
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                latency_ms=latency_ms(), status=200, cache_hit=True, saved_usd=cost or 0.0)
            cache_header = "coalesced"

        async def after_response():
            # Runs after the response is sent, so none of this adds latency.
            if leader:
                metrics.record_usage(api_key.name, provider_name, pt, ct, cost)
                if is_fallback:
                    metrics.FALLBACKS.labels(provider_name).inc()
                if est and (pt + ct) > 0:
                    await _redis_call(state.limiter.adjust(
                        tpm_bucket, api_key.tpm, api_key.tpm, est - (pt + ct)))
                if cost:
                    await _redis_call(state.spend.add(api_key.name, cost))
            else:
                metrics.record_usage(api_key.name, provider_name, 0, 0, None, cost or 0.0)
            if not leader and est:  # follower used no tokens: give its estimate back
                await _redis_call(state.limiter.adjust(tpm_bucket, api_key.tpm, api_key.tpm, est))
            if store is not None:
                await store.record(record)
            # Remember this answer for similar future questions (primary provider only).
            if leader and sem_vec is not None and not is_fallback:
                try:
                    await state.semantic.store(sem_bucket, sem_vec, text, provider_name, body)
                except Exception as e:  # noqa: BLE001
                    log.warning("semantic store skipped: %r", e)

        headers = {"X-Gateway-Provider": provider_name}
        if cache_header:
            headers["X-Gateway-Cache"] = cache_header
        return JSONResponse(body, headers=headers, background=BackgroundTask(after_response))
    finally:
        if not slot.handed_off:
            slot.release()


def _require_admin(x_admin_key: str) -> None:
    if not settings.admin_key or not hmac.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(403, "forbidden")


@app.post("/admin/cache/flush")
async def admin_cache_flush(
    request: Request, key_name: str | None = None, x_admin_key: str = Header(default="")
):
    """Delete cached responses: for one key name, or all of them when key_name is omitted."""
    _require_admin(x_admin_key)
    cache = request.app.state.cache
    if cache is None:
        raise HTTPException(503, "cache is disabled")
    deleted = await cache.flush(key_name)
    if request.app.state.semantic is not None:
        deleted += await request.app.state.semantic.flush(key_name)
    return {"deleted": deleted}


class NewKey(BaseModel):
    name: str
    rpm: int = Field(gt=0, le=1_000_000)
    tpm: int = Field(default=0, ge=0)
    monthly_budget_usd: float = Field(default=0.0, ge=0)


class KeyChanges(BaseModel):
    rpm: int | None = Field(default=None, gt=0, le=1_000_000)
    tpm: int | None = Field(default=None, ge=0)
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    active: bool | None = None


def _keystore(request: Request) -> KeyStore:
    ks = request.app.state.keystore
    if ks is None:
        raise HTTPException(503, "key management needs DATABASE_URL")
    return ks


def _static_names() -> set[str]:
    from .auth import _load_keys
    return {k.name for k in _load_keys().values()}


@app.post("/admin/keys", status_code=201)
async def admin_create_key(request: Request, body: NewKey, x_admin_key: str = Header(default="")):
    """Create a key. The plaintext `key` in the response is shown ONCE and cannot be recovered."""
    _require_admin(x_admin_key)
    if not NAME_RE.match(body.name):
        raise HTTPException(422, "name must match [a-z0-9][a-z0-9_-]{0,62}")
    if body.name in _static_names():
        raise HTTPException(409, "name is used by a key defined in GATEWAY_KEYS")
    try:
        info, token = await _keystore(request).create(
            body.name, body.rpm, body.tpm, body.monthly_budget_usd)
    except NameTaken:
        raise HTTPException(409, "a key with this name already exists")
    return {**info, "key": token}


@app.get("/admin/keys")
async def admin_list_keys(request: Request, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)
    return {"keys": await _keystore(request).list()}


@app.patch("/admin/keys/{name}")
async def admin_update_key(request: Request, name: str, body: KeyChanges,
                           x_admin_key: str = Header(default="")):
    """Change limits or (re)activate/deactivate. Takes effect within KEY_CACHE_TTL_S seconds."""
    _require_admin(x_admin_key)
    try:
        return await _keystore(request).update(name, body.model_dump(exclude_none=True))
    except NotFound:
        raise HTTPException(404, "no such key")


@app.post("/admin/keys/{name}/rotate")
async def admin_rotate_key(request: Request, name: str, x_admin_key: str = Header(default="")):
    """Issue a new secret for the same name (limits, spend and history carry over).
    The old secret stops working immediately."""
    _require_admin(x_admin_key)
    try:
        info, token = await _keystore(request).rotate(name)
    except NotFound:
        raise HTTPException(404, "no such active key")
    return {**info, "key": token}


@app.delete("/admin/keys/{name}")
async def admin_revoke_key(request: Request, name: str, x_admin_key: str = Header(default="")):
    """Revoke (soft delete: the row stays so past usage keeps its name)."""
    _require_admin(x_admin_key)
    try:
        return await _keystore(request).revoke(name)
    except NotFound:
        raise HTTPException(404, "no such key")


@app.get("/admin/usage/daily")
async def admin_usage_daily(
    request: Request, days: int = 7, x_admin_key: str = Header(default="")
):
    """Per team (key name) per day: requests, cache hits, errors, tokens, cost, savings."""
    _require_admin(x_admin_key)
    store = request.app.state.store
    if store is None:
        raise HTTPException(503, "usage tracking is not configured")
    days = max(1, min(days, 365))
    return {"days": days, "rows": await store.daily(days)}


@app.get("/admin/usage")
async def admin_usage(
    request: Request, days: int = 7, x_admin_key: str = Header(default="")
):
    _require_admin(x_admin_key)
    store = request.app.state.store
    if store is None:
        raise HTTPException(503, "usage tracking is not configured")
    days = max(1, min(days, 365))
    return {"days": days, "rows": await store.summary(days)}
