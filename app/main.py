import logging
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import ApiKey, authenticate
from .config import settings
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .reliability import AllProvidersFailed, CircuitBreaker, ReliableRouter, RetryPolicy
from .ratelimit import TokenBucketLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared connection pool per process, not one per request.
    app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
    app.state.redis = redis.from_url(settings.redis_url)
    app.state.limiter = TokenBucketLimiter(app.state.redis)
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
    yield
    await app.state.http.aclose()
    await app.state.redis.aclose()


log = logging.getLogger("gateway")
app = FastAPI(title="LLM Gateway", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, api_key: ApiKey = Depends(authenticate)
):
    payload = await request.json()
    if payload.get("stream"):
        raise HTTPException(501, "streaming arrives on day 7")

    # Rate limit per key: burst = rpm, sustained = rpm per minute.
    try:
        result = await request.app.state.limiter.check(
            bucket=f"{api_key.name}:rpm", capacity=api_key.rpm, per_minute=api_key.rpm
        )
    except (redis.RedisError, OSError) as e:
        # Fail open: a Redis outage shouldn't take the whole gateway down.
        # (Tradeoff: limits aren't enforced while Redis is unreachable.)
        log.warning("rate limiter unavailable, allowing request: %r", e)
        result = None
    if result is not None and not result.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded"},
            headers={"Retry-After": str(max(1, result.retry_after_ms // 1000))},
        )

    try:
        provider_name, body = await request.app.state.router.chat(payload)
    except AllProvidersFailed as e:
        raise HTTPException(502, f"all providers failed: {e}")
    return JSONResponse(body, headers={"X-Gateway-Provider": provider_name})
