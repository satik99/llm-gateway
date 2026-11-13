"""Prometheus metrics. One private registry (not the global default) so tests stay isolated.

Labels are deliberately low-cardinality: key NAME (a handful of teams), provider (2-3),
status code, cache outcome. Never model names from user input, never prompts, never IDs.

Runs per process: with N uvicorn workers each exposes its own counters. Run one worker per
container/process and let Prometheus scrape each (or add multiprocess mode later).
"""
import json
import logging
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()
access_log = logging.getLogger("gateway.access")
ACCESS_LOG_ENABLED = True       # set from settings at startup

_BUCKETS = (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60)

REQUESTS = Counter("gateway_requests_total", "Chat completion requests by outcome.",
                   ["key", "provider", "status", "cache"], registry=REGISTRY)
LATENCY = Histogram("gateway_request_duration_seconds",
                    "Total time to serve a chat completion (streams: until the last byte).",
                    ["provider", "cache"], buckets=_BUCKETS, registry=REGISTRY)
INFLIGHT = Gauge("gateway_inflight_requests", "Chat completions currently being served.",
                 registry=REGISTRY)
TOKENS = Counter("gateway_tokens_total", "Tokens billed by the provider (cache hits excluded).",
                 ["key", "provider", "type"], registry=REGISTRY)
COST = Counter("gateway_cost_usd_total", "Estimated spend in USD.", ["key", "provider"],
               registry=REGISTRY)
SAVED = Counter("gateway_saved_usd_total", "Estimated USD not spent thanks to the cache.",
                ["key"], registry=REGISTRY)
LIMITED = Counter("gateway_limited_total", "Requests rejected before reaching a provider.",
                  ["key", "reason"], registry=REGISTRY)      # rpm | tpm | budget | too_large
FALLBACKS = Counter("gateway_fallbacks_total", "Requests served by a non-primary provider.",
                    ["provider"], registry=REGISTRY)
CIRCUIT = Gauge("gateway_circuit_state", "Circuit breaker: 0 closed, 1 half-open, 2 open.",
                ["provider"], registry=REGISTRY)
EMBED_LATENCY = Histogram("gateway_embedding_duration_seconds", "Semantic-cache embedding call.",
                          buckets=_BUCKETS, registry=REGISTRY)
EMBED_ERRORS = Counter("gateway_embedding_errors_total",
                       "Embedding/semantic lookups skipped because of an error or timeout.",
                       registry=REGISTRY)
REDIS_ERRORS = Counter("gateway_redis_errors_total", "Redis calls that failed (limits fail open).",
                       registry=REGISTRY)

_STATE = {"closed": 0, "half_open": 1, "open": 2}


def bind_breakers(breakers: dict) -> None:
    """Expose each provider's breaker state, read live at scrape time."""
    for name, br in breakers.items():
        CIRCUIT.labels(name).set_function(lambda br=br: _STATE.get(br.state, 0))


def record_usage(key: str, provider: str, prompt_tokens: int, completion_tokens: int,
                 cost: float | None, saved: float = 0.0) -> None:
    if prompt_tokens:
        TOKENS.labels(key, provider, "prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(key, provider, "completion").inc(completion_tokens)
    if cost:
        COST.labels(key, provider).inc(cost)
    if saved:
        SAVED.labels(key).inc(saved)


class MetricsMiddleware:
    """Pure-ASGI middleware: sees every response (incl. 401/402/429/502 and streams) without
    touching the handlers. Provider and cache outcome are read from the response headers the
    gateway already sets; the key name is left in the ASGI scope by `authenticate`."""

    PATH = "/v1/chat/completions"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != self.PATH:
            return await self.app(scope, receive, send)
        t0 = time.perf_counter()
        seen = {"status": None, "headers": {}, "done": False}
        INFLIGHT.inc()

        def finish(status: int) -> None:
            if seen["done"]:
                return
            seen["done"] = True
            h = seen["headers"]
            provider, cache = h.get("x-gateway-provider", "none"), h.get("x-gateway-cache", "none")
            key = scope.get("gateway_key", "unauthenticated")
            REQUESTS.labels(key, provider, str(status), cache).inc()
            elapsed = time.perf_counter() - t0
            LATENCY.labels(provider, cache).observe(elapsed)
            if ACCESS_LOG_ENABLED:
                access_log.info(json.dumps({
                    "request_id": scope.get("request_id"), "key": key, "status": status,
                    "provider": provider, "cache": cache, "ms": round(elapsed * 1000, 1)}))

        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                seen["status"] = message["status"]
                seen["headers"] = {k.decode().lower(): v.decode() for k, v in message["headers"]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finish(seen["status"] or 500)

        try:
            await self.app(scope, receive, wrapped_send)
        except BaseException:
            finish(seen["status"] or 500)
            raise
        finally:
            # Response never completed: the client went away mid-stream -> 499 (nginx convention)
            finish(499 if seen["status"] is not None else 500)
            INFLIGHT.dec()
