# LLM Gateway: day-by-day plan (~3 weeks part-time)

Rule: every day ends with something runnable and committed. Measure as you go; resume numbers come from week 3.

## Week 1: the minimum version (steps 1-3, 6, basic load test)
- **Day 1 (done in scaffold):** FastAPI app, auth, OpenAI provider, Redis token bucket + tests. Run it against the real API.
- **Day 2:** Anthropic provider + translate OpenAI-format requests/responses to/from Anthropic's Messages format. Add Ollama (it's OpenAI-compatible, so mostly a base_url).
- **Day 3:** Reliability: per-provider timeouts, retries with exponential backoff + jitter (only on 429/5xx/timeouts), fallback chain, circuit breaker (closed -> open -> half-open). Unit-test with a fake provider that fails on command.
- **Day 4:** Postgres: `api_keys`, `usage` tables (key, model, provider, prompt/completion tokens, cost, latency, cache_hit, ts). Move keys out of env. Cost from a per-model price table.
- **Day 5:** Token-per-minute limit (use `cost=estimated_tokens` in the same limiter; reconcile with real usage after the response). Quota per key per month.
- **Day 6:** docker-compose with redis + postgres + gateway. First k6 script against a mock upstream (a tiny FastAPI app with configurable latency/error rate, so load tests are free).
- **Day 7:** Streaming (SSE) pass-through. Handle client disconnect: cancel the upstream request so you stop paying for tokens nobody reads. Count tokens from the final usage chunk.

## Week 2: cache, metrics, dashboards
- **Day 8:** Exact-match cache: hash of (model, messages, temperature, other params), Redis with TTL. Only cache when temperature == 0 (or make it opt-in).
- **Day 9:** Semantic cache: embed the last user message, nearest-neighbor lookup (Redis vector search or pgvector), similarity threshold, scoped per model + system prompt so answers don't leak across contexts.
- **Day 10:** Prometheus metrics: request latency histogram, errors by provider, cache hit rate, tokens and cost by key. Grafana dashboard.
- **Day 11:** Admin API + minimal dashboard (create/revoke keys, usage per team per day).
- **Day 12-13:** Hardening: request size limits, graceful shutdown, per-key concurrency cap, logging with request IDs. Fix whatever load testing reveals.
- **Day 14:** Buffer / catch up.

## Week 3: measurement (this is what goes on the resume)
- **Day 15-16:** Build a replay workload (e.g. a public FAQ/support-style dataset with natural paraphrases, or generate paraphrases). Run with cache off vs exact vs semantic.
- **Day 17:** Semantic cache quality: label a sample of hits as right/wrong, plot hit rate vs wrong-answer rate across thresholds, pick your threshold and justify it.
- **Day 18:** Failure tests: kill/slow the upstream mid-load; record error rate, fallback latency, circuit breaker behavior.
- **Day 19:** Capacity test: p50/p95/p99, max sustained req/s, on your machine (state the hardware).
- **Day 20-21:** README with architecture diagram, results table, design decisions and tradeoffs. Record a 2-minute demo.

## Interview material to prepare
- Why a Lua script for the rate limiter (atomicity) and why Redis TIME (clock skew).
- Retry storms: why jitter + circuit breaker, and why not to retry non-idempotent client errors.
- Semantic cache risk: false hits. Threshold tradeoff, scoping, TTLs.
- Streaming: backpressure and cancelling upstream on disconnect.
- Token estimation before the call vs reconciliation after.
