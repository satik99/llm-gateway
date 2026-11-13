# Load testing

Everything runs locally against mock upstreams, so it costs nothing.

## One-time setup
    docker compose up -d redis postgres
    docker exec llm-gateway-postgres-1 createdb -U gateway gateway_load

## Run (4 terminals, all from the project root, venv active)
    python tools/mock_upstream.py 9000          # 1) fake "openai"
    python tools/mock_upstream.py 9001          # 2) fake "ollama" (the fallback)
    bash load/run_gateway.sh                    # 3) gateway with load-test config (.env.load)
    docker run --rm -i --network host -v "$PWD":/work -w /work \
      grafana/k6 run -e RATE=100 -e SECONDS=90 load/k6_gateway.js          # 4) k6

The summary prints at the end. Add `-e SAVE_JSON=1` to also write load/results/summary.json
(needs `chmod 777 load/results` when k6 runs in Docker).

## Knobs
- `-e OUTAGE=0`: no injected failure. Use this for the baseline and max-throughput runs.
- `-e GATEWAY_URL=http://localhost:9000`: send the same load straight at the mock = your baseline.
- `RATE`, `SECONDS` (k6 -e flags): arrival rate and total duration. The outage is the middle third.
- `BREAKER_RESET_TIMEOUT_S=3 bash load/run_gateway.sh`: how fast traffic returns to the primary after it recovers.
- Mock behaviour at runtime: `curl -X POST localhost:9000/control -d '{"latency_ms": 800, "error_rate": 0.2}'`

## Honest-numbers rules
- Report hardware, RATE, mock latency, and that upstreams are mocks.
- Reset between runs: `redis-cli -n 1 flushdb` (docker: `docker exec llm-gateway-redis-1 redis-cli -n 1 flushdb`).
- Find max throughput by raising RATE until errors or p95 climb; report the last clean rate.

## Measuring gateway overhead and max throughput
    # baseline: straight at the mock (no gateway)
    k6 run --quiet -e RATE=100 -e SECONDS=30 -e OUTAGE=0 -e GATEWAY_URL=http://localhost:9000 load/k6_gateway.js
    # same load through the gateway; overhead = difference in p50/p95
    k6 run --quiet -e RATE=100 -e SECONDS=30 -e OUTAGE=0 load/k6_gateway.js
    # then raise RATE (200, 300, 400...) until p95 climbs or errors appear
Note: one uvicorn process uses one CPU core. `bash load/run_gateway.sh --workers 4` scales across cores,
but the circuit breaker state is then per worker (rate limits and budgets live in Redis, so they stay global).

## Streaming check
    bash load/check_stream.sh     # needs mock on :9000 and the gateway from load/run_gateway.sh
Proves a full stream completes and that a client hang-up reaches the upstream (mock counts disconnects at GET :9000/stats).

## Cache benchmark (cache off vs on, one command)
    bash load/cache_benchmark.sh
    RATE=200 SECS=30 DISTINCT=1000 bash load/cache_benchmark.sh     # harsher: fewer repeats
Needs the mock on :9000 and the gateway from `bash load/run_gateway.sh`. It flushes the cache first, runs the same
workload with temperature unset (not cacheable) and with temperature 0 (cacheable), and compares upstream calls
(= API spend) and latency. The hit rate depends entirely on how often questions repeat, so ALWAYS report the
workload (`DISTINCT`, rate, duration) next to the number.
