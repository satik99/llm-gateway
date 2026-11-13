#!/usr/bin/env bash
# Live check: streaming works end to end, and a client disconnect reaches the (mock) provider.
# Needs: mock on :9000, gateway on :8000 (bash load/run_gateway.sh). Run: bash load/check_stream.sh
GW=${GATEWAY_URL:-http://localhost:8000}; MOCK=${MOCK_URL:-http://localhost:9000}
KEY=${GATEWAY_KEY:-gw-load-key}
body='{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"tell me a story"}]}'
call() { curl -sSN "$@" "$GW/v1/chat/completions" -H "Authorization: Bearer $KEY" \
           -H "Content-Type: application/json" -d "$body"; }
check() { if echo "$2" | grep -q "$3"; then echo "   PASS: $1"; else echo "   FAIL: $1 (got $2)"; fi; }

# Preflight: say clearly what is missing instead of failing silently.
curl -sf "$MOCK/stats" >/dev/null || { echo "STOP: mock not reachable at $MOCK."
  echo "  Start it:  python tools/mock_upstream.py 9000   (venv active)"; exit 1; }
curl -sf "$GW/healthz" >/dev/null || { echo "STOP: gateway not reachable at $GW."
  echo "  Start it in another terminal:  bash load/run_gateway.sh   (venv active), wait ~5s,"
  echo "  and read that terminal: a Python traceback there means it crashed on startup."; exit 1; }

curl -s -X POST "$MOCK/stats/reset" >/dev/null

echo "1) full stream, read to the end (first 8 lines):"
call > /tmp/stream_out.txt; head -8 /tmp/stream_out.txt | cut -c1-110
sleep 1; s=$(curl -s "$MOCK/stats"); echo "   mock stats: $s"
check "stream completed upstream" "$s" '"streams_completed":1'
check "ends with [DONE]" "$(tail -3 /tmp/stream_out.txt)" '\[DONE\]'

echo "2) client hangs up after 0.4s (the mock would still need ~1s to finish):"
call --max-time 0.4 > /dev/null; sleep 1.5
s=$(curl -s "$MOCK/stats"); echo "   mock stats: $s"
check "upstream saw the disconnect" "$s" '"client_disconnects":1'
check "upstream did NOT finish the cancelled stream" "$s" '"streams_completed":1'
