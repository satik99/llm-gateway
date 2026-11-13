#!/usr/bin/env bash
# Cache off vs cache on, same workload, one command. Prints the numbers for your README/resume.
#   bash load/cache_benchmark.sh                 # defaults: 100 req/s, 20s, 50 distinct questions
#   RATE=200 SECS=30 DISTINCT=200 bash load/cache_benchmark.sh
# Needs: mock on :9000, gateway on :8000 via `bash load/run_gateway.sh`. K6_BIN=/path/to/k6 skips Docker.
GW=${GATEWAY_URL:-http://localhost:8000}; MOCK=${MOCK_URL:-http://localhost:9000}
ADMIN=${ADMIN_KEY:-load-admin}; RATE=${RATE:-100}; SECS=${SECS:-20}; DISTINCT=${DISTINCT:-50}

k6run() {
  if [ -n "$K6_BIN" ]; then "$K6_BIN" run --quiet "$@"
  else docker run --rm -i --network host -v "$PWD":/work -w /work grafana/k6 run --quiet "$@"; fi
}
curl -sf "$MOCK/stats" >/dev/null || { echo "STOP: mock not reachable at $MOCK (python tools/mock_upstream.py 9000)"; exit 1; }
curl -sf "$GW/healthz" >/dev/null || { echo "STOP: gateway not reachable at $GW (bash load/run_gateway.sh, wait ~5s)"; exit 1; }

run_case() {   # $1 label, $2 CACHE=0|1
  curl -s -X POST "$GW/admin/cache/flush" -H "X-Admin-Key: $ADMIN" >/dev/null   # start cold
  curl -s -X POST "$MOCK/stats/reset" >/dev/null
  out=$(k6run -e RATE="$RATE" -e SECONDS="$SECS" -e OUTAGE=0 -e CACHE="$2" -e DISTINCT="$DISTINCT" \
        -e GATEWAY_KEY="${GATEWAY_KEY:-gw-load-key}" -e GATEWAY_URL="$GW" load/k6_gateway.js 2>&1)
  sleep 1
  sent=$(echo "$out" | sed -n 's/.*requests sent *: *\([0-9]*\).*/\1/p')
  lat=$(echo "$out" | sed -n 's/.*latency p50 \/ p95 \/ p99 : \(.*\) ms/\1/p')
  err=$(echo "$out" | sed -n 's/.*error rate *: *\([0-9.]*\) %.*/\1/p')
  ch=$(echo "$out" | sed -n 's/.*cache hit\/coalesced\/miss : \(.*\)/\1/p')
  up=$(curl -s "$MOCK/stats" | sed -n 's/.*"requests":\([0-9]*\).*/\1/p')
  echo "$1|$sent|$lat|$err|$ch|$up"
}

echo "workload: $RATE req/s for ${SECS}s, $DISTINCT distinct questions, mock upstream latency ~300ms"
off=$(run_case "cache off" 0); on=$(run_case "cache on " 1)
echo
printf "%-10s %9s %-22s %7s %-18s %s\n" "" "requests" "p50 / p95 / p99 (ms)" "errors" "hit/coal/miss" "upstream calls"
for row in "$off" "$on"; do IFS='|' read -r l n lat e ch up <<<"$row"
  printf "%-10s %9s %-22s %6s%% %-18s %s\n" "$l" "$n" "$lat" "$e" "${ch:-0 / 0 / 0}" "$up"; done
IFS='|' read -r _ _ _ _ _ up_off <<<"$off"; IFS='|' read -r _ n_on lat_on _ _ up_on <<<"$on"
IFS='|' read -r _ _ lat_off _ _ _ <<<"$off"
p95() { echo "$1" | awk -F'/' '{gsub(/ /,"",$2); print $2}'; }
awk -v a="$up_off" -v b="$up_on" -v po="$(p95 "$lat_off")" -v pn="$(p95 "$lat_on")" 'BEGIN{
  if (a>0) printf "\nupstream calls (= API spend) reduced by %.1f%%  (%d -> %d)\n", (1-b/a)*100, a, b;
  if (po>0) printf "p95 latency reduced by %.1f%%  (%d ms -> %d ms)\n", (1-pn/po)*100, po, pn }'
