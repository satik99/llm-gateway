#!/usr/bin/env bash
# Ramp the request rate through the gateway until it saturates, one summary line per rate.
#   bash load/ramp.sh "1 worker"                  # default rates
#   bash load/ramp.sh "4 workers" 600 800 1000    # custom rates
# Stops at the first rate where achieved < 90% of target, errors > 1%, or p95 > 1000 ms.
# Set K6_BIN=/path/to/k6 to use a local k6 instead of Docker. SECS=20 by default.
label=${1:-run}; shift
rates=${*:-"400 600 800 1000 1500"}
secs=${SECS:-20}

k6run() {
  if [ -n "$K6_BIN" ]; then "$K6_BIN" run --quiet "$@"
  else docker run --rm -i --network host -v "$PWD":/work -w /work grafana/k6 run --quiet "$@"; fi
}

echo "== $label =="
for r in $rates; do
  out=$(k6run -e RATE="$r" -e SECONDS="$secs" -e OUTAGE=0 load/k6_gateway.js 2>&1)
  ach=$(echo "$out" | sed -n 's/.*achieved \([0-9.]*\) req\/s.*/\1/p' | head -1)
  lat=$(echo "$out" | sed -n 's/.*latency p50 \/ p95 \/ p99 : \(.*\) ms/\1/p')
  err=$(echo "$out" | sed -n 's/.*error rate *: \([0-9.]*\) %.*/\1/p')
  p95=$(echo "$lat" | awk -F'/' '{gsub(/ /,"",$2); print $2}')
  echo "rate $r -> achieved ${ach:-?}/s | p50/p95/p99 ${lat:-?} ms | errors ${err:-?}%"
  bad=$(awk -v r="$r" -v a="${ach:-0}" -v e="${err:-100}" -v p="${p95:-99999}" \
        'BEGIN{ print (a < 0.9*r || e > 1 || p > 1000) ? 1 : 0 }')
  if [ "$bad" = "1" ]; then echo ">> saturated at $r req/s (previous rate was the last clean one)"; break; fi
done
echo "== done =="
