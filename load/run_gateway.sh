#!/usr/bin/env bash
# Start the gateway with the load-test config (env vars override .env).
cd "$(dirname "$0")/.." || exit 1
set -a; source .env.load; set +a
exec python -m uvicorn app.main:app --port 8000 --log-level warning --timeout-graceful-shutdown 30 "$@"
