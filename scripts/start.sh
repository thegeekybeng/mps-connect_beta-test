#!/usr/bin/env sh
set -eu

# Sanitize PORT to ensure it is a valid integer (default 8080 for Cloud Run)
PORT_CLEAN="$(printf '%s' "${PORT:-8080}" | grep -Eo '^[0-9]+$' || true)"
[ -z "$PORT_CLEAN" ] && PORT_CLEAN=8000

export PORT="$PORT_CLEAN"
echo "[start] Using PORT=$PORT" >&2

exec uvicorn api.app:app --host 0.0.0.0 --port "$PORT"
