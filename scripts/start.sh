#!/usr/bin/env sh
set -eu

# Sanitize PORT to ensure it is a valid integer
PORT_CLEAN="$(printf '%s' "${PORT:-8000}" | grep -Eo '^[0-9]+$' || true)"
[ -z "$PORT_CLEAN" ] && PORT_CLEAN=8000

export PORT="$PORT_CLEAN"
echo "[start] Using PORT=$PORT" >&2

exec uvicorn api.app:app --host 0.0.0.0 --port "$PORT"

