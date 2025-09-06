#!/usr/bin/env sh
set -eu

# Sanitize PORT to ensure it's an integer (trim trailing decimals)
PORT_CLEAN=${PORT%%.*}
if [ -z "${PORT_CLEAN}" ]; then
  PORT_CLEAN=8000
fi

export PORT="${PORT_CLEAN}"
echo "[start] Using PORT=${PORT}" >&2

exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT}"

