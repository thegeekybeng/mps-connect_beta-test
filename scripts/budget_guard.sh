#!/usr/bin/env bash
set -euo pipefail

# Budget/guard gate for CI/CD. Prints a decision and writes an output suitable
# for GitHub Actions to $GITHUB_OUTPUT if present.

# Inputs via env vars (usually provided as GitHub secrets):
# - DEPLOY_PAUSE: if 'true', blocks deployment unconditionally
# - BUDGET_GUARD_ENABLED: 'true' to enable budget checks
# - BUDGET_MAX_MONTHLY_USD: numeric threshold (e.g., 500)
# - BUDGET_USAGE_URL: optional HTTPS endpoint returning JSON {"month_usd": <number>}
# - BUDGET_USAGE_TOKEN: optional bearer token for BUDGET_USAGE_URL

echo "[budget_guard] starting"

DEPLOY_PAUSE=${DEPLOY_PAUSE:-false}
BUDGET_GUARD_ENABLED=${BUDGET_GUARD_ENABLED:-false}
BUDGET_MAX_MONTHLY_USD=${BUDGET_MAX_MONTHLY_USD:-}
BUDGET_USAGE_URL=${BUDGET_USAGE_URL:-}
BUDGET_USAGE_TOKEN=${BUDGET_USAGE_TOKEN:-}

decision="continue"
reason=""

if [[ "${DEPLOY_PAUSE}" == "true" ]]; then
  decision="block"
  reason="DEPLOY_PAUSE=true"
fi

if [[ "${decision}" == "continue" && "${BUDGET_GUARD_ENABLED}" == "true" ]]; then
  if [[ -n "${BUDGET_MAX_MONTHLY_USD}" ]]; then
    current=""
    if [[ -n "${BUDGET_USAGE_URL}" ]]; then
      echo "[budget_guard] querying usage from URL"
      auth=()
      [[ -n "${BUDGET_USAGE_TOKEN}" ]] && auth=(-H "Authorization: Bearer ${BUDGET_USAGE_TOKEN}")
      set +e
      resp=$(curl -fsS "${BUDGET_USAGE_URL}" "${auth[@]}" 2>/dev/null)
      code=$?
      set -e
      if [[ $code -eq 0 ]]; then
        current=$(python - <<'PY'
import json,sys,os
try:
  data=json.loads(sys.stdin.read())
  v=float(data.get('month_usd', 0.0))
  print(v)
except Exception:
  print(0.0)
PY
<<EOF
${resp}
EOF
)
      else
        echo "[budget_guard] WARN: failed to query usage URL; proceeding"
      fi
    fi

    if [[ -n "${current}" ]]; then
      # numeric compare using awk for portability
      over=$(awk -v c="$current" -v m="$BUDGET_MAX_MONTHLY_USD" 'BEGIN{print (c>=m)?"1":"0"}')
      if [[ "$over" == "1" ]]; then
        decision="block"
        reason="usage ${current} >= max ${BUDGET_MAX_MONTHLY_USD}"
      fi
    fi
  fi
fi

echo "[budget_guard] decision=${decision} ${reason}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "continue=$([[ "$decision" == "continue" ]] && echo true || echo false)" >> "$GITHUB_OUTPUT"
  echo "reason=${reason}" >> "$GITHUB_OUTPUT"
fi

if [[ "$decision" != "continue" ]]; then
  exit 0  # let the workflow decide via outputs; do not hard-fail here
fi

exit 0

