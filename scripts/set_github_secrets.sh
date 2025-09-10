#!/usr/bin/env bash
set -euo pipefail

# Helper to set GitHub repo secrets for Cloud Run deploy workflow
# Requires: gh CLI authenticated (gh auth login)

usage() {
  cat <<'USAGE'
Usage: scripts/set_github_secrets.sh [--repo owner/name]

Reads values from environment variables (recommended) and sets secrets:

Required:
  GCP_PROJECT_ID           e.g. mps-connect-pilot

Optional (with defaults):
  GCP_REGION               default: asia-southeast1
  ARTIFACT_REPO            default: mps-connect

Authentication (choose one):
  Workload Identity Federation:
    GCP_WORKLOAD_IDENTITY_PROVIDER   resource name of WIF provider
    GCP_SERVICE_ACCOUNT              deploy service account email
  OR Service Account Key JSON:
    GCP_SA_KEY                       JSON string
    or GCP_SA_KEY_FILE               path to JSON file

Example:
  GCP_PROJECT_ID=mps-connect-pilot \
  GCP_REGION=asia-southeast1 \
  ARTIFACT_REPO=mps-connect \
  GCP_WORKLOAD_IDENTITY_PROVIDER="projects/123456789/locations/global/workloadIdentityPools/github/providers/github-oidc" \
  GCP_SERVICE_ACCOUNT=deploy@mps-connect-pilot.iam.gserviceaccount.com \
  scripts/set_github_secrets.sh

Or with SA key:
  GCP_PROJECT_ID=mps-connect-pilot \
  GCP_SA_KEY_FILE=./sa-key.json \
  scripts/set_github_secrets.sh
USAGE
}

REPO_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    --repo)
      [[ $# -ge 2 ]] || { echo "--repo requires value" >&2; exit 1; }
      REPO_ARG=("--repo" "$2"); shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

command -v gh >/dev/null 2>&1 || { echo "gh CLI is required" >&2; exit 1; }
gh auth status ${REPO_ARG[@]} >/dev/null 2>&1 || {
  echo "gh not authenticated (run: gh auth login)" >&2; exit 1;
}

set_secret() {
  local name="$1"; local value="$2"
  if [[ -n "${value}" ]]; then
    printf "Setting secret %s\n" "$name"
    printf "%s" "$value" | gh secret set "$name" ${REPO_ARG[@]} -b- >/dev/null
  else
    printf "Skipping %s (empty)\n" "$name"
  fi
}

: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
GCP_REGION="${GCP_REGION:-asia-southeast1}"
ARTIFACT_REPO="${ARTIFACT_REPO:-mps-connect}"

# Load SA key from file if provided
if [[ -z "${GCP_SA_KEY:-}" && -n "${GCP_SA_KEY_FILE:-}" ]]; then
  GCP_SA_KEY="$(cat "${GCP_SA_KEY_FILE}")"
fi

set_secret GCP_PROJECT_ID "$GCP_PROJECT_ID"
set_secret GCP_REGION "$GCP_REGION"
set_secret ARTIFACT_REPO "$ARTIFACT_REPO"

# Auth secrets (set whichever are provided)
set_secret GCP_WORKLOAD_IDENTITY_PROVIDER "${GCP_WORKLOAD_IDENTITY_PROVIDER:-}"
set_secret GCP_SERVICE_ACCOUNT "${GCP_SERVICE_ACCOUNT:-}"
set_secret GCP_SA_KEY "${GCP_SA_KEY:-}"

echo "Done. Verify secrets under GitHub Settings -> Secrets and variables -> Actions."

