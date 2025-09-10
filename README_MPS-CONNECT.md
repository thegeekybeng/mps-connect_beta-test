# MPS Connect (Beta)

MPS Connect is an AI-powered assistant for Member of Parliament (MP) offices. It helps staff gather constituent information through a guided chat and produces professional letters in an MP office tone.

## Project Goals

- Streamline case intake with a warm, context-aware chat assistant.
- Generate agency-ready letters from extracted facts.
- Provide secure, auditable infrastructure suitable for government use.

## Core Components

- **`api/`** – FastAPI backend (entry point: [`api/app.py`](api/app.py)).
- **`web/`** – Static frontend served via [`index.html`](index.html) and containerized with [`web/Dockerfile`](web/Dockerfile).
- **`database/`** – SQLAlchemy models, schema, and Alembic migrations.
- **`security/`** – Authentication, encryption, and middleware.
- **`governance/`** – Immutable audit logs and compliance tooling.
- **`scripts/`** – Deployment and monitoring helpers.

## Development Status

Core chat and letter-generation features are complete, with ongoing work focusing on production deployment (database, security, governance, Docker, hosting) and advanced chat improvements.

## Setup

### Docker (local-only)

```bash
docker compose -f docker-compose.dev.yml up --build
```

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn numpy scikit-learn pydantic
uvicorn api.app:app --reload
```

## Documentation

- Cloud Run deployment (Artifact Registry) and CI/CD below.
- [Security Guide](SECURITY_GUIDE.md)
- [Governance Guide](GOVERNANCE_GUIDE.md)

## Cloud Run Deployment (Artifact Registry)

Requirements
- `gcloud` authenticated and project selected: `gcloud config set project <PROJECT_ID>`
- Enable APIs once: `run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com iam.googleapis.com compute.googleapis.com`
- Artifact Registry repo exists: name `mps-connect`, region `asia-southeast1`.

Build and push
```
gcloud builds submit --tag \
  asia-southeast1-docker.pkg.dev/<PROJECT_ID>/mps-connect/mps-connect-api:cloud-dev .
```

Grant runtime image pull (once)
```
PN=$(gcloud projects describe <PROJECT_ID> --format='value(projectNumber)')
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:${PN}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

Deploy
```
gcloud run deploy mps-connect-api \
  --image asia-southeast1-docker.pkg.dev/<PROJECT_ID>/mps-connect/mps-connect-api:cloud-dev \
  --region asia-southeast1 --allow-unauthenticated \
  --cpu 2 --memory 2Gi --timeout 300 \
  --env-vars-file env.yaml
```

Notes
- `env.yaml` uses absolute container paths (`/app/api/...`).
- A declarative spec is available in `cloud-run-config.yaml` (update `image:` path then `gcloud run services replace`).

## GitHub Actions (cloud_dev)

Workflow
- `.github/workflows/deploy-cloud-run.yml` builds with Cloud Build and deploys to Cloud Run on push to `cloud_dev`.

Secrets required (repo → Settings → Secrets and variables → Actions)
- `GCP_PROJECT_ID` (e.g., `mpsconnect-cw-pilot`)
- `GCP_REGION` (default `asia-southeast1`)
- `ARTIFACT_REPO` (default `mps-connect`)
- Auth: either Workload Identity Federation (`GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT`) or `GCP_SA_KEY` (JSON).

Helper
- See `.github/WORKFLOW_SECRETS.md` and `scripts/set_github_secrets.sh`.

## Localhost Branch Strategy

Goal
- Keep application code identical to cloud; differ only in configuration (compose/env/CORS) so local behaves the same.

Recommended
- Regularly merge `cloud_dev` into `localhost`:
  - `git checkout localhost && git merge cloud_dev`
- Use the same container/Dockerfile and start command. Local differences live in:
  - `docker-compose.dev.yml` (ports, volumes, dev DB/Redis)
  - `.env` (local values for `CORS_ORIGINS`, `DATABASE_URL`, secrets)
- For the web UI, ensure the API base URL points to local during dev (e.g., `http://localhost:8000`).

Run locally (compose)
```
docker compose -f docker-compose.dev.yml up --build
```

Tips
- Avoid diverging code paths; gate behavior with env vars rather than branches when possible.
- If you need a quick switch between local and cloud API for `index.html`, consider reading `window.API_BASE` from a small `env.js` included only in dev.

## Usage

- **API**: `uvicorn api.app:app --reload` then visit `http://localhost:8000`.
- **Frontend**: open [`index.html`](index.html) directly in your browser, or serve the folder via `python -m http.server 8080` and browse `http://localhost:8080`.
