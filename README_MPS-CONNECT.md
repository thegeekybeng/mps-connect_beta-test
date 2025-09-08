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

### Docker

```bash
docker compose -f docker-compose.dev.yml up --build
```

For a production-like stack:

```bash
docker compose -f docker-compose.prod.yml up --build
```

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn numpy scikit-learn pydantic
uvicorn api.app:app --reload
```

## Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md)
- Google Cloud deployment: see `DEPLOY_GCP.md` at the project root
- [Security Guide](SECURITY_GUIDE.md)
- [Governance Guide](GOVERNANCE_GUIDE.md)

## Usage

- **API**: `uvicorn api.app:app --reload` then visit `http://localhost:8000`.
- **Frontend**: open [`index.html`](index.html) in a browser or build using [`web/Dockerfile`](web/Dockerfile).
