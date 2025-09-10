Cloud Run Deploy Workflow — GitHub Secrets
=========================================

Required GitHub Actions secrets for `.github/workflows/deploy-cloud-run.yml`.

Minimum required
- GCP_PROJECT_ID: mps-connect-pilot

Optional with defaults
- GCP_REGION: asia-southeast1
- ARTIFACT_REPO: mps-connect

Authentication (choose one)
- Workload Identity Federation (recommended)
  - GCP_WORKLOAD_IDENTITY_PROVIDER: resource name of your WIF provider
  - GCP_SERVICE_ACCOUNT: service account email with deploy permissions
- OR Service Account Key
  - GCP_SA_KEY: JSON string of the service account key

Set secrets using gh CLI
------------------------

1) Authenticate gh CLI:
   gh auth login

2) Set secrets (WIF example):

   GCP_PROJECT_ID=mps-connect-pilot \
   GCP_REGION=asia-southeast1 \
   ARTIFACT_REPO=mps-connect \
   GCP_WORKLOAD_IDENTITY_PROVIDER="projects/123456789/locations/global/workloadIdentityPools/github/providers/github-oidc" \
   GCP_SERVICE_ACCOUNT="deploy@mps-connect-pilot.iam.gserviceaccount.com" \
   scripts/set_github_secrets.sh

3) Or with a SA key file:

   GCP_PROJECT_ID=mps-connect-pilot \
   GCP_SA_KEY_FILE=./sa-key.json \
   scripts/set_github_secrets.sh

GCP IAM roles to deploy
-----------------------

On the deploy service account (the account used by GitHub Actions auth):
- roles/run.admin
- roles/iam.serviceAccountUser
- roles/artifactregistry.admin (or writer)
- roles/cloudbuild.builds.editor

Runtime Service Account access
------------------------------

The workflow grants the runtime compute default SA `roles/artifactregistry.reader` automatically:

  ${PROJECT_NUMBER}-compute@developer.gserviceaccount.com

If your org policy restricts this, grant it manually or deploy with a dedicated runtime SA.

Notes
-----
- If using WIF, set up a Workload Identity Pool and Provider for GitHub OIDC and trust your repo.
- If using a SA key, prefer rotating keys and migrate to WIF when possible.

