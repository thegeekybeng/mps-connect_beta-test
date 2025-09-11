#!/bin/bash
# Deploy MPS Connect Frontend to Cloud Run

set -euo pipefail

# Configuration
PROJECT_ID="mpsconnect-cw-pilot"
REGION="asia-southeast1"
SERVICE_NAME="mps-connect-web"
REPO_NAME="mps-connect"
IMAGE_TAG="web-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Deploying MPS Connect Frontend to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Image Tag: $IMAGE_TAG"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run:"
    echo "   gcloud auth login"
    echo "   gcloud config set project $PROJECT_ID"
    exit 1
fi

# Set project
gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

# Enable required services
echo "📦 Enabling required GCP services..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iam.googleapis.com compute.googleapis.com

# Create Artifact Registry repo if it doesn't exist
echo "🏗️ Setting up Artifact Registry..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="MPS Connect container images"
else
    echo "✅ Artifact Registry repository already exists"
fi

# Configure Docker to use Artifact Registry
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and push the frontend image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "🏗️ Building frontend image..."
docker build -f Dockerfile.web -t "$IMAGE_URI" .

echo "📤 Pushing image to Artifact Registry..."
docker push "$IMAGE_URI"

# Grant necessary permissions
echo "🔑 Setting up service account permissions..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/artifactregistry.reader" \
    --quiet || true

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_URI" \
    --region "$REGION" \
    --allow-unauthenticated \
    --min-instances "0" \
    --cpu 1 \
    --memory 512Mi \
    --timeout 60 \
    --max-instances 5 \
    --execution-environment gen2 \
    --port 80 \
    --concurrency 1000

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')
echo "✅ Frontend deployed successfully!"
echo "🌐 Frontend URL: $SERVICE_URL"

# Update API CORS origins to include the new frontend URL
echo "🔄 Updating API CORS origins..."
API_SERVICE="mps-connect-api"

# Get current CORS origins
CURRENT_CORS=$(gcloud run services describe "$API_SERVICE" --region="$REGION" --format='value(spec.template.spec.containers[0].env[?name=="CORS_ORIGINS"].value)' 2>/dev/null || echo "")

if [[ -z "$CURRENT_CORS" ]]; then
    NEW_CORS="$SERVICE_URL"
elif [[ "$CURRENT_CORS" != *"$SERVICE_URL"* ]]; then
    NEW_CORS="${CURRENT_CORS},$SERVICE_URL"
else
    NEW_CORS="$CURRENT_CORS"
fi

echo "📝 New CORS origins: $NEW_CORS"
gcloud run services update "$API_SERVICE" \
    --region="$REGION" \
    --set-env-vars="CORS_ORIGINS=$NEW_CORS" \
    --quiet

echo "🎉 Deployment complete!"
echo "📱 Frontend: $SERVICE_URL"
echo "🔗 API: https://mps-connect-api-987575541268.asia-southeast1.run.app"

# Test the frontend
echo "🧪 Testing frontend..."
if curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL" | grep -q "200"; then
    echo "✅ Frontend is responding correctly"
else
    echo "⚠️ Frontend may not be responding correctly"
fi
