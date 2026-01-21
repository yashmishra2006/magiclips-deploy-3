#!/bin/bash

# Configuration
PROJECT_ID="new-backend-475414"
SERVICE_NAME="magi-clips"
REGION="asia-southeast1" 
REPO_NAME="magiclips-repo" 
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"


# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Whisper transcription service images" \
  --project=${PROJECT_ID} \
  2>/dev/null || echo "Repository already exists"

# Build image in the cloud using Cloud Build
echo "Building Docker image in Cloud Build..."
gcloud builds submit \
  --tag ${IMAGE_NAME} \
  --project=${PROJECT_ID} \
  --timeout=30m

# Deploy to Cloud Run with L4 GPU
echo "Deploying to Cloud Run .."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --memory 16Gi \
  --cpu 8 \
  --timeout 900 \
  --concurrency 2 \
  --max-instances 3 \
  --min-instances 0 \
  --allow-unauthenticated 

echo "Deployment complete!"
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --project=${PROJECT_ID} \
  --format='value(status.url)'