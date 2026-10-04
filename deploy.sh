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

# Deploy to Cloud Run.
# - IAM-only (contract C1 / SEC-4): only the Cloud Tasks invoker SA may call it.
#   Grant it once:
#     gcloud run services add-iam-policy-binding ${SERVICE_NAME} --region ${REGION} \
#       --member="serviceAccount:${TASKS_INVOKER_SA}" --role=roles/run.invoker --project=${PROJECT_ID}
# - SC-8: concurrency 1 and a 30 min request timeout. The backend must create the
#   Cloud Task with dispatch_deadline=1800s (queue: viral-clips-queue, max attempts ~3).
#   A retry that arrives while the first attempt still holds its lease gets 409 and
#   is retried later; completed/published runs are acked as duplicates.
# - SEC-1: no key files. The runtime SA (RUNTIME_SA) needs: roles/datastore.user on
#   the Firebase project caps-85254, storage object read/write on subtitlegen-input,
#   roles/pubsub.publisher on clip-processing-jobs, roles/aiplatform.user, and
#   roles/run.invoker on the Whisper service. INTERNAL_API_TOKEN comes from Secret Manager.
#   Set RUNTIME_SA=<sa email> and INTERNAL_TOKEN_SECRET=<secret name> before running;
#   both flags are skipped when unset so the deploy still works before they exist.
EXTRA_FLAGS=()
[ -n "${RUNTIME_SA:-}" ] && EXTRA_FLAGS+=(--service-account "${RUNTIME_SA}")
[ -n "${INTERNAL_TOKEN_SECRET:-}" ] && EXTRA_FLAGS+=(--update-secrets "INTERNAL_API_TOKEN=${INTERNAL_TOKEN_SECRET}:latest")
echo "Deploying to Cloud Run .."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --timeout 1800 \
  --concurrency 1 \
  --max-instances 3 \
  --min-instances 0 \
  --no-traffic --tag "${TAG_NAME:-v2}" \
  "${EXTRA_FLAGS[@]}" \
  --update-env-vars "GCP_PROJECT_ID=${PROJECT_ID},FIREBASE_PROJECT_ID=caps-85254,GCP_BUCKET_NAME=subtitlegen-input,ALLOWED_INPUT_BUCKETS=subtitlegen-input,CLIP_TOPIC=clip-processing-jobs,YOUTUBE_DOWNLOADER_BUCKET=subtitlegen-input,YOUTUBE_DOWNLOADER_PREFIX=youtube_downloads/,WHISPER_SERVICE_URL=https://v2---magi-clips-whisper-transcription-627061442405.asia-southeast1.run.app"
# Also set (not hard-coded here): WHISPER_SERVICE_URL=https://<whisper service>.run.app,
# YOUTUBE_DOWNLOADER_URL=https://<downloader host>/download (HTTPS; see SEC-11 note).

echo "Deployment complete!"
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --project=${PROJECT_ID} \
  --format='value(status.url)'

# ---------------------------------------------------------------------------
# New revisions go out WITHOUT production traffic, reachable only through the
# tag URL (https://${TAG_NAME:-v2}---<service>-627061442405.asia-southeast1.run.app).
# Auth is service-wide, so this script never flips it. After callers use the
# tag URL with ID tokens, make the service IAM-only:
#   gcloud run services remove-iam-policy-binding ${SERVICE_NAME} --region ${REGION} \
#     --member=allUsers --role=roles/run.invoker
# ---------------------------------------------------------------------------
