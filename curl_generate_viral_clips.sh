#!/usr/bin/env bash
set -euo pipefail

# Hardcoded config (edit these two lines if needed)
API_BASE="http://localhost:8080"
VIDEO_URL="http://127.0.0.1:9999/input.mp4"

echo "POST ${API_BASE}/api/generate-viral-clips"
echo "videoUrl: ${VIDEO_URL}"

curl -sS -X POST "${API_BASE}/api/generate-viral-clips" \
  -H "Content-Type: application/json" \
  -d "{\"videoUrl\": \"${VIDEO_URL}\"}" | python3 -m json.tool
