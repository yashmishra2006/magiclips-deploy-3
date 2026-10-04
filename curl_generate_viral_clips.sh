set -euo pipefail

# Hardcoded config
API_BASE="http://localhost:8080"
# API_BASE="https://magi-clips-627061442405.asia-southeast1.run.app/"
VIDEO_URL="https://youtu.be/-0WsAAidqV8?si=XlpDwiikH4bKF9kX"
AUTH_TOKEN="${AUTH_TOKEN:?set AUTH_TOKEN to a Firebase ID token}"
echo "POST ${API_BASE}/api/generate-viral-clips"
echo "videoUrl: ${VIDEO_URL}"

curl -sS -X POST "${API_BASE}/api/generate-viral-clips" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -d "{\"videoUrl\": \"${VIDEO_URL}\"}" | python3 -m json.tool
