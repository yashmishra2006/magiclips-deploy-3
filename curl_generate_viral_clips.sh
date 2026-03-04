set -euo pipefail

# Hardcoded config
API_BASE="http://localhost:8080"
# API_BASE="https://magi-clips-627061442405.asia-southeast1.run.app/"
VIDEO_URL="https://youtu.be/-0WsAAidqV8?si=XlpDwiikH4bKF9kX"
AUTH_TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6IjRiMTFjYjdhYjVmY2JlNDFlOTQ4MDk0ZTlkZjRjNWI1ZWNhMDAwOWUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2Fwcy04NTI1NCIsImF1ZCI6ImNhcHMtODUyNTQiLCJhdXRoX3RpbWUiOjE3NzEwNDQ1ODMsInVzZXJfaWQiOiJGSmFHS3R3OU0wUTBIUUVRU3dZc0M4amFZRzcyIiwic3ViIjoiRkphR0t0dzlNMFEwSFFFUVN3WXNDOGphWUc3MiIsImlhdCI6MTc3MTA0NDU4MywiZXhwIjoxNzcxMDQ4MTgzLCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsidGVzdEBleGFtcGxlLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6InBhc3N3b3JkIn19.J32XoTGNOEPq4fztiz1aPC77xk1ThET9-PwvFRS31JkcaRSWmah4nQX7pu9jMv_j8BtIEkaJ5JfAyzQSumy2WBOayS3s8khVTUYQul4q1yzNwJq3znmlhBju7CwaPSsddgfrtzs5o2fXHLmgBoyJPCL9eJVaYtQVki3IMq4qcGgUfC4_gmZ-8cP5RUd4ZpUUq9ncYN0pPeSwayW8Dit0Kxdud7JpQLhouCISVYRvn71w4ZQx8qvzYG8T691O9ZrSU-kup6r_Vbf4PAomxgpkxnWWfBsrMWH3JLE4Rqq7umEXxO8iC8GPvn5rw1gDgU3zT6o9Nx_Gt17sl7o8abspNQ"
echo "POST ${API_BASE}/api/generate-viral-clips"
echo "videoUrl: ${VIDEO_URL}"

curl -sS -X POST "${API_BASE}/api/generate-viral-clips" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -d "{\"videoUrl\": \"${VIDEO_URL}\"}" | python3 -m json.tool
