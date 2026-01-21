set -euo pipefail

# Hardcoded config
API_BASE="http://localhost:8080"
# API_BASE="https://magi-clips-627061442405.asia-southeast1.run.app/"
VIDEO_URL="https://youtu.be/-0WsAAidqV8?si=XlpDwiikH4bKF9kX"
AUTH_TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ4Mjg5MmZhMzJlY2QxM2E0ZTBhZWZlNjI4ZGQ5YWFlM2FiYThlMWUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2Fwcy04NTI1NCIsImF1ZCI6ImNhcHMtODUyNTQiLCJhdXRoX3RpbWUiOjE3NjgyOTA3ODgsInVzZXJfaWQiOiJGSmFHS3R3OU0wUTBIUUVRU3dZc0M4amFZRzcyIiwic3ViIjoiRkphR0t0dzlNMFEwSFFFUVN3WXNDOGphWUc3MiIsImlhdCI6MTc2ODI5MDc4OCwiZXhwIjoxNzY4Mjk0Mzg4LCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsidGVzdEBleGFtcGxlLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6InBhc3N3b3JkIn19.CbN-05weF8KheWH8O_dd88nVmD3Ri9iXbTWmTaUeqJ70WQIxsK8UcAgBeWM1qntqztjuMFwlFvki1P-0-Y6CewZm_QUcU3aGJBE7uvIKglHo3do19OQtSyQ7qIOycghZRxtemCYaF9j47Ri5q1zDcMctedzmmcF7u6JWfAHBP0nMb2beRiJ4MCrIHzef_BQnHE1aE77C25OHcE2ZpOdu6H1nbsZ3eFxoEtsa2wIDRgyC9qko1osbUT7vHeKWEuybXXFiPPkzQ0fGMgxwBmMaIB7bjsuTH8ohrDUuqMyQaC2SauP5Zp75JBsVDOs7RDvcAbxe9Yrryghh8p_Wk_P5SA"
echo "POST ${API_BASE}/api/generate-viral-clips"
echo "videoUrl: ${VIDEO_URL}"

curl -sS -X POST "${API_BASE}/api/generate-viral-clips" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -d "{\"videoUrl\": \"${VIDEO_URL}\"}" | python3 -m json.tool
