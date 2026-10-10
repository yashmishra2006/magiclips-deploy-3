from pathlib import Path
import os
import sys

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import uuid
import time
import re
import math
import asyncio
import json
import subprocess
import tempfile
from urllib.parse import quote, urlparse
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.responses import JSONResponse

import httpx
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore

from gemini_analysis import analyze_audio_with_gemini
from internal_auth import verify_internal_token, internal_auth_headers
from pubsub_publisher import create_clip_publisher
import obs


# ──────────────────────────────────────────────────────────────
# Basic setup & logging
# ──────────────────────────────────────────────────────────────

load_dotenv()

# Structured JSON logs (Cloud Logging); level from LOG_LEVEL (default INFO).
obs.setup_logging("magi-clips")
logger = logging.getLogger("capsai.main")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "subtitlegen-input")
# Buckets the worker may read a source video from (comma list).
ALLOWED_INPUT_BUCKETS = {
    b.strip() for b in os.getenv("ALLOWED_INPUT_BUCKETS", GCP_BUCKET_NAME).split(",") if b.strip()
}

# SC-8: hard caps checked BEFORE any download / audio extraction.
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_BYTES", str(6 * 1024 ** 3)))          # 6 GiB
MAX_SOURCE_DURATION_SECONDS = float(os.getenv("MAX_SOURCE_DURATION_SECONDS", "10800"))  # 3 h
# SC-8: a claimed run holds a lease; a Cloud Tasks retry that arrives while the
# lease is live gets 409 (retry later) instead of starting duplicate work.
WORKER_LEASE_SECONDS = float(os.getenv("WORKER_LEASE_SECONDS", "1900"))
FFPROBE_TIMEOUT_SECONDS = int(os.getenv("FFPROBE_TIMEOUT_SECONDS", "90"))
AUDIO_EXTRACT_TIMEOUT_SECONDS = int(os.getenv("AUDIO_EXTRACT_TIMEOUT_SECONDS", "900"))

# SEC-11: YouTube downloader endpoint. HTTPS only unless explicitly overridden.
YOUTUBE_DOWNLOADER_ALLOW_INSECURE_HTTP = os.getenv("YOUTUBE_DOWNLOADER_ALLOW_INSECURE_HTTP", "") == "1"
YOUTUBE_DOWNLOADER_BUCKET = os.getenv("YOUTUBE_DOWNLOADER_BUCKET", GCP_BUCKET_NAME)
YOUTUBE_DOWNLOADER_PREFIX = os.getenv("YOUTUBE_DOWNLOADER_PREFIX", "youtube_downloads/")

PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
UID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Firebase initialization
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "caps-85254")
FIREBASE_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    f"{FIREBASE_PROJECT_ID}.appspot.com"
)


def _init_firebase():
    """SEC-1: use the Cloud Run runtime service account (ADC) in production.

    A key file is used only when FIREBASE_CERT_PATH points at an existing file
    (local development). Key files are git/docker/gcloud-ignored, so the image
    never contains one and falls through to ADC.
    """
    if firebase_admin._apps:
        return
    cert_path = os.getenv("FIREBASE_CERT_PATH", "caps-85254-firebase-adminsdk.json")
    if cert_path and Path(cert_path).is_file():
        cred = credentials.Certificate(cert_path)
        logger.info("Firebase: using key file %s", cert_path)
    else:
        cred = credentials.ApplicationDefault()
        logger.info("Firebase: using Application Default Credentials")
    firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": FIREBASE_STORAGE_BUCKET,
    })


_init_firebase()
db = firestore.client()

_storage_client = None


def get_storage_client():
    """Storage client from the shipped service-account key when present, else ADC."""
    global _storage_client
    if _storage_client is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID") or None
        key_path = os.getenv("GCS_CERT_PATH", "new-backend-475414-228aab3a465a.json")
        if Path(key_path).is_file():
            _storage_client = storage.Client.from_service_account_json(key_path, project=project)
        else:
            _storage_client = storage.Client(project=project)
    return _storage_client


def upload_to_gcs_sync(local_path: Path, dest_name: str, content_type: str | None = None) -> str:
    bucket = get_storage_client().bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(dest_name)
    started = time.perf_counter()
    status = "error"
    try:
        blob.upload_from_filename(str(local_path), content_type=content_type)
        status = "ok"
    finally:
        try:
            nbytes = Path(local_path).stat().st_size
        except OSError:
            nbytes = None
        obs.storage_event("gcs", "upload", GCP_BUCKET_NAME, dest_name, nbytes,
                          obs.ms_since(started), status=status, via="sdk",
                          level=logging.INFO if status == "ok" else logging.WARNING)
    return f"gs://{bucket.name}/{dest_name}"


# ──────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────

def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("Not a gs:// URI")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob or ".." in blob.split("/"):
        raise ValueError("Malformed gs:// URI")
    return bucket, blob


def is_allowed_source_uri(uri: str) -> bool:
    try:
        bucket, _ = parse_gs_uri(uri)
    except ValueError:
        return False
    return bucket in ALLOWED_INPUT_BUCKETS


def gcs_access_token() -> str:
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def gcs_http_input(bucket: str, blob: str) -> tuple[list[str], str]:
    """(header args, url) to stream an object over HTTPS with range requests (no full download)."""
    url = f"https://storage.googleapis.com/{bucket}/{quote(blob, safe='/')}"
    args = [
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-headers", f"Authorization: Bearer {gcs_access_token()}\r\n",
    ]
    return args, url


class PermanentJobError(Exception):
    """A failure that retrying cannot fix: mark the project failed, refund, ack the task."""


class ClaimRejected(Exception):
    """The task does not match the project (missing, wrong owner/type). Touch nothing."""


def probe_duration_seconds_gcs(bucket: str, blob: str) -> float:
    http_args, url = gcs_http_input(bucket, blob)
    cmd = ["ffprobe", "-v", "error", *http_args,
           "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", url]
    # ffprobe range-reads the container header over HTTPS (bytes read unknown).
    started = time.perf_counter()
    status = "error"
    try:
        probe = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS)
        status = "ok" if probe.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        status = "timeout"
        raise PermanentJobError("Timed out reading the source video's duration")
    finally:
        obs.storage_event("gcs", "download", bucket, blob, None, obs.ms_since(started),
                          range_read=True, status=status, via="ffprobe_http",
                          level=logging.INFO if status == "ok" else logging.WARNING)
    if probe.returncode != 0:
        raise PermanentJobError("Could not read the source video (ffprobe failed)")
    try:
        value = float((probe.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        value = 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def extract_audio_from_gcs_sync(bucket: str, blob: str, audio_path: str,
                                source_bytes: int | None = None, media_seconds: float | None = None) -> str:
    """Stream the source from GCS and extract mono MP3 audio (no full video download).

    source_bytes / media_seconds only feed the cost log lines.
    """
    http_args, url = gcs_http_input(bucket, blob)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-v", "error",
        "-fflags", "+genpts+discardcorrupt",
        "-err_detect", "ignore_err",
        *http_args, "-i", url,
        "-map", "0:a:0",
        "-vn", "-dn", "-sn",
        "-af", "aresample=async=1:first_pts=0",
        "-ar", "48000", "-ac", "1",
        "-c:a", "libmp3lame", "-q:a", "2",
        audio_path,
    ]
    started = time.perf_counter()
    status = "error"
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, timeout=AUDIO_EXTRACT_TIMEOUT_SECONDS)
        status = "ok" if result.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        status = "timeout"
        raise PermanentJobError("Audio extraction timed out")
    finally:
        duration_ms = obs.ms_since(started)
        try:
            output_bytes = Path(audio_path).stat().st_size
        except OSError:
            output_bytes = None
        # ffmpeg streams (≈ the whole object, audio is interleaved) from GCS over HTTPS.
        obs.storage_event("gcs", "download", bucket, blob, source_bytes, duration_ms,
                          status=status, via="ffmpeg_http", bytes_estimated=True,
                          level=logging.INFO if status == "ok" else logging.WARNING)
        obs.log_event("ffmpeg_done", logging.INFO if status == "ok" else logging.WARNING,
                      step="extract_audio", input_bytes=source_bytes, output_bytes=output_bytes,
                      media_seconds=media_seconds, duration_ms=duration_ms, status=status)
    if result.returncode != 0:
        stderr = (result.stderr or "")[-600:]
        if "matches no streams" in stderr or "Stream map" in stderr:
            raise PermanentJobError("The video has no audio track")
        logger.error("Audio extraction ffmpeg failed: %s", stderr)
        raise PermanentJobError("Could not extract audio from the video")
    if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
        raise PermanentJobError("Audio extraction produced empty output")
    return audio_path


# ──────────────────────────────────────────────────────────────
# YouTube helpers (SEC-11)
# ──────────────────────────────────────────────────────────────

YOUTUBE_REGEX = re.compile(
    r'(?:https?:\/\/)?(?:www\.|m\.)?(?:'
    r'youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.*?v=|shorts\/)'
    r')([\w\-]{11})(?:[^\w\-]|$)'
)


def is_youtube_url(url: str) -> bool:
    return isinstance(url, str) and bool(YOUTUBE_REGEX.search(url))


def youtube_downloader_url() -> str:
    url = os.getenv("YOUTUBE_DOWNLOADER_URL", "").strip()
    if not url:
        raise PermanentJobError("YouTube import is not configured (YOUTUBE_DOWNLOADER_URL unset)")
    scheme = urlparse(url).scheme
    if scheme != "https" and not (scheme == "http" and YOUTUBE_DOWNLOADER_ALLOW_INSECURE_HTTP):
        raise PermanentJobError("YouTube downloader must be reached over HTTPS")
    return url


_YT_URI_RE_TEMPLATE = r"^gs://{bucket}/{prefix}[A-Za-z0-9_-]+/{name}$"


def validate_downloader_uris(video_gcs: str, audio_gcs: str) -> None:
    """Only accept objects the downloader is supposed to write (bucket + prefix + file name)."""
    bucket = re.escape(YOUTUBE_DOWNLOADER_BUCKET)
    prefix = re.escape(YOUTUBE_DOWNLOADER_PREFIX)
    video_re = _YT_URI_RE_TEMPLATE.format(bucket=bucket, prefix=prefix, name=r"video\.mp4")
    audio_re = _YT_URI_RE_TEMPLATE.format(bucket=bucket, prefix=prefix, name=r"audio\.(?:mp3|m4a)")
    if not isinstance(video_gcs, str) or not re.fullmatch(video_re, video_gcs):
        raise PermanentJobError("Downloader returned an unexpected video location")
    if not isinstance(audio_gcs, str) or not re.fullmatch(audio_re, audio_gcs):
        raise PermanentJobError("Downloader returned an unexpected audio location")


async def download_youtube_video_external(youtube_url: str) -> tuple[str, str, float]:
    """Download via the external downloader, which uploads to GCS itself.

    Returns (video_gcs_uri, audio_gcs_uri, duration_seconds).
    """
    downloader_url = youtube_downloader_url()
    headers = await asyncio.to_thread(internal_auth_headers, downloader_url)
    match = YOUTUBE_REGEX.search(youtube_url)
    logger.info("Calling YouTube downloader for video %s", match.group(1) if match else "?")
    headers = {**obs.outbound_headers(), **headers}
    started = time.perf_counter()
    call_status: int | str = "error"
    try:
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=False) as client:
            response = await client.post(downloader_url, json={"url": youtube_url}, headers=headers)
        call_status = response.status_code
    except httpx.HTTPError as e:
        raise PermanentJobError(f"YouTube downloader unreachable: {e}")
    finally:
        obs.log_event("outbound_call_done", target="yt-downloader", method="POST",
                      path=urlparse(downloader_url).path, status=call_status,
                      duration_ms=obs.ms_since(started), attempt=1)
    if response.status_code >= 400:
        raise PermanentJobError(f"YouTube download failed (HTTP {response.status_code})")
    try:
        data = response.json()
    except ValueError:
        raise PermanentJobError("YouTube downloader returned invalid JSON")
    if data.get("status") != "success":
        raise PermanentJobError("YouTube downloader reported an error")

    video_gcs = data.get("video_gcs")
    audio_gcs = data.get("audio_gcs")
    validate_downloader_uris(video_gcs, audio_gcs)
    try:
        duration_seconds = float(data.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        duration_seconds = 0.0
    if not math.isfinite(duration_seconds):
        duration_seconds = 0.0
    return video_gcs, audio_gcs, duration_seconds


# ──────────────────────────────────────────────────────────────
# Pure helpers (unit-tested)
# ──────────────────────────────────────────────────────────────

def srt_time_to_seconds(time_str) -> float:
    """Converts HH:MM:SS / MM:SS / seconds to float."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    try:
        time_str = str(time_str).strip()
        time_str = time_str.replace(', ', ':').replace('. ', ':').replace(',', '.').replace(' ', '')
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception as e:
        logger.error(f"Time parsing error for '{time_str}': {e}")
        return float("nan")


def normalize_clip_windows(viral_clips_meta: list, source_duration: float) -> list[dict]:
    """Parse Gemini's clip windows, clamp them to the source and drop invalid ones.

    A clip starting at 0 is valid (the old `if not start` check dropped it).
    """
    out = []
    seen = set()
    for meta in viral_clips_meta or []:
        if not isinstance(meta, dict):
            continue
        start, end = meta.get("start"), meta.get("end")
        if start is None or end is None:
            continue
        start_sec = srt_time_to_seconds(start)
        end_sec = srt_time_to_seconds(end)
        if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
            continue
        start_sec = max(0.0, start_sec)
        if source_duration and source_duration > 0:
            end_sec = min(end_sec, source_duration)
        if end_sec - start_sec < 1.0:
            continue
        key = (round(start_sec, 2), round(end_sec, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "reason": str(meta.get("reason", "") or "")[:2000],
            "duration": round(end_sec - start_sec, 3),
        })
    return out


def seconds_to_credit_minutes(duration_seconds: float) -> float:
    """Convert video duration in seconds to billable minutes."""
    if not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return 0.0
    return round(float(duration_seconds) / 60.0, 2)


def get_user_magic_balance(user_data: dict) -> float:
    if not isinstance(user_data, dict):
        return 0.0
    try:
        return round(float(user_data.get("magicmins", 0.0) or 0.0), 2)
    except (TypeError, ValueError):
        return 0.0


class InsufficientCredits(PermanentJobError):
    pass


def plan_reservation(project_data: dict, user_data: dict, minutes: float) -> dict:
    """Decide what the reservation transaction writes (SC-9 / C7).

    Returns {"reuse": True} when this project already holds an unsettled
    reservation (e.g. a retry after a crash), else the new balances.
    Raises InsufficientCredits when the balance cannot cover `minutes`.
    """
    existing = float(project_data.get("reservedMagicMins") or 0.0)
    if existing > 0 and project_data.get("billingSettled") is False:
        return {"reuse": True, "reserved": existing}
    balance = get_user_magic_balance(user_data)
    if balance + 1e-9 < minutes:
        raise InsufficientCredits(
            f"Insufficient magic credits. Required {minutes:.2f} minutes, available {balance:.2f} minutes."
        )
    return {"reuse": False, "reserved": round(minutes, 2), "new_balance": round(balance - minutes, 2)}


# "processing" without clipsPublishedAt means a live attempt is publishing right now.
WORKER_ACTIVE_STATUSES = {
    "claimed", "downloading", "validating_credits", "extracting_audio", "uploading_audio",
    "transcribing", "finding_clips", "queueing_clips", "processing",
}


def plan_claim(project_data: dict | None, uid: str, attempt_id: str, now: float) -> str:
    """Entry decision for a worker call (SC-8 idempotency).

    "claim"  -> start a run
    "noop"   -> already done / clips already published, ack the task
    "busy"   -> another attempt holds a live lease, ask Cloud Tasks to retry later
    Raises ClaimRejected for missing project / ownership / type problems.
    """
    if project_data is None:
        raise ClaimRejected("Project not found")
    if project_data.get("userId") != uid:
        raise ClaimRejected("Project does not belong to this user")
    if project_data.get("type") != "magiclip":
        raise ClaimRejected("Project is not a Smart Clips project")
    status = project_data.get("status")
    if status == "completed":
        return "noop"
    if status == "processing" and project_data.get("clipsPublishedAt") and project_data.get("expectedClipIds"):
        total = int(project_data.get("totalClips") or 0)
        done = int(project_data.get("completedClips") or 0) + int(project_data.get("failedClips") or 0)
        if done < total:
            return "noop"  # clips of this run are still being processed
    lease = project_data.get("workerLeaseUntil")
    if (
        status in WORKER_ACTIVE_STATUSES
        and isinstance(lease, (int, float))
        and lease > now
        and project_data.get("workerAttemptId") != attempt_id
    ):
        return "busy"
    return "claim"


# ──────────────────────────────────────────────────────────────
# Firestore transactions
# ──────────────────────────────────────────────────────────────

def _project_ref(project_id: str):
    return db.collection("projects").document(project_id)


def claim_project(project_id: str, uid: str, attempt_id: str) -> tuple[str, dict]:
    project_ref = _project_ref(project_id)
    txn = db.transaction()

    @firestore.transactional
    def _claim(transaction):
        snap = project_ref.get(transaction=transaction)
        data = snap.to_dict() if snap.exists else None
        decision = plan_claim(data, uid, attempt_id, time.time())
        if decision == "claim":
            transaction.update(project_ref, {
                "status": "claimed",
                "workerAttemptId": attempt_id,
                "workerLeaseUntil": time.time() + WORKER_LEASE_SECONDS,
                "workerStartedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                # reset per-run clip progress (billing fields are owned by reserve/settle)
                "clips": [],
                "clipCount": 0,
                "completedClips": 0,
                "failedClips": 0,
                "failedClipIds": [],
                "expectedClipIds": [],
                "clipsPublishedAt": firestore.DELETE_FIELD,
                "error": firestore.DELETE_FIELD,
            })
        return decision, data or {}

    return _claim(txn)


def reserve_magic_minutes(project_id: str, uid: str, minutes: float) -> float:
    project_ref = _project_ref(project_id)
    user_ref = db.collection("users").document(uid)
    txn = db.transaction()

    @firestore.transactional
    def _reserve(transaction):
        project_snap = project_ref.get(transaction=transaction)
        user_snap = user_ref.get(transaction=transaction)
        project_data = project_snap.to_dict() or {}
        user_data = user_snap.to_dict() if user_snap.exists else {}
        plan = plan_reservation(project_data, user_data, minutes)
        if plan["reuse"]:
            return plan["reserved"]
        transaction.update(user_ref, {
            "magicmins": plan["new_balance"],
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        transaction.update(project_ref, {
            "reservedMagicMins": plan["reserved"],
            "billingSettled": False,
            "billingStatus": "reserved",
            "reservedAt": firestore.SERVER_TIMESTAMP,
            "minutesToDeduct": plan["reserved"],
        })
        return plan["reserved"]

    return _reserve(txn)


def fail_project_and_refund(project_id: str, uid: str, message: str, attempt_id: str):
    """Mark the run failed and give back any unsettled reservation, once."""
    project_ref = _project_ref(project_id)
    user_ref = db.collection("users").document(uid)
    txn = db.transaction()

    @firestore.transactional
    def _fail(transaction):
        snap = project_ref.get(transaction=transaction)
        if not snap.exists:
            return
        data = snap.to_dict() or {}
        if data.get("workerAttemptId") != attempt_id:
            return  # not ours (never claimed, or a newer attempt owns it)
        if data.get("status") == "completed":
            return
        updates = {
            "status": "failed",
            "error": message[:1000],
            "errorMessage": message[:1000],
            "failedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "workerLeaseUntil": 0,
            # late Pub/Sub messages for this run become no-ops in the processor
            "expectedClipIds": [],
        }
        reserved = float(data.get("reservedMagicMins") or 0.0)
        if reserved > 0 and data.get("billingSettled") is False and data.get("userId") == uid:
            transaction.update(user_ref, {"magicmins": firestore.Increment(reserved)})
            updates.update({
                "billingSettled": True,
                "billingStatus": "refunded",
                "chargedMagicMins": 0.0,
                "refundedMagicMins": reserved,
                "minutesDeducted": 0.0,
            })
        transaction.update(project_ref, updates)

    try:
        _fail(txn)
    except Exception as e:
        logger.error("Failed to mark project %s failed / refund: %s", project_id, e, exc_info=True)


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CapsAI - Viral Clips Worker",
    description="Internal Cloud Tasks worker: analyses a source and queues per-clip jobs",
    version="0.2.0",
)
app.add_middleware(obs.RequestContextMiddleware)


class WorkerBusy(Exception):
    pass


async def run_viral_clips_job(project_id: str, uid: str, video_url: str, attempt_id: str) -> dict:
    """One Smart Clips run: validate, reserve credits, analyse, publish clip jobs.

    The per-clip processor appends clips and settles billing (contract C7).
    """
    log = logging.getLogger("capsai.viral_clips")

    decision, project = await asyncio.to_thread(claim_project, project_id, uid, attempt_id)
    if decision == "noop":
        log.info("Project %s already processed/published; acking duplicate task", project_id)
        obs.log_event("job_duplicate_skipped", job_type="viral_clips", reason="already_processed")
        return {"success": True, "projectId": project_id, "duplicate": True}
    if decision == "busy":
        obs.log_event("job_duplicate_skipped", logging.WARNING, job_type="viral_clips", reason="lease_held")
        raise WorkerBusy()

    def update_status(status: str, **extra):
        payload = {
            "status": status,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "workerLeaseUntil": time.time() + WORKER_LEASE_SECONDS,
        }
        payload.update(extra)
        try:
            _project_ref(project_id).update(payload)
        except Exception as e:
            log.error("Failed to update status to %s: %s", status, e)

    with tempfile.TemporaryDirectory(dir=str(UPLOAD_DIR)) as tmp:
        tmp_dir = Path(tmp)
        update_status("downloading")

        # ── 1. Resolve the source. Never fetch arbitrary URLs (SSRF). ──────
        audio_gcs_uri = None
        clip_audio_uri = None
        source_duration = 0.0
        raw_video_url = project.get("rawVideoUrl") or ""

        if is_youtube_url(video_url):
            video_gcs_uri, audio_gcs_uri, source_duration = await download_youtube_video_external(video_url)
            clip_audio_uri = audio_gcs_uri  # downloader video track has no audio
            v_bucket, v_blob = parse_gs_uri(video_gcs_uri)
            blob = await asyncio.to_thread(get_storage_client().bucket(v_bucket).get_blob, v_blob)
            if blob is None:
                raise PermanentJobError("Downloaded video not found in storage")
            if (blob.size or 0) > MAX_SOURCE_BYTES:
                raise PermanentJobError("Source video is too large")
            if source_duration <= 0:
                source_duration = await asyncio.to_thread(probe_duration_seconds_gcs, v_bucket, v_blob)
            source_uri = video_gcs_uri
        else:
            # Uploads: the source is the project's own upload, not the caller's URL.
            source_uri = raw_video_url
            if not is_allowed_source_uri(source_uri):
                raise PermanentJobError("Project has no valid uploaded source video")
            if video_url and video_url != source_uri:
                log.warning("Ignoring caller-supplied video_url for project %s; using rawVideoUrl", project_id)
            v_bucket, v_blob = parse_gs_uri(source_uri)
            # SC-8: size and duration checks BEFORE touching the media.
            blob = await asyncio.to_thread(get_storage_client().bucket(v_bucket).get_blob, v_blob)
            if blob is None:
                raise PermanentJobError("Uploaded video not found in storage")
            if (blob.size or 0) > MAX_SOURCE_BYTES:
                raise PermanentJobError(
                    f"Source video is too large ({blob.size / 1024 ** 3:.1f} GiB, max "
                    f"{MAX_SOURCE_BYTES / 1024 ** 3:.1f} GiB)"
                )
            probed = project.get("probedDurationSeconds")
            if project.get("probeStatus") == "ready" and isinstance(probed, (int, float)) and probed > 0:
                source_duration = float(probed)
            else:
                source_duration = await asyncio.to_thread(probe_duration_seconds_gcs, v_bucket, v_blob)

        if source_duration <= 0:
            raise PermanentJobError("Could not determine the source video's duration")
        if source_duration > MAX_SOURCE_DURATION_SECONDS:
            raise PermanentJobError(
                f"Source video is too long ({source_duration / 60:.0f} min, max "
                f"{MAX_SOURCE_DURATION_SECONDS / 60:.0f} min)"
            )

        # ── 2. Reserve credits atomically (SC-9 / C7). ─────────────────────
        minutes = seconds_to_credit_minutes(source_duration)
        update_status("validating_credits", originalDuration=source_duration, minutesToDeduct=minutes)
        reserved = await asyncio.to_thread(reserve_magic_minutes, project_id, uid, minutes)
        log.info("Reserved %.2f magicmins for project %s (%.1fs source)", reserved, project_id, source_duration)

        # ── 3. Audio for transcription. ────────────────────────────────────
        if audio_gcs_uri is None:
            existing_audio = project.get("audioUrl")
            if project.get("audioStatus") == "ready" and is_allowed_source_uri(existing_audio or ""):
                audio_gcs_uri = existing_audio
                log.info("Reusing extracted audio %s", audio_gcs_uri)
            else:
                update_status("extracting_audio")
                audio_path = tmp_dir / "audio.mp3"
                await asyncio.to_thread(
                    extract_audio_from_gcs_sync, v_bucket, v_blob, str(audio_path),
                    getattr(blob, "size", None), source_duration,
                )
                update_status("uploading_audio")
                audio_gcs_uri = await asyncio.to_thread(
                    upload_to_gcs_sync, audio_path, f"{v_blob}.mp3", "audio/mpeg"
                )

        # ── 4. Gemini analysis. ────────────────────────────────────────────
        update_status("transcribing")
        viral_clips_meta = await asyncio.to_thread(
            analyze_audio_with_gemini,
            "",
            gcs_audio_uri=audio_gcs_uri,
            total_duration_seconds=source_duration,
        )
        update_status("finding_clips")
        clips = normalize_clip_windows(viral_clips_meta, source_duration)
        if not clips:
            raise PermanentJobError("No viral clips were found in this video")

        # ── 5. Record the run, then publish one job per clip (C7). ─────────
        for clip in clips:
            clip["clip_id"] = uuid.uuid4().hex
        clip_ids = [c["clip_id"] for c in clips]
        update_status(
            "processing",
            totalClips=len(clips),
            completedClips=0,
            failedClips=0,
            failedClipIds=[],
            clips=[],
            clipCount=0,
            expectedClipIds=clip_ids,
            clipSourceUri=source_uri,
            clipAudioUri=clip_audio_uri,
            originalDuration=source_duration,
            reservedMagicMins=reserved,
        )

        publisher = create_clip_publisher()
        message_ids = await asyncio.to_thread(
            publisher.publish_clips_batch, clips, source_uri, clip_audio_uri, project_id,
        )
        if any(mid is None for mid in message_ids):
            raise PermanentJobError("Failed to queue clip jobs")

        _project_ref(project_id).update({
            "clipsPublishedAt": firestore.SERVER_TIMESTAMP,
            "queuedClips": len(message_ids),
            "workerLeaseUntil": 0,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        log.info("Queued %d clip jobs for project %s", len(clips), project_id)
        obs.bind(clips=len(clips), source_seconds=round(source_duration, 1))
        return {"success": True, "projectId": project_id, "queuedClips": len(clips)}


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/process-viral-clips-worker")
async def process_viral_clips_worker(
    project_id: str = Body(...),
    uid: str = Body(...),
    video_url: str = Body(""),
    x_internal_token: str | None = Header(None),
    x_cloudtasks_taskretrycount: str | None = Header(None),
    x_cloudtasks_taskexecutioncount: str | None = Header(None),
):
    """Cloud Tasks worker.

    Auth: Cloud Run IAM (OIDC token from the backend's Cloud Task) plus the
    optional X-Internal-Token check (contract C1).
    Status codes: 2xx = done (including handled failures, so Cloud Tasks does not
    retry a permanent error); 409 = another attempt is still running, retry later;
    5xx = transient, retry.
    """
    verify_internal_token(x_internal_token)
    if not PROJECT_ID_RE.fullmatch(project_id or "") or not UID_RE.fullmatch(uid or ""):
        raise HTTPException(status_code=400, detail="Invalid project_id or uid")

    attempt_id = uuid.uuid4().hex
    attempt = _cloud_tasks_attempt(x_cloudtasks_taskretrycount)
    obs.bind(job_id=attempt_id, project_id=project_id)
    job = {"job_type": "viral_clips", "attempt": attempt}
    if x_cloudtasks_taskexecutioncount is not None:
        job["execution_count"] = x_cloudtasks_taskexecutioncount
    obs.log_event("job_started", **job)
    if attempt > 1:
        obs.log_event("job_retry", logging.WARNING, reason="cloud_tasks_retry", **job)
    started = time.perf_counter()
    try:
        result = await run_viral_clips_job(project_id, uid, (video_url or "").strip(), attempt_id)
        ctx = obs.get_context()
        obs.log_event("job_done", duration_ms=obs.ms_since(started), status="duplicate" if result.get("duplicate") else "ok",
                      clips=ctx.get("clips"), source_seconds=ctx.get("source_seconds"), **job)
        return result
    except WorkerBusy:
        # job_duplicate_skipped(reason=lease_held) already logged; Cloud Tasks retries on 409.
        return JSONResponse(status_code=409, content={"detail": "Another attempt is running"})
    except ClaimRejected as e:
        # Never write to a project the caller does not own; ack so the task is not retried.
        logger.warning("Rejected task for project %s: %s", project_id, e)
        obs.log_event("job_failed", logging.WARNING, reason="claim_rejected", status="rejected",
                      duration_ms=obs.ms_since(started), **job)
        return {"success": False, "projectId": project_id, "error": str(e)}
    except PermanentJobError as e:
        logger.warning("Project %s failed permanently: %s", project_id, e)
        obs.log_event("job_failed", logging.WARNING, reason="permanent_error", status="failed",
                      error=type(e).__name__, duration_ms=obs.ms_since(started), **job)
        await asyncio.to_thread(fail_project_and_refund, project_id, uid, str(e), attempt_id)
        return {"success": False, "projectId": project_id, "error": str(e)}
    except Exception as e:
        # Unexpected errors are terminal for this run too (refund + failed);
        # retrying a half-done Gemini/publish step would only duplicate work.
        logger.error("Worker failed for project %s: %s", project_id, e, exc_info=True)
        obs.log_event("job_failed", logging.ERROR, reason="internal_error", status="failed",
                      error=type(e).__name__, duration_ms=obs.ms_since(started), **job)
        await asyncio.to_thread(
            fail_project_and_refund, project_id, uid, "Smart Clips processing failed. Please try again.", attempt_id
        )
        return {"success": False, "projectId": project_id, "error": "internal error"}


def _cloud_tasks_attempt(retry_count: str | None) -> int:
    """1-based delivery attempt from X-CloudTasks-TaskRetryCount (1 when absent)."""
    try:
        return max(0, int(retry_count or 0)) + 1
    except ValueError:
        return 1


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
