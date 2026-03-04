from pathlib import Path
import os

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import uuid
import time
import re
import asyncio
import json
import subprocess
from datetime import timedelta, datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import httpx
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore, auth

import yt_dlp
from gemini_analysis import analyze_audio_with_gemini


# ──────────────────────────────────────────────────────────────
# Basic setup & logging
# ──────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s'
)
logger = logging.getLogger("capsai.main")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "subtitlegen-input")
CLIP_OUTPUT_BUCKET = os.getenv("CLIP_OUTPUT_BUCKET", "subtitlegen-output")
CLIP_OUTPUT_PREFIX = os.getenv("CLIP_OUTPUT_PREFIX", "viral_")
CLIP_OUTPUT_EXT = os.getenv("CLIP_OUTPUT_EXT", ".mp4")
LOGO_PATH = Path(os.getenv("WATERMARK_LOGO_PATH", "logo.png"))
WATERMARK_MARGIN_PX = int(os.getenv("WATERMARK_MARGIN_PX", "24"))
WATERMARK_WIDTH_RATIO = float(os.getenv("WATERMARK_WIDTH_RATIO", "0.16"))

# Firebase initialization
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "caps-85254")
FIREBASE_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    f"{FIREBASE_PROJECT_ID}.appspot.com"
)

if not firebase_admin._apps:
    cert_path = os.getenv("FIREBASE_CERT_PATH", "caps-85254-firebase-adminsdk.json")
    if not Path(cert_path).exists():
        raise FileNotFoundError(f"Firebase credentials not found: {cert_path}")

    cred = credentials.Certificate(cert_path)
    firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": FIREBASE_STORAGE_BUCKET,
    })
    logger.info("Firebase initialized successfully")

db = firestore.client()

def get_storage_client():
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or FIREBASE_PROJECT_ID
    cert_path = os.getenv("FIREBASE_CERT_PATH", "new-backend-475414-228aab3a465a.json")
    if Path(cert_path).exists():
        return storage.Client.from_service_account_json(cert_path, project=project)
    return storage.Client(project=project)

async def upload_to_gcs(local_path: Path, dest_name: str) -> str:
    import logging
    logger = logging.getLogger("capsai.gcp.upload")
    try:
        logger.info(f"Starting GCP upload: local_path={local_path}, dest_name={dest_name}, bucket={GCP_BUCKET_NAME}")
        client = get_storage_client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(dest_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, blob.upload_from_filename, str(local_path))
        logger.info(f"Upload to GCP bucket succeeded: gs://{bucket.name}/{dest_name}")
        # Return the GCP gs:// URL (not signed, not public)
        gcp_url = f"gs://{bucket.name}/{dest_name}"
        logger.info(f"Returning GCP URL: {gcp_url}")
        return gcp_url
    except Exception as e:
        logger.error(f"GCP upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GCP upload failed: {e}")


# ──────────────────────────────────────────────────────────────
# YouTube helpers
# ──────────────────────────────────────────────────────────────

YOUTUBE_REGEX = re.compile(
    r'(?:https?:\/\/)?(?:www\.)?(?:'
    r'youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.*?v=|shorts\/)'
    r')([\w\-]{11})(?:[^\w\-]|$)'
)

def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.search(url))

async def process_youtube_via_external_api(youtube_url: str) -> str:
    """
    Call external API to process YouTube video and return GCS URL.
    Returns: gs:// URL of the processed video
    """
    api_url = "http://34.131.185.69:8002/process"
    # api_url = "http://localhost:8002/process"
    
    logger.info(f"Calling external API for YouTube URL: {youtube_url}")
    
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.get(api_url, params={"url": youtube_url})
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "success":
                raise HTTPException(500, f"External API returned error: {data}")
            
            gcs_url = data.get("gcs_url")
            if not gcs_url:
                raise HTTPException(500, "External API did not return gcs_url")
            
            logger.info(f"External API returned GCS URL: {gcs_url}")
            return gcs_url
            
    except httpx.HTTPError as e:
        logger.error(f"External API call failed: {e}")
        raise HTTPException(500, f"Failed to process YouTube URL via external API: {str(e)}")

async def download_youtube_video(url: str, output_dir: Path) -> Path:
    """
    Download YouTube video using yt-dlp
    Returns path to downloaded file
    """
    video_id_match = YOUTUBE_REGEX.search(url)
    if not video_id_match:
        raise ValueError("Not a valid YouTube video URL")

    video_id = video_id_match.group(1)
    filename = f"yt_{video_id}_{int(time.time())}.mp4"
    output_path = output_dir / filename

    cookies_path = os.getenv("COOKIES_PATH", "/cookies.txt")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'quiet': True,
        'continuedl': True,
        'retries': 10,
        'fragment_retries': 10,
        'merge_output_format': 'mp4',

        # ──── Most important changes ─────
        'cookiefile': cookies_path if Path(cookies_path).is_file() else None,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],   # very helpful in 2026
                'lang': ['en'],
            }
        },
        # Optional but often helps
        'force_generic_extractor': False,
        'source_address': '0.0.0.0',  # sometimes helps in containers
    }

    logger.info(f"Starting YouTube download → {url} (id: {video_id})")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await asyncio.to_thread(ydl.download, [url])

        if not output_path.exists():
            raise RuntimeError("Downloaded file not found after yt-dlp finished")

        logger.info(f"YouTube video saved → {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"YouTube download failed: {e}", exc_info=True)
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass
        raise

# ──────────────────────────────────────────────────────────────
# Download helpers (HTTP + GCS)
# ──────────────────────────────────────────────────────────────

async def download_to_tmp(url: str, suffix: str = ".mp4") -> Path:
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        logger.error(f"HTTP download failed: {e}")
        raise HTTPException(500, f"Download failed: {str(e)}")


async def download_from_gcs(gs_uri: str, suffix: str = ".mp4") -> Path:
    if not gs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI format")

    _, path = gs_uri.split("gs://", 1)
    bucket_name, *parts = path.split("/", 1)
    blob_name = parts[0] if parts else ""

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    logger.info(f"Downloading from GCS: {gs_uri}")
    await asyncio.to_thread(blob.download_to_filename, str(tmp_path))

    return tmp_path


def extract_audio_sync(video_path: str, audio_path: str) -> str:
    import subprocess
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'libmp3lame',
            '-q:a', '4',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}", exc_info=True)
        raise

# ──────────────────────────────────────────────────────────────
# Parallel Cloud Run viral clip processing
# ──────────────────────────────────────────────────────────────


def srt_time_to_seconds(time_str) -> float:
    """Converts HH:MM:SS / MM:SS / seconds to float."""
    if isinstance(time_str, (int, float)):
        return float(time_str)

    try:
        time_str = str(time_str).strip()
        time_str = time_str.replace(', ', ':').replace('. ', ':').replace(',', '.').replace(' ', '')

        parts = time_str.split(':')
        seconds = 0.0

        if len(parts) == 3:
            seconds += int(parts[0]) * 3600
            seconds += int(parts[1]) * 60
            seconds += float(parts[2])
        elif len(parts) == 2:
            seconds += int(parts[0]) * 60
            seconds += float(parts[1])
        elif len(parts) == 1:
            seconds = float(parts[0])

        logger.info(f"Parsed time '{time_str}' -> {seconds}s")
        return seconds
    except Exception as e:
        logger.error(f"Time parsing error for '{time_str}': {e}")
        return 0.0


CLOUD_RUN_URL = "https://smart-clip-processor-627061442405.asia-southeast1.run.app/process-fields"


async def process_clips_parallel(clips: list) -> list:
    """
    Process all clips in parallel by calling the Cloud Run endpoint.
    Each item in clips should be a dict with the required fields for /process-fields.
    Returns a list of (status_code, response_json) tuples.
    """

    results = []
    async with httpx.AsyncClient(timeout=600.0) as client:
        tasks = [client.post(CLOUD_RUN_URL, json=clip) for clip in clips]
        print(clips)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, Exception):
                results.append((None, {"error": str(resp)}))
            else:
                try:
                    results.append((resp.status_code, resp.json()))
                except Exception as e:
                    results.append((resp.status_code, {"error": str(e)}))
    return results


def resolve_user_status(user_data: dict) -> str:
    """Resolve account tier from user document data."""
    if not isinstance(user_data, dict):
        return "free"

    status_fields = [
        user_data.get("usertype"),
        user_data.get("userType"),
        user_data.get("status"),
        user_data.get("plan"),
        user_data.get("tier"),
        user_data.get("subscription"),
    ]
    status_text = " ".join(str(v).strip().lower() for v in status_fields if v is not None)

    if user_data.get("isPremium") is True or user_data.get("premium") is True:
        return "premium"
    if any(token in status_text for token in ("premium", "pro", "paid", "plus")):
        return "premium"
    if any(token in status_text for token in ("free", "basic")):
        return "free"
    return "free"


def should_generate_watermarked_clips(user_status: str) -> bool:
    """Current product rule: generate watermark variants for both free and premium users."""
    normalized = (user_status or "free").strip().lower()
    return normalized in {"free", "premium"}


def _parse_json_maybe(value):
    """Parse JSON if value is a JSON string; otherwise return None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_output_video_url(response_data: dict) -> str | None:
    """Best-effort extraction of output clip URL from Cloud Run response JSON."""
    preferred_keys = (
        "videoUrl", "video_url", "video",
        "clipUrl", "clip_url",
        "outputUrl", "output_url",
        "gcs_url", "gcsUrl",
        "signed_url", "signedUrl",
        "download_url", "downloadUrl",
        "url",
    )

    def is_valid_url(value: str) -> bool:
        return isinstance(value, str) and value.startswith(("gs://", "https://", "http://"))

    def looks_like_blob_path(value: str) -> bool:
        if not isinstance(value, str):
            return False
        v = value.strip()
        if not v:
            return False
        if v.startswith(("gs://", "http://", "https://")):
            return False
        # Typical blob path in this project.
        return "/" in v and "." in v

    def from_blob_to_gs(value: str) -> str:
        blob_name = value.lstrip("/")
        return f"gs://{GCP_BUCKET_NAME}/{blob_name}"

    def walk(node):
        if isinstance(node, dict):
            # 1) direct preferred keys (case-insensitive)
            lowered = {str(k).lower(): v for k, v in node.items()}
            for wanted in preferred_keys:
                value = lowered.get(wanted.lower())
                if is_valid_url(value):
                    return value

            # 2) blob/path key fallback
            for k, v in node.items():
                kl = str(k).lower()
                if "input" in kl:
                    continue
                if isinstance(v, str) and ("blob" in kl or "path" in kl) and looks_like_blob_path(v):
                    return from_blob_to_gs(v)

            # 3) recurse into parsed JSON string fields first (e.g., body)
            for k, v in node.items():
                if isinstance(v, str):
                    parsed = _parse_json_maybe(v)
                    if parsed is not None:
                        found = walk(parsed)
                        if found:
                            return found
                    if is_valid_url(v):
                        return v

            # 4) recurse generic
            for k, v in node.items():
                if "input" in str(k).lower():
                    continue
                found = walk(v)
                if found:
                    return found

        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found

        elif isinstance(node, str):
            if is_valid_url(node):
                return node
            parsed = _parse_json_maybe(node)
            if parsed is not None:
                return walk(parsed)

        return None

    return walk(response_data)


def infer_output_url_from_clip_id(clip_id: str) -> str | None:
    """Infer deterministic output location from clip_id when Cloud Run omits URL."""
    if not isinstance(clip_id, str) or not clip_id.strip():
        return None
    clean_id = clip_id.strip()
    return f"gs://{CLIP_OUTPUT_BUCKET}/{CLIP_OUTPUT_PREFIX}{clean_id}{CLIP_OUTPUT_EXT}"


def add_logo_watermark_sync(
    input_video: Path,
    output_video: Path,
    logo_path: Path,
    margin_px: int = WATERMARK_MARGIN_PX,
    width_ratio: float = WATERMARK_WIDTH_RATIO,
) -> None:
    """Overlay logo at top-right corner using ffmpeg."""
    if not input_video.exists():
        raise FileNotFoundError(f"Input clip not found: {input_video}")
    if not logo_path.exists():
        raise FileNotFoundError(f"Logo file not found: {logo_path}")

    filter_complex = (
        f"[1:v][0:v]scale2ref=w=main_w*{width_ratio}:h=ow/mdar[logo][video];"
        f"[video][logo]overlay=W-w-{margin_px}:{margin_px}"
    )

    base_cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-i", str(logo_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
    ]

    # Attempt audio stream copy first; fall back to AAC re-encode when needed.
    cmd_copy_audio = base_cmd + ["-c:a", "copy", str(output_video)]
    result = subprocess.run(cmd_copy_audio, capture_output=True, text=True)
    if result.returncode == 0:
        return

    cmd_encode_audio = base_cmd + ["-c:a", "aac", "-b:a", "128k", str(output_video)]
    result2 = subprocess.run(cmd_encode_audio, capture_output=True, text=True)
    if result2.returncode != 0:
        raise RuntimeError(
            "Watermark ffmpeg failed.\n"
            f"copy-audio stderr: {result.stderr[-1000:]}\n"
            f"encode-audio stderr: {result2.stderr[-1000:]}"
        )

# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CapsAI - Viral Clips Generator",
    description="Generate viral clips from YouTube / direct video links",
    version="0.1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_firebase_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded["uid"]
        user_doc = db.collection("users").document(uid).get()
        return {
            "uid": uid,
            "user_data": user_doc.to_dict() if user_doc.exists else {}
        }
    except Exception as e:
        raise HTTPException(401, f"Token verification failed: {str(e)}")


class ViralClipRequest(BaseModel):
    videoUrl: str   # more permissive than HttpUrl


# ──────────────────────────────────────────────────────────────
# Background processing with YouTube support
# ──────────────────────────────────────────────────────────────

async def process_viral_clips_background(project_id: str, uid: str, video_url: str):
    """Background task to process viral clips using parallel Lambda invocation."""
    logger = logging.getLogger("capsai.viral_clips")
    logger.info(f"Background processing started for project {project_id}")

    tmp_video_path = None
    tmp_audio_path = None
    audio_gcp_url = None
    input_video_gcp_uri = None
    user_data = {}
    user_status = "free"
    generated_clips = []
    watermark_clips = []

    def update_status(status: str, **extra_fields):
        if not db:
            return
        try:
            doc_ref = db.collection("projects").document(project_id)
            payload = {
                "status": status,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
            if extra_fields:
                payload.update(extra_fields)
            doc_ref.update(payload)
            logger.info(f"Status updated → {status}")
        except Exception as e:
            logger.error(f"Failed to update status to {status}: {e}")
    
    try:
        # Resolve user status once for tier-based behavior.
        if db:
            try:
                user_doc_snapshot = db.collection("users").document(uid).get()
                if user_doc_snapshot.exists:
                    user_data = user_doc_snapshot.to_dict() or {}
                user_status = resolve_user_status(user_data)
                logger.info(f"Resolved user status for {uid}: {user_status}")
            except Exception as e:
                logger.warning(f"Could not resolve user status for {uid}, defaulting to free: {e}")
                user_status = "free"

        update_status("downloading")
        suffix = ".mp4"
        
        if video_url.startswith("gs://"):
            logger.info(f"Downloading from GCS: {video_url}")
            tmp_video_path = await download_from_gcs(video_url, suffix)
            input_video_gcp_uri = video_url # Already in GCS
        elif is_youtube_url(video_url):
            logger.info(f"Processing YouTube URL via external API: {video_url}")
            # Call external API to process YouTube video and get GCS URL
            gcs_url = await process_youtube_via_external_api(video_url)
            # Download the processed video from GCS
            tmp_video_path = await download_from_gcs(gcs_url, suffix)
            input_video_gcp_uri = gcs_url # Already in GCS
        else:
            logger.info(f"Downloading from HTTP URL: {video_url}")
            tmp_video_path = await download_to_tmp(video_url, suffix)
        
        # Will upload to GCS later
        
        # 2. Extract Audio 
        update_status("extracting_audio")
        logger.info("Extracting audio...")
        audio_filename = f"audio_{uid}_{int(time.time())}.mp3"
        tmp_audio_path = UPLOAD_DIR / audio_filename
        await asyncio.to_thread(extract_audio_sync, str(tmp_video_path), str(tmp_audio_path))
        
        # 3. Upload Audio to GCP 
        update_status("uploading_audio")
        logger.info("Uploading audio to GCP...")
        audio_gcp_url = await upload_to_gcs(tmp_audio_path, audio_filename)
        logger.info(f"Audio uploaded: {audio_gcp_url}")

        # Ensure Source Video is in GCS for Lambda
        if not input_video_gcp_uri:
            update_status("uploading_video")
            video_filename = f"input_{uid}_{int(time.time())}.mp4"
            logger.info(f"Uploading source video to GCS for Lambda: {video_filename}")
            input_video_gcp_uri = await upload_to_gcs(tmp_video_path, video_filename)
        
        # 4. Analyze with Gemini (SKIPPED FOR TESTING)
        update_status("transcribing")
        viral_clips_meta = await asyncio.to_thread(
            analyze_audio_with_gemini, 
            str(tmp_video_path),
            gcs_audio_uri=audio_gcp_url
        )
    
        update_status("finding_clips")
        
        if not viral_clips_meta:
            raise HTTPException(status_code=400, detail="No viral clips found by Gemini.")
            
        logger.info(f"Gemini found {len(viral_clips_meta)} clips: {viral_clips_meta}")


        # 5. Parallel Processing via Cloud Run
        # Prepare payload for each clip
        # Extract blob name from GCS URI (gs://bucket/blob...)
        if input_video_gcp_uri.startswith("gs://"):
            parts = input_video_gcp_uri[5:].split('/', 1)
            if len(parts) == 2:
                input_blob_name = parts[1]
            else:
                input_blob_name = ''
        else:
            input_blob_name = input_video_gcp_uri

        # Generate signed URL for input video (GET) and use it for both download and Cloud Run
        storage_client = get_storage_client()
        bucket = storage_client.bucket(GCP_BUCKET_NAME)
        def generate_signed_url(blob_name, method="GET"):
            blob = bucket.blob(blob_name)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=15),
                method=method,
                content_type="video/mp4" if method == "PUT" else None
            )
            return url

        input_signed_url = generate_signed_url(input_blob_name, "GET")
        # Use input_signed_url for both download (if needed) and for Cloud Run payload
        cloud_run_clips = []
        for i, clip_meta in enumerate(viral_clips_meta):
            start_time = clip_meta.get("start")
            end_time = clip_meta.get("end")
            if not start_time or not end_time:
                continue
            # Convert SRT time to float seconds if needed
            if isinstance(start_time, str):
                start_sec = srt_time_to_seconds(start_time)
                logger.info(f"Parsed time '{start_time}' -> {start_sec}s")
            else:
                start_sec = float(start_time)
            if isinstance(end_time, str):
                end_sec = srt_time_to_seconds(end_time)
                logger.info(f"Parsed time '{end_time}' -> {end_sec}s")
            else:
                end_sec = float(end_time)
            clip_id = f"{uid}_{int(time.time())}_{i}"
            payload = {
                "input_signed_url": input_signed_url,
                "start": start_sec,  # ensure float, not string
                "end": end_sec,      # ensure float, not string
                "clip_id": clip_id,
                "project_id": project_id,
                "reason": clip_meta.get("reason", ""),
                "duration": end_sec - start_sec
            }
            logger.info(f"Cloud Run payload for clip {i}: {payload}")
            cloud_run_clips.append(payload)

        update_status("processing_clips_parallel")
        results = await process_clips_parallel(cloud_run_clips)

        success_count = 0
        for i, (status, response) in enumerate(results):
            logger.info(f"Cloud Run response for clip index {i}: status={status}, body={response}")
            if status == 200 and response.get("Status") == "SUCCEEDED":
                success_count += 1
                logger.info(f"Cloud Run success for clip index {i}")
                payload = cloud_run_clips[i]
                output_url = extract_output_video_url(response)
                if not output_url:
                    inferred = infer_output_url_from_clip_id(payload.get("clip_id"))
                    if inferred:
                        output_url = inferred
                        logger.info(
                            "Inferred output URL for clip index %d from clip_id: %s",
                            i,
                            output_url,
                        )
                    else:
                        logger.warning(f"Could not extract output URL for clip index {i}. Response payload: {response}")
                else:
                    logger.info(f"Extracted output URL for clip index {i}: {output_url}")
                clip_entry = {
                    "videoUrl": output_url,
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "reason": payload.get("reason", ""),
                    "duration": payload.get("duration", 0),
                    "processedAt": datetime.utcnow().isoformat(),
                }
                generated_clips.append(clip_entry)
            else:
                logger.error(f"Cloud Run returned failure: {response}")

        if success_count == 0 and len(viral_clips_meta) > 0:
            raise HTTPException(status_code=500, detail="Failed to generate any clips via Cloud Run")

        # If Cloud Run writes clips directly to Firestore, use those as fallback
        # when output URLs were not extracted from HTTP responses.
        if db:
            try:
                has_any_output_url = any(c.get("videoUrl") for c in generated_clips)
                if not has_any_output_url:
                    project_snapshot = db.collection("projects").document(project_id).get()
                    project_data = project_snapshot.to_dict() if project_snapshot.exists else {}
                    existing_clips = project_data.get("clips", [])
                    if isinstance(existing_clips, list) and existing_clips:
                        generated_clips = existing_clips
                        logger.info(
                            "Loaded %d existing clips from Firestore fallback for project %s",
                            len(generated_clips),
                            project_id,
                        )
            except Exception as existing_clip_error:
                logger.warning(f"Could not load existing clips fallback: {existing_clip_error}")

        # 6. Generate watermark clips (top-right logo overlay) using ffmpeg.
        if should_generate_watermarked_clips(user_status) and generated_clips:
            if not LOGO_PATH.exists():
                logger.error(f"Watermark logo missing at {LOGO_PATH}. Skipping watermark generation.")
            else:
                update_status("watermarking")
                for i, clip in enumerate(generated_clips):
                    clip_url = clip.get("videoUrl")
                    if not clip_url:
                        logger.warning(f"Skipping watermark for clip {i}: output URL missing")
                        continue

                    local_input = None
                    local_output = None
                    try:
                        if str(clip_url).startswith("gs://"):
                            local_input = await download_from_gcs(str(clip_url), suffix=".mp4")
                        elif str(clip_url).startswith(("http://", "https://")):
                            local_input = await download_to_tmp(str(clip_url), suffix=".mp4")
                        else:
                            logger.warning(f"Unsupported clip URL format for watermarking: {clip_url}")
                            continue

                        local_output = UPLOAD_DIR / f"wm_{project_id}_{uid}_{int(time.time())}_{i}.mp4"
                        await asyncio.to_thread(
                            add_logo_watermark_sync,
                            local_input,
                            local_output,
                            LOGO_PATH,
                            WATERMARK_MARGIN_PX,
                            WATERMARK_WIDTH_RATIO,
                        )

                        dest_name = f"watermarked/{project_id}/wm_{uid}_{int(time.time())}_{i}.mp4"
                        watermark_gcs_url = await upload_to_gcs(local_output, dest_name)

                        watermark_entry = dict(clip)
                        watermark_entry["videoUrl"] = watermark_gcs_url
                        watermark_clips.append(watermark_entry)
                    except Exception as wm_error:
                        logger.error(f"Watermark generation failed for clip index {i}: {wm_error}")
                    finally:
                        if local_input and local_input.exists():
                            try:
                                local_input.unlink()
                            except Exception:
                                pass
                        if local_output and local_output.exists():
                            try:
                                local_output.unlink()
                            except Exception:
                                pass
            
        minutes_to_deduct = 0 # SKIPPED FOR TESTING
        # 7. Final Status Update
        if db:
            try:
                doc_ref = db.collection("projects").document(project_id)
                doc_ref.update({
                    "status": "completed",
                    "completedAt": firestore.SERVER_TIMESTAMP,
                    "clipCount": len(generated_clips) if generated_clips else success_count,
                    "clips": generated_clips,
                    "watermarkclips": watermark_clips,
                    "watermarkClipCount": len(watermark_clips),
                    "userStatus": user_status,
                })
                logger.info(f"Project {project_id} marked as completed")
            except Exception as e:
                logger.error(f"Failed to update final status: {e}")
        
        # 8. Debit Minutes (Billing)
        if minutes_to_deduct > 0 and db:
            logger.info(f"Debiting {minutes_to_deduct} minutes from user {uid}")
            try:
                user_ref = db.collection("users").document(uid)
                user_doc = user_ref.get()
                if user_doc.exists:
                    current_mins = float(user_doc.to_dict().get("videomins", 0.0) or 0.0)
                    new_balance = max(current_mins - minutes_to_deduct, 0.0)
                    user_ref.update({"videomins": new_balance, "updatedAt": firestore.SERVER_TIMESTAMP})
                    logger.info(f"User {uid} new balance: {new_balance}")
            except Exception as e:
                logger.error(f"Failed to debit minutes for user {uid}: {e}")
        
        logger.info(f"Background processing completed for project {project_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for project {project_id}: {e}", exc_info=True)

        update_status("failed", error=str(e), failedAt=firestore.SERVER_TIMESTAMP)
        
        # Update Firestore with error status
        if db:
            try:
                doc_ref = db.collection("projects").document(project_id)
                doc_ref.update({
                    "status": "failed",
                    "error": str(e),
                    "failedAt": firestore.SERVER_TIMESTAMP
                })
            except Exception as update_error:
                logger.error(f"Failed to update error status: {update_error}")
    
    finally:
        # Cleanup temp files
        if tmp_video_path and tmp_video_path.exists():
            try:
                tmp_video_path.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup video: {e}")
        if tmp_audio_path and tmp_audio_path.exists():
            try:
                tmp_audio_path.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup audio: {e}")



# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/process-viral-clips-worker")
async def process_viral_clips_worker(
    project_id: str = Body(...),
    uid: str = Body(...),
    video_url: str = Body(...)
):
    """Worker endpoint called by Cloud Tasks. No auth required - uses service account."""
    logger.info(f"Worker started for project {project_id}")
    
    await process_viral_clips_background(project_id, uid, video_url)
    
    return {"success": True, "projectId": project_id}


# @app.post("/api/generate-viral-clips")
# async def generate_viral_clips(
#     request: ViralClipRequest,
#     user = Depends(verify_firebase_token)
# ):
#     uid = user["uid"]
#     video_url = request.videoUrl.strip()

#     logger.info(f"[user:{uid}] New viral clips request → {video_url}")

#     # Create processing document
#     doc_ref = db.collection("magi-clips").document()
#     project_id = doc_ref.id

#     doc_ref.set({
#         "uid": uid,
#         "originalVideoUrl": video_url,
#         "status": "processing",
#         "createdAt": firestore.SERVER_TIMESTAMP,
#         "clips": [],
#         "clipCount": 0
#     }, merge=True)

#     # Start background task
#     asyncio.create_task(
#         process_viral_clips_background(project_id, uid, video_url)
#     )

#     return {
#         "success": True,
#         "projectId": project_id,
#         "status": "processing",
#         "message": "Viral clip generation started"
#     }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=True
    )
