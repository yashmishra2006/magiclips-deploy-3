from pathlib import Path
import os
import uuid
import time
import re
import asyncio
import logging
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
from video_processing import detect_faces_and_crop, srt_time_to_seconds


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
    """Background task to process viral clips and update Firestore document."""
    logger = logging.getLogger("capsai.viral_clips")
    logger.info(f"Background processing started for project {project_id}")
    
    tmp_video_path = None
    tmp_audio_path = None
    audio_gcp_url = None
    
    try:
        # 1. Download Video to Temp (supports gs://, YouTube, and HTTP URLs)
        logger.info("Downloading video...")
        suffix = ".mp4"
        
        if video_url.startswith("gs://"):
            logger.info(f"Downloading from GCS: {video_url}")
            tmp_video_path = await download_from_gcs(video_url, suffix)
        elif is_youtube_url(video_url):
            logger.info(f"Downloading from YouTube: {video_url}")
            tmp_video_path = await download_youtube_video(video_url, UPLOAD_DIR)
        else:
            logger.info(f"Downloading from HTTP URL: {video_url}")
            tmp_video_path = await download_to_tmp(video_url, suffix)
        
        # 2. Extract Audio
        logger.info("Extracting audio...")
        audio_filename = f"audio_{uid}_{int(time.time())}.mp3"
        tmp_audio_path = UPLOAD_DIR / audio_filename
        await asyncio.to_thread(extract_audio_sync, str(tmp_video_path), str(tmp_audio_path))
        
        # 3. Upload Audio to GCP
        logger.info("Uploading audio to GCP...")
        audio_gcp_url = await upload_to_gcs(tmp_audio_path, audio_filename)
        logger.info(f"Audio uploaded: {audio_gcp_url}")
        
        # 4. Analyze with Gemini (pass gs:// audio URI to Whisper)
        logger.info("Analyzing with Gemini & Whisper...")
        viral_clips_meta = await asyncio.to_thread(
            analyze_audio_with_gemini, 
            str(tmp_video_path),
            gcs_audio_uri=audio_gcp_url
        )
        
        if not viral_clips_meta:
            raise HTTPException(status_code=400, detail="No viral clips found by Gemini.")
            
        logger.info(f"Gemini found {len(viral_clips_meta)} clips: {viral_clips_meta}")

        generated_clips = []
        
        # 5. Process each clip
        for i, clip_meta in enumerate(viral_clips_meta):
            start_time = clip_meta.get("start")
            end_time = clip_meta.get("end")
            reason = clip_meta.get("reason")
            
            if not start_time or not end_time:
                continue
                
            clip_filename = f"viral_{uid}_{int(time.time())}_{i}.mp4"
            clip_local_path = UPLOAD_DIR / clip_filename
            
            # A. Smart Crop & Cut
            logger.info(f"Processing clip {i+1}: {start_time}-{end_time}")
            
            # Run blocking CV2/FFmpeg in thread pool
            processed_path = await asyncio.to_thread(
                detect_faces_and_crop, 
                str(tmp_video_path), 
                str(clip_local_path), 
                start_time, 
                end_time
            )
            
            if not processed_path:
                logger.error(f"Clip {i+1} processing failed. Skipping.")
                continue
            
            # B. Upload Clip to GCS
            gcs_url = await upload_to_gcs(Path(processed_path), clip_filename)
            
            # Cleanup clip temp files
            try:
                Path(processed_path).unlink()
            except:
                pass
                
            generated_clips.append({
                "videoUrl": gcs_url,
                "start": start_time,
                "end": end_time,
                "reason": reason,
                "duration": srt_time_to_seconds(end_time) - srt_time_to_seconds(start_time)
            })
        
        if not generated_clips:
            raise HTTPException(status_code=500, detail="Failed to generate any clips")
            
        # 6. Calculate total duration and minutes to deduct
        from video_processing import get_video_duration
        original_duration = get_video_duration(str(tmp_video_path))
        
        total_duration_seconds = sum(c.get("duration", 0) for c in generated_clips)
        minutes_to_deduct = round(total_duration_seconds / 60.0, 2)
        
        # 7. Update Firestore document with completed status
        if db:
            try:
                doc_data = {
                    "originalVideoUrl": video_url,
                    "audioGcpUrl": audio_gcp_url,
                    "originalDuration": original_duration,
                    "clips": generated_clips,
                    "clipCount": len(generated_clips),
                    "totalClipDuration": total_duration_seconds,
                    "minutesDeducted": minutes_to_deduct,
                    "status": "completed",
                    "completedAt": firestore.SERVER_TIMESTAMP
                }
                
                doc_ref = db.collection("magi-clips").document(project_id)
                doc_ref.update(doc_data)
                logger.info(f"Updated Firestore document: {project_id}")
            except Exception as e:
                logger.error(f"Failed to update Firestore: {e}")
        
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
        
        # Update Firestore with error status
        if db:
            try:
                doc_ref = db.collection("magi-clips").document(project_id)
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
        reload=False
    )