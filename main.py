from pathlib import Path
import os
import uuid
import time
import random
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import Body
from fastapi.responses import StreamingResponse
# Load environment variables from .env file
load_dotenv()

# Enable logging output for all loggers
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import hmac
import hashlib
from typing import List, Dict

import httpx
from google.cloud import storage
from firebase_admin import credentials, firestore, auth, storage as fb_storage
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "subtitlegen-input")
# Default to the full transcribe path used by your sample curl
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "https://whisper-infer-gpu-627061442405.asia-southeast1.run.app/transcribe")

app = FastAPI(title="CapsAI Minimal Upload", version="0 .0.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from gemini_analysis import analyze_audio_with_gemini
from video_processing import detect_faces_and_crop, srt_time_to_seconds

# ========= Firebase / Firestore =========
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "caps-85254")  # your real project id
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", f"{FIREBASE_PROJECT_ID}.firebasestorage.app")

import firebase_admin

if not firebase_admin._apps:
    cert_path = os.getenv("FIREBASE_CERT_PATH", "caps-85254-firebase-adminsdk.json")
    if not Path(cert_path).exists():
        raise FileNotFoundError(f"Firebase credentials file not found at {cert_path}")
    
    cred = credentials.Certificate(cert_path)
    firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": FIREBASE_STORAGE_BUCKET,
    })
    print("✅ Firebase initialized successfully")

db = firestore.client()

# GCP storage client using default credentials
# Set project via GOOGLE_CLOUD_PROJECT or GCP_PROJECT environment variable
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


# ------------------ Auth helpers ------------------
async def verify_firebase_token(authorization: str = Header(None)):
    """Verify Firebase ID token from Authorization header and return uid and user doc."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Load user doc if exists
        user_doc = db.collection("users").document(uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        return {"uid": uid, "user": user_data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")
        

# Authenticated endpoint to stream video from GCP
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from datetime import datetime, timedelta
from urllib.parse import urlparse

@app.get("/api/download-video")
def generate_signed_url(video_url: str):
    """Generate a signed URL for a video in Google Cloud Storage with 10 minutes expiration."""
    try:
        # Parse the provided public URL to extract the bucket name and blob name
        parsed_url = urlparse(video_url)
        path_parts = parsed_url.path.strip('/').split('/')

        if len(path_parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid video URL format")

        bucket_name = path_parts[0]  # The first part is the bucket name
        blob_name = '/'.join(path_parts[1:])  # The rest is the blob name

        # Get current time and add 10 minutes
        expiration_time = datetime.utcnow() + timedelta(minutes=10)

        # Generate the signed URL
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        signed_url = blob.generate_signed_url(
            version="v4", 
            expiration=expiration_time,  # Set expiration as current time + 10 minutes
            method="GET"
        )

        # Return the signed URL
        return {"signed_url": signed_url}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {e}")


# =============================
# Video Duration Helper Section
# =============================

async def download_to_tmp(url: str, suffix: str = ".mp4") -> Path:
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

async def download_from_gcs(gs_uri: str, suffix: str = ".mp4") -> Path:
    """Download a file from GCS (gs://bucket/path) to local temp storage."""
    logger = logging.getLogger("capsai.gcs.download")
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    
    try:
        # Parse gs://bucket/blob format
        if not gs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI format: {gs_uri}")
        
        parts = gs_uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS URI format: {gs_uri}")
        
        bucket_name, blob_name = parts
        
        logger.info(f"Downloading from GCS: bucket={bucket_name}, blob={blob_name}")
        
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to local file
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, blob.download_to_filename, str(tmp_path))
        
        logger.info(f"Downloaded from GCS to: {tmp_path}")
        return tmp_path
        
    except Exception as e:
        logger.error(f"GCS download failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GCS download failed: {e}")

def extract_audio_sync(video_path: str, audio_path: str) -> str:
    """Extract audio from video using ffmpeg (synchronous)."""
    import subprocess
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-q:a', '4',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        return audio_path
    except Exception as e:
        logger = logging.getLogger("capsai.audio")
        logger.error(f"Audio extraction failed: {e}")
        raise
    
# ==========================================
# VIRAL CLIPS GENERATION ENDPOINT (NEW)
# ==========================================

class ViralClipRequest(BaseModel):
    videoUrl: str

async def process_viral_clips_background(project_id: str, uid: str, video_url: str):
    """Background task to process viral clips and update Firestore document."""
    logger = logging.getLogger("capsai.viral_clips")
    logger.info(f"Background processing started for project {project_id}")
    
    tmp_video_path = None
    tmp_audio_path = None
    audio_gcp_url = None
    
    try:
        # 1. Download Video to Temp (supports both HTTP URLs and gs:// URIs)
        logger.info("Downloading video...")
        suffix = ".mp4"
        
        if video_url.startswith("gs://"):
            logger.info(f"Downloading from GCS: {video_url}")
            tmp_video_path = await download_from_gcs(video_url, suffix)
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


@app.post("/api/process-viral-clips-worker")
async def process_viral_clips_worker(
    project_id: str = Body(...),
    uid: str = Body(...),
    video_url: str = Body(...)
):
    """Worker endpoint called by Cloud Tasks. No auth required - uses service account."""
    logger = logging.getLogger("capsai.viral_clips")
    logger.info(f"Worker started for project {project_id}")
    
    await process_viral_clips_background(project_id, uid, video_url)
    
    return {"success": True, "projectId": project_id}


# @app.post("/api/generate-viral-clips")
# async def generate_viral_clips(
#     request: ViralClipRequest,
#     user=Depends(verify_firebase_token)
# ):
#     uid = user["uid"]
#     video_url = request.videoUrl
#     
#     logger = logging.getLogger("capsai.viral_clips")
#     logger.info(f"Starting viral clip generation for user {uid} on video {video_url}")

#     # Create Firestore document immediately with "processing" status
#     project_id = None
#     if db:
#         try:
#             doc_ref = db.collection("magi-clips").document()
#             project_id = doc_ref.id
#             
#             initial_data = {
#                 "uid": uid,
#                 "originalVideoUrl": video_url,
#                 "status": "processing",
#                 "createdAt": firestore.SERVER_TIMESTAMP,
#                 "clips": [],
#                 "clipCount": 0
#             }
#             
#             doc_ref.set(initial_data)
#             logger.info(f"Created Firestore document: {project_id}")
#             
#         except Exception as e:
#             logger.error(f"Failed to create Firestore document: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")
#     else:
#         raise HTTPException(status_code=500, detail="Database not available")
#     
#     # Process synchronously (foreground)
#     await process_viral_clips_background(project_id, uid, video_url)
#     
#     # Fetch the completed document to return full data
#     doc = db.collection("magi-clips").document(project_id).get()
#     result = doc.to_dict() if doc.exists else {}
#     
#     return {
#         "success": True,
#         "projectId": project_id,
#         "status": result.get("status", "completed"),
#         "clips": result.get("clips", []),
#         "clipCount": result.get("clipCount", 0),
#         "totalClipDuration": result.get("totalClipDuration", 0),
#         "minutesDeducted": result.get("minutesDeducted", 0)
#     }


@app.get("/api/me")
async def me(user=Depends(verify_firebase_token)):
    """Return basic user info and remaining minutes."""
    return {
        "uid": user["uid"],
        "user": user["user"],
        "videomins": user["user"].get("videomins", 0) if isinstance(user["user"], dict) else 0
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=int(os.getenv("PORT", 8080)))
