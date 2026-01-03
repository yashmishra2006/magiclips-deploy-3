from pathlib import Path
import os
import uuid
import time
import random
import asyncio
import subprocess
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import Body
from fastapi.responses import StreamingResponse
import io

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
import google
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage as fb_storage
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import base64
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "subtitlegen-input")
# Default to the full transcribe path used by your sample curl
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "https://whisper-infer-gpu-627061442405.asia-southeast1.run.app/transcribe")

app = FastAPI(title="CapsAI Minimal Upload", version="0.0.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


import google.generativeai
from gemini_analysis import analyze_audio_with_gemini
from video_processing import detect_faces_and_crop, srt_time_to_seconds

# ========= Firebase / Firestore =========
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "caps-85254")  # your real project id
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", f"{FIREBASE_PROJECT_ID}.firebasestorage.app")

# Make Firebase optional for testing
db = None
firebase_initialized = False

if not firebase_admin._apps:
    # cert_path = os.getenv("FIREBASE_CERT_PATH", "caps-85254-firebase-adminsdk.json")
    if Path(cert_path).exists():
        try:
            cred = credentials.Certificate(cert_path)
            firebase_admin.initialize_app(cred, {
                "projectId": FIREBASE_PROJECT_ID,
                "storageBucket": FIREBASE_STORAGE_BUCKET,
            })
            db = firestore.client()
            firebase_initialized = True
            print("✅ Firebase initialized successfully")
        except Exception as e:
            print(f"⚠️ Firebase initialization failed: {e}")
            print("🔄 Running in test mode without Firebase")
    else:
        print(f"⚠️ Firebase credentials file not found at {cert_path}")
        print("🔄 Running in test mode without Firebase")


# GCP storage functions are disabled for local/dev
def get_storage_client():
    return storage.Client()

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
    
# ==========================================
# VIRAL CLIPS GENERATION ENDPOINT (NEW)
# ==========================================

class ViralClipRequest(BaseModel):
    videoUrl: str

@app.post("/api/generate-viral-clips")
async def generate_viral_clips(
    request: ViralClipRequest, 
):
    # uid = user["uid"]
    video_url = request.videoUrl
    
    logger = logging.getLogger("capsai.viral_clips")
    logger.info(f"Starting viral clip generation for user {uid} on video {video_url}")

    tmp_video_path = None
    
    try:
        # 1. Download Video to Temp
        logger.info("Downloading video...")
        suffix = ".mp4"
        tmp_video_path = await download_to_tmp(video_url, suffix)
        
        # 2. Analyze with Gemini (Audio)
        logger.info("Analyzing with Gemini...")
        viral_clips_meta = await asyncio.to_thread(analyze_audio_with_gemini, str(tmp_video_path))
        
        if not viral_clips_meta:
            raise HTTPException(status_code=400, detail="No viral clips found by Gemini.")
            
        logger.info(f"Gemini found {len(viral_clips_meta)} clips: {viral_clips_meta}")

        generated_clips = []
        
        # 3. Process each clip
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
            
            # B. Upload Clip to GCS
            gcs_url = await upload_to_gcs(Path(processed_path), clip_filename)
            
            # # C. Transcribe Clip (Optional: Can be done by frontend or separate call)
            # # For now, we just return the clip URL so frontend can call /api/process-video or similar if needed
            # # OR we can call forward_to_whisper here if "Auto-Subtitle" is desired immediately.
            
            # # Let's get the transcription for this short clip to be helpful
            # whisper_result = await forward_to_whisper(gcs_url, word_timestamps=True)
            # srt_content = transcription_json_to_srt(whisper_result, 4)
            
            # # Upload SRT
            # srt_filename = clip_filename.replace(".mp4", ".srt")
            # srt_local_path = UPLOAD_DIR / srt_filename
            # with open(srt_local_path, "w", encoding="utf-8") as f:
            #     f.write(srt_content)
            # srt_url = await upload_to_gcs(srt_local_path, srt_filename)
            
            # Cleanup clip temp files
            try:
                Path(processed_path).unlink()
                # srt_local_path.unlink()
            except:
                pass
                
            generated_clips.append({
                "videoUrl": gcs_url,
                # "srtUrl": srt_url,
                "start": start_time,
                "end": end_time,
                "reason": reason,
                # "transcription": srt_content
            })
            
        # Cleanup original video
        if tmp_video_path and tmp_video_path.exists():
            tmp_video_path.unlink()

    except Exception as e:
        logger.error(f"Viral clip generation failed: {e}", exc_info=True)
        if tmp_video_path and tmp_video_path.exists():
            try:
                tmp_video_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Debit Minutes (Billing)
    # Calculate total duration in minutes
    total_duration_seconds = 0.0
    for clip in generated_clips:
        try:
            s = srt_time_to_seconds(clip["start"])
            e = srt_time_to_seconds(clip["end"])
            total_duration_seconds += (e - s)
        except:
            pass
            
    minutes_to_deduct = round(total_duration_seconds / 60.0, 2)
    
    if minutes_to_deduct > 0:
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

    return {
        "success": True,
        "clips": generated_clips,
        "minutes_deducted": minutes_to_deduct
    }


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
