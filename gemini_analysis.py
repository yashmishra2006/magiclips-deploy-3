import google.generativeai as genai
import os
import json
import logging
import re
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ffmpeg
import time

def extract_audio(video_path: str) -> str:
    """Extracts audio from video to a temporary mp3 file if it doesn't already exist."""
    audio_path = "input.mp4.mp3"
    
    # Check if MP3 already exists
    if os.path.exists(audio_path):
        logger.info(f"Using existing MP3 file: {audio_path}")
        return audio_path
    
    try:
        logger.info(f"Extracting audio to: {audio_path}")
        # Fast audio extraction
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='libmp3lame', q=4, loglevel="quiet")
            .overwrite_output()
            .run()
        )
        return audio_path
    except ffmpeg.Error as e:
        logger.warning(f"Audio extraction failed, falling back to video upload: {e}")
        return video_path

def get_video_duration(video_path: str) -> float:
    """Get the duration of the video in seconds."""
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return 0.0

def srt_time_to_seconds(time_str):
    """Converts HH:MM:SS or HH:MM:SS,mmm to seconds."""
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
            
        time_str = str(time_str).replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except:
        return 0.0

def transcribe_audio(audio_path: str, gcs_uri: str = None) -> dict:
    """
    Transcribes audio using the external Whisper service.
    Returns the full JSON response from the service.
    """
    whisper_url = os.getenv("WHISPER_SERVICE_URL")
    if not whisper_url:
        logger.warning("WHISPER_SERVICE_URL not set. Skipping external transcription.")
        return None

    if gcs_uri:
        # New Whisper endpoint format
        url = f"{whisper_url.rstrip('/')}/transcribe"
        logger.info(f"Sending GCS URI to Whisper service: {url} | URI: {gcs_uri}")
        try:
            # Increased timeout for long audio files (15 minutes)
            with httpx.Client(timeout=900.0) as client:
                payload = {
                    "gs_url": gcs_uri,
                    "delete_after": False
                }
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Whisper GCS transcription failed: {e}")
            return None
    else:
        url = f"{whisper_url.rstrip('/')}/transcribe"
        logger.info(f"Sending audio to Whisper service: {url}")

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                # Increase timeout for large files
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(url, files=files)
                    response.raise_for_status()
                    return response.json()
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None

def format_transcript(whisper_data: dict) -> str:
    """Formats Whisper JSON into a readable transcript with timestamps."""
    if not whisper_data or "segments" not in whisper_data:
        return ""
    
    lines = []
    for seg in whisper_data["segments"]:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.2f} - {end:.2f}] {text}")
    
    return "\n".join(lines)

def analyze_audio_with_gemini(media_path: str, api_key: str = None, num_clips: int = 5, max_duration: int = 30, gcs_audio_uri: str = None) -> list:
    """
    Analyzes the audio/video file using Gemini 1.5 Flash to find viral moments.
    Returns a list of dicts: {'start': 'HH:MM:SS', 'end': 'HH:MM:SS', 'reason': '...'}
    
    Args:
        media_path: Local path to media file
        api_key: Google API key for Gemini
        num_clips: Number of viral clips to find
        max_duration: Max duration per clip in seconds
        gcs_audio_uri: Optional GCS URI (gs://...) of extracted audio to use for Whisper transcription
    """
    
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        
    if not api_key:
        logger.error("No Google API Key provided for Gemini analysis.")
        raise ValueError("GOOGLE_API_KEY not found")

    genai.configure(api_key=api_key)

    # Get video duration for validation
    total_duration_seconds = get_video_duration(media_path)
    total_duration_str = time.strftime('%H:%M:%S', time.gmtime(total_duration_seconds))
    logger.info(f"Video duration: {total_duration_str} ({total_duration_seconds} seconds)")

    temp_audio_path = None
    upload_path = media_path
    
    # Attempt audio extraction to save bandwidth/processing (only if no GCS URI provided)
    if not gcs_audio_uri and media_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        logger.info("Extracting audio for fa    ster analysis...")
        extracted = extract_audio(media_path)
        if extracted != media_path:
            upload_path = extracted
            temp_audio_path = extracted

    # Try to transcribe with Whisper first
    transcript_text = ""
    
    if gcs_audio_uri:
        # Use provided GCS audio URI for transcription
        logger.info(f"Using GCS audio URI for Whisper: {gcs_audio_uri}")
        whisper_data = transcribe_audio(None, gcs_uri=gcs_audio_uri)
        
        if whisper_data:
            transcript_text = format_transcript(whisper_data)
            logger.info(f"Transcription successful. Length: {len(transcript_text)} chars")
        else:
            logger.warning("Transcription failed or returned no data.")
    elif temp_audio_path or media_path.lower().endswith(('.mp3', '.wav', '.m4a')):
        audio_source = temp_audio_path if temp_audio_path else media_path
        logger.info(f"Attempting to transcribe from local file: {audio_source}")
        
        whisper_data = transcribe_audio(audio_source)
        
        if whisper_data:
            transcript_text = format_transcript(whisper_data)
            logger.info(f"Transcription successful. Length: {len(transcript_text)} chars")
        else:
            logger.warning("Transcription failed or returned no data.")

    try:

        # Create the model - Using Gemini 1.5 Pro for best video understanding
        model = genai.GenerativeModel(model_name="gemini-3-pro-preview")

        if transcript_text:
            # Prompt using the provided transcript
            prompt = f"""
            Task:
            Analyze the provided TRANSCRIPT to identify the TOP {num_clips} most viral moments suitable for {max_duration} seconds.
            
            TRANSCRIPT WITH TIMESTAMPS:
            {transcript_text}
            
            CRITICAL: 
            - Use the EXACT timestamps provided in the transcript.
            - Do NOT hallucinate timestamps. Trust the transcript.
            - ANY timestamp beyond {total_duration_seconds} seconds is INVALID.
            
            STRICT REQUIREMENTS:
            - Return EXACTLY {num_clips} clips maximum.
            - Each clip must be STRICTLY LESS THAN {max_duration} SECONDS.
            - Focus on the most emotionally engaging, controversial, or surprising moments.
            
            TIMESTAMP FORMAT:
            - Provide timestamps in SECONDS (integer or float).
            - Example: 120.5 (for 2 minutes and 0.5 seconds).
            
            Return STRICTLY valid JSON in this format:
            {{
              "clips": [
                {{
                  "start": 120.5, 
                  "end": 150.5,
                  "reason": "Why this specific moment is viral",
                  "transcript": "The spoken text in this clip"
                }}
              ]
            }}
            """
        else:
            # Fallback prompt (original logic)
            prompt = f"""
            Task:
            1. First, generate a COMPLETE TRANSCRIPT of the entire audio file.
            2. Then, analyze this transcript to identify the TOP {num_clips} most viral moments suitable for {max_duration} seconds.
            
            CRITICAL: 
            - You MUST listen to the ACTUAL audio content.
            - The timestamps must be EXACT and align perfectly with the spoken words in the transcript.
            - ANY timestamp beyond {total_duration_seconds} seconds is HALLUCINATED and INVALID.
            
            STRICT REQUIREMENTS:
            - Return EXACTLY {num_clips} clips maximum.
            - Each clip must be STRICTLY LESS THAN {max_duration} SECONDS.
            - Timestamps must correspond to ACTUAL moments in the audio.
            - Listen carefully and note the REAL time when interesting moments occur.
            - Focus on the most emotionally engaging, controversial, or surprising moments.
            
            TIMESTAMP FORMAT:
            - Provide timestamps in SECONDS (integer or float).
            - Example: 120.5 (for 2 minutes and 0.5 seconds).
            - DO NOT use HH:MM:SS format.
            
            TIMESTAMP ACCURACY: 
            - Listen to when each moment actually happens in the audio.
            - If you hear something interesting at 12 minutes in, timestamp it as 720 (12 * 60).
            - ABSOLUTELY NO TIMESTAMPS PAST {total_duration_seconds}.
            
            Return STRICTLY valid JSON in this format:
            {{
              "full_transcript": "The complete transcript of the video...",
              "clips": [
                {{
                  "start": 120.5, 
                  "end": 150.5,
                  "reason": "Why this specific moment is viral",
                  "transcript": "The spoken text in this clip"
                }}
              ]
            }}
            
            MAXIMUM {num_clips} CLIPS ONLY. Use REAL timestamps from the audio content.
            """

        response = model.generate_content([prompt], generation_config={"response_mime_type": "application/json"})
        
        logger.info(f"Gemini response: {response.text}")
        
        try:
            # Robust JSON extraction
            text = response.text
            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            # Find the outer-most JSON object
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            clips = []
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
                try:
                    data = json.loads(json_str)
                    clips = data.get("clips", [])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse extracted JSON, trying to fix...")
                    # Attempt to fix unclosed JSON (common if max tokens reached or hallucination)
                    # If it ends with a closing brace for an item but not the list/root
                    if json_str.strip().endswith('}'):
                         # Check if it looks like it's missing ]}
                         try:
                             fixed_json = json_str + "]}"
                             data = json.loads(fixed_json)
                             clips = data.get("clips", [])
                         except:
                             pass
            
            if not clips:
                # Fallback to simple load
                try:
                    data = json.loads(text)
                    clips = data.get("clips", [])
                except:
                    pass

            # Validate clips against video duration
            valid_clips = []
            for clip in clips:
                start_sec = srt_time_to_seconds(clip.get('start', 0))
                end_sec = srt_time_to_seconds(clip.get('end', 0))
                
                if start_sec > total_duration_seconds:
                    logger.warning(f"Skipping hallucinated clip starting at {start_sec}s (video duration: {total_duration_seconds}s)")
                    continue
                    
                if end_sec > total_duration_seconds:
                    logger.warning(f"Adjusting clip end time from {end_sec}s to {total_duration_seconds}s")
                    end_sec = total_duration_seconds
                    clip['end'] = end_sec

                # Enforce max_duration
                duration = end_sec - start_sec
                if duration > max_duration:
                    logger.warning(f"Clip duration {duration}s exceeds max {max_duration}s. Truncating.")
                    end_sec = start_sec + max_duration
                    clip['end'] = end_sec
                
                # Ensure start and end are numbers in the output
                clip['start'] = start_sec
                clip['end'] = end_sec
                
                valid_clips.append(clip)
                
            return valid_clips

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return []

    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        raise e
    finally:
        # Cleanup local temp audio
        try:
            # if temp_audio_path and os.path.exists(temp_audio_path):
            #     os.unlink(temp_audio_path)
            pass
        except:
            pass
