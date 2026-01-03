import os
import cv2
import ffmpeg
import logging
import torch
import numpy as np
import subprocess
from sklearn.cluster import KMeans
from collections import deque

from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

FFMPEG_BINARY = '/usr/local/bin/ffmpeg'
if os.path.exists(FFMPEG_BINARY):
    os.environ['FFMPEG_BINARY'] = FFMPEG_BINARY
    ffmpeg._run.DEFAULT_FFMPEG_PATH = FFMPEG_BINARY

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if CUDA_AVAILABLE else "cpu"

YOLO_FACE_MODEL_PATH = "yolov12n-face.pt"
FACE_PADDING_RATIO = 0.15
MAX_ANALYSIS_SECONDS = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if CUDA_AVAILABLE:
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU processing")

# ============================================================
# LOAD YOLO FACE MODEL
# ============================================================

try:
    FACE_MODEL = YOLO(YOLO_FACE_MODEL_PATH)
    logger.info(f"Loaded YOLO face model: {YOLO_FACE_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO face model: {e}")
    raise


# ============================================================
# UTIL: TIME PARSER
# ============================================================

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


# ============================================================
# OPTIMIZED FACE DETECTION & CROPPING
# ============================================================

def detect_faces_and_crop(video_path: str, output_path: str, start_time: str, end_time: str):
    """
    Optimized smart 9:16 vertical crop with:
      - Efficient face detection (skip frames)
      - K-Means only on mode transitions
      - Proper audio merging
      - Better error handling
    """

    # --------------------------------------------------------
    # AUDIO SOURCE DETECTION
    # --------------------------------------------------------
    mp3_audio_path = video_path + ".mp3"
    has_separate_audio = os.path.exists(mp3_audio_path)

    if has_separate_audio:
        logger.info(f"Found separate audio file: {mp3_audio_path}")

    # --------------------------------------------------------
    # TIME RANGE
    # --------------------------------------------------------
    t_start = srt_time_to_seconds(start_time)
    t_end = srt_time_to_seconds(end_time)
    duration = t_end - t_start

    if duration <= 0:
        logger.error(f"Invalid time range: start={t_start}, end={t_end}")
        return None

    logger.info(f"Processing clip: {start_time} to {end_time} ({duration:.2f}s)")

    # --------------------------------------------------------
    # OPEN VIDEO & PROPS
    # --------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width == 0 or height == 0:
        logger.error("Failed to read video properties")
        cap.release()
        return None

    logger.info(f"Video props: {width}x{height} @ {fps:.2f} fps")

    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)

    # --------------------------------------------------------
    # CROP DIMENSIONS
    # --------------------------------------------------------
    target_aspect = 9 / 16
    crop_height = height
    crop_width = int(crop_height * target_aspect)
    if crop_width > width:
        crop_width = width
        crop_height = int(width / target_aspect)

    split_crop_height = int(crop_width * (8 / 9))

    # --------------------------------------------------------
    # OUTPUT SETUP
    # --------------------------------------------------------
    out_w, out_h = 720, 1280
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'h264_nvenc' if CUDA_AVAILABLE else 'libx264',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        temp_video_path
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, 
        stdin=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    # --------------------------------------------------------
    # STATE VARIABLES
    # --------------------------------------------------------
    history_buffer = deque(maxlen=10)
    frames_to_process = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    logger.info("Starting optimized frame processing (Exact Visualization Logic)...")

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # --------------------------------------------------------
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {i}")
            break

        # 1. Detect Faces (Every frame for smoothness)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = FACE_MODEL.predict(rgb_frame, device=DEVICE, verbose=False, conf=0.5)

        current_frame_faces = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                current_frame_faces.append((cx, cy))

        current_frame_faces.sort(key=lambda p: p[0])
        history_buffer.append(current_frame_faces)

        # 2. Decide Mode based on History (Last 10 frames)
        dual_frame_count = 0
        all_points = []

        for faces in history_buffer:
            if len(faces) >= 2:
                if abs(faces[-1][0] - faces[0][0]) > width * 0.15:
                    dual_frame_count += 1
            all_points.extend(faces)

        is_split_mode = dual_frame_count >= (len(history_buffer) / 2) and len(history_buffer) > 0
        
        # 3. Render
        final_frame = None

        if is_split_mode:
            # Split Mode: Find 2 clusters in all_points (x, y)
            if len(all_points) >= 2:
                points_array = np.array(all_points)
                try:
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points_array)
                    centers = kmeans.cluster_centers_
                    centers = sorted(centers, key=lambda p: p[0])
                    c1 = centers[0]
                    c2 = centers[1]
                except Exception:
                    c1 = (width // 3, height // 2)
                    c2 = (2 * width // 3, height // 2)
            else:
                c1 = (width // 3, height // 2)
                c2 = (2 * width // 3, height // 2)

            def get_crop_rect(center, w, h, max_w, max_h):
                cx, cy = center
                x = int(cx - (w // 2))
                y = int(cy - (h // 2))
                x = max(0, min(max_w - w, x))
                y = max(0, min(max_h - h, y))
                return int(x), int(y)

            x1, y1 = get_crop_rect(c1, crop_width, split_crop_height, width, height)
            x2, y2 = get_crop_rect(c2, crop_width, split_crop_height, width, height)

            crop1 = frame[y1:y1+split_crop_height, x1:x1+crop_width]
            crop2 = frame[y2:y2+split_crop_height, x2:x2+crop_width]

            if crop1.size > 0 and crop2.size > 0:
                target_half_h = out_h // 2
                r1 = cv2.resize(crop1, (out_w, target_half_h))
                r2 = cv2.resize(crop2, (out_w, target_half_h))
                final_frame = np.vstack((r1, r2))
            else:
                final_frame = cv2.resize(frame, (out_w, out_h))

        else:
            # Single Mode: Average center
            if all_points:
                avg_x = int(np.mean([p[0] for p in all_points]))
            else:
                avg_x = width // 2

            x = avg_x - (crop_width // 2)
            x = max(0, min(width - crop_width, x))

            crop_img = frame[0:height, x:x+crop_width]
            if crop_img.size > 0:
                final_frame = cv2.resize(crop_img, (out_w, out_h))
            else:
                final_frame = cv2.resize(frame, (out_w, out_h))

        # Write frame to FFmpeg
        try:
            if process.poll() is not None:
                # Process died unexpectedly
                _, stderr = process.communicate()
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                logger.error(f"FFmpeg process died at frame {i}: {error_msg}")
                break
                
            process.stdin.write(final_frame.tobytes())
            
        except BrokenPipeError:
            logger.error(f"FFmpeg pipe broken at frame {i}")
            _, stderr = process.communicate()
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            logger.error(f"FFmpeg stderr: {error_msg}")
            break
        except Exception as e:
            logger.error(f"Error writing frame {i}: {e}")
            break

    # --------------------------------------------------------
    # CLEANUP & FINALIZE VIDEO ENCODING
    # --------------------------------------------------------
    logger.info("Finalizing video encoding...")
    
    try:
        # Important: subprocess.communicate() may attempt to flush stdin.
        # If we already closed it, this can raise "flush of closed file".
        if process.stdin is not None:
            try:
                process.stdin.close()
            finally:
                # Detach stdin so communicate() doesn't try to flush it.
                process.stdin = None

        stdout, stderr = process.communicate(timeout=30)
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg encoding failed: {error_msg}")
            cap.release()
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg encoding timeout, terminating...")
        process.kill()
        stdout, stderr = process.communicate()
        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error during FFmpeg finalization: {e}")
        process.kill()
        cap.release()
        return None

    cap.release()
    logger.info("Frame processing complete")
    
    # Verify temp file was created successfully
    if not os.path.exists(temp_video_path):
        logger.error(f"Temp video file not created: {temp_video_path}")
        return None
        
    # Check if temp file is valid (has non-zero size)
    temp_size = os.path.getsize(temp_video_path)
    if temp_size < 1024:  # Less than 1KB is suspicious
        logger.error(f"Temp video file too small ({temp_size} bytes), likely corrupted")
        return None
        
    logger.info(f"Temp video created: {temp_video_path} ({temp_size / 1024 / 1024:.2f} MB)")

    # --------------------------------------------------------
    # AUDIO MERGE (FIXED & OPTIMIZED)
    # --------------------------------------------------------
    try:
        logger.info("Merging audio with processed video...")

        # 1. Determine audio source
        audio_input_path = None

        if has_separate_audio:
            audio_input_path = mp3_audio_path
            logger.info(f"Using separate audio: {mp3_audio_path}")
        else:
            # Check for audio in original video
            try:
                probe = ffmpeg.probe(video_path)
                audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
                if audio_streams:
                    audio_input_path = video_path
                    logger.info("Using audio from original video")
                else:
                    logger.warning("No audio stream found")
            except Exception as e:
                logger.warning(f"Could not probe for audio: {e}")

        # 2. Build FFmpeg merge command
        if audio_input_path:
            # Use manual command for better debugging
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,           # Video input (already encoded)
                '-ss', str(t_start),             # Audio start time
                '-t', str(duration),             # Audio duration
                '-i', audio_input_path,          # Audio input
                '-map', '0:v:0',                 # Map first video stream
                '-map', '1:a:0',                 # Map first audio stream
                '-c:v', 'copy',                  # Copy video (no re-encode)
                '-c:a', 'aac',                   # Encode audio to AAC
                '-b:a', '192k',                  # Audio bitrate
                '-shortest',                     # Match shortest stream
                output_path
            ]
            
            logger.info(f"Audio merge command: {' '.join(merge_cmd)}")
            
            result = subprocess.run(
                merge_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg audio merge failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise Exception(f"Audio merge failed with return code {result.returncode}")

        else:
            # No audio available - just copy video
            logger.warning("No audio available. Creating video-only output.")
            
            copy_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c', 'copy',
                '-an',  # No audio
                output_path
            ]
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Video copy failed: {result.stderr}")
                raise Exception("Failed to copy video")

        # 3. Cleanup temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info("Removed temporary video file")

        logger.info(f"✓ Clip created successfully: {output_path}")
        return output_path

    except subprocess.TimeoutExpired as e:
        logger.error(f"Subprocess timeout: {e}")
        if os.path.exists(temp_video_path):
            logger.warning(f"Temp video preserved at: {temp_video_path}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during audio processing: {e}")
        if os.path.exists(temp_video_path):
            logger.warning(f"Temp video preserved at: {temp_video_path}")
        raise e


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Example usage
    detect_faces_and_crop(
        video_path="input_video.mp4",
        output_path="output_vertical.mp4",
        start_time="00:00:10",
        end_time="00:00:40"
    )