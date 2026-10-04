from google import genai
import os
import json
import logging
import re
import httpx
from internal_auth import internal_auth_headers
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ffmpeg
import time

# Vertex AI Configuration (Enterprise)
_GCP_SA_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "new-backend-475414-228aab3a465a.json")
if os.path.exists(_GCP_SA_KEY):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _GCP_SA_KEY
VERTEX_PROJECT = os.getenv("GCP_PROJECT_ID", "new-backend-475414")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
TRANSCRIPT_CHUNK_MAX_CHARS = int(os.getenv("TRANSCRIPT_CHUNK_MAX_CHARS", "12000"))
TRANSCRIPT_CHUNK_OVERLAP_SEGMENTS = int(os.getenv("TRANSCRIPT_CHUNK_OVERLAP_SEGMENTS", "6"))
PASS1_MAX_CANDIDATES_PER_CHUNK = int(os.getenv("PASS1_MAX_CANDIDATES_PER_CHUNK", "10"))
MAX_CANDIDATES_FOR_RERANK = int(os.getenv("MAX_CANDIDATES_FOR_RERANK", "60"))
CLIP_MIN_DURATION = float(os.getenv("CLIP_MIN_DURATION", "10"))
GEMINI_JSON_RETRIES = int(os.getenv("GEMINI_JSON_RETRIES", "3"))
MAX_TRANSCRIPT_FOR_RERANK_CHARS = int(os.getenv("MAX_TRANSCRIPT_FOR_RERANK_CHARS", "45000"))
TIME_RANGE_RE = re.compile(r"^\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)$")


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
        # Try ffmpeg first
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except Exception as e:
        logger.warning(f"ffmpeg probe failed for duration: {e}. Trying cv2 fallback...")
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0:
                duration = frame_count / fps
                logger.info(f"cv2 duration: {duration}s")
                return duration
        except Exception as cv2_e:
            logger.error(f"cv2 duration check failed: {cv2_e}")

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
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except Exception:
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
        url = f"{whisper_url.rstrip('/')}/transcribe"
        logger.info(f"Sending GCS URI to Whisper service: {url} | URI: {gcs_uri}")
        try:
            with httpx.Client(timeout=900.0) as client:
                payload = {
                    "gs_url": gcs_uri,
                    "delete_after": False
                }
                # C1: Whisper-Magiclips is IAM-only; send an ID token (+ X-Internal-Token).
                response = client.post(url, json=payload, headers=internal_auth_headers(url))
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Whisper GCS transcription failed: {e}")
            return None

    url = f"{whisper_url.rstrip('/')}/transcribe"
    logger.info(f"Sending audio to Whisper service: {url}")

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, files=files, headers=internal_auth_headers(url))
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    clean = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"```\s*", "", clean)

    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start_idx = clean.find("{")
    end_idx = clean.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    try:
        data = json.loads(clean[start_idx:end_idx + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _call_gemini_for_json(model_fn, model_name: str, prompt: str, expected_key: str) -> Dict[str, Any]:
    """Call Vertex AI Gemini and parse JSON response"""
    last_text = ""
    for attempt in range(1, GEMINI_JSON_RETRIES + 1):
        try:
            response = model_fn(
                model=model_name,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            last_text = response.text if hasattr(response, 'text') else str(response)
            data = _parse_json_from_text(last_text)
            if isinstance(data, dict) and expected_key in data and isinstance(data[expected_key], list):
                return data

            logger.warning(
                "Gemini JSON parse/schema mismatch on attempt %d/%d (expected key: %s)",
                attempt,
                GEMINI_JSON_RETRIES,
                expected_key,
            )
        except Exception as e:
            logger.warning(f"Gemini API call failed on attempt {attempt}/{GEMINI_JSON_RETRIES}: {e}")
            last_text = str(e)

    logger.error("Gemini failed to return valid JSON for key '%s'. Last response: %s", expected_key, last_text[:800])
    return {expected_key: []}


def _parse_transcript_segments(transcript_text: str) -> List[Dict[str, Any]]:
    segments = []
    for line in transcript_text.splitlines():
        match = TIME_RANGE_RE.match(line.strip())
        if not match:
            continue
        start_sec = _safe_float(match.group(1), 0.0)
        end_sec = _safe_float(match.group(2), start_sec)
        text = match.group(3).strip()
        if end_sec <= start_sec:
            end_sec = start_sec + 0.1
        segments.append({"start": start_sec, "end": end_sec, "text": text})
    return segments


def _segment_to_line(seg: Dict[str, Any]) -> str:
    return f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}"


def _build_transcript_chunks(transcript_text: str) -> List[Dict[str, Any]]:
    segments = _parse_transcript_segments(transcript_text)
    if not segments:
        if not transcript_text.strip():
            return []
        return [{
            "chunk_index": 1,
            "start_sec": 0.0,
            "end_sec": 0.0,
            "text": transcript_text[:TRANSCRIPT_CHUNK_MAX_CHARS],
        }]

    chunks: List[Dict[str, Any]] = []
    start_idx = 0
    chunk_index = 1

    while start_idx < len(segments):
        end_idx = start_idx
        chars = 0

        while end_idx < len(segments):
            next_line = _segment_to_line(segments[end_idx])
            if end_idx > start_idx and chars + len(next_line) + 1 > TRANSCRIPT_CHUNK_MAX_CHARS:
                break
            chars += len(next_line) + 1
            end_idx += 1

        chunk_segments = segments[start_idx:end_idx]
        if not chunk_segments:
            break

        chunks.append({
            "chunk_index": chunk_index,
            "start_sec": chunk_segments[0]["start"],
            "end_sec": chunk_segments[-1]["end"],
            "text": "\n".join(_segment_to_line(s) for s in chunk_segments),
        })
        chunk_index += 1

        if end_idx >= len(segments):
            break

        next_start = max(end_idx - TRANSCRIPT_CHUNK_OVERLAP_SEGMENTS, start_idx + 1)
        start_idx = next_start

    return chunks


def _normalize_candidate(
    clip: Dict[str, Any],
    total_duration_seconds: float,
    max_duration: float,
    clip_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    start_raw = clip.get("start_sec", clip.get("start", 0))
    end_raw = clip.get("end_sec", clip.get("end", 0))

    start_sec = max(0.0, srt_time_to_seconds(start_raw))
    end_sec = srt_time_to_seconds(end_raw)
    if total_duration_seconds > 0:
        end_sec = min(end_sec, total_duration_seconds)

    if end_sec <= start_sec:
        return None

    duration = end_sec - start_sec
    if duration > max_duration:
        end_sec = start_sec + max_duration
        duration = max_duration

    if duration < CLIP_MIN_DURATION:
        if total_duration_seconds > 0 and start_sec + CLIP_MIN_DURATION <= total_duration_seconds:
            end_sec = start_sec + CLIP_MIN_DURATION
        else:
            start_sec = max(0.0, end_sec - CLIP_MIN_DURATION)
        duration = end_sec - start_sec

    if duration <= 0:
        return None

    reason = str(clip.get("reason", "")).strip()
    if not reason:
        reason = "Strong standalone moment from transcript."

    hook_type = str(clip.get("hook_type", "other")).strip().lower() or "other"
    must_include_quote = str(clip.get("must_include_quote", clip.get("transcript", ""))).strip()
    confidence = _safe_float(clip.get("confidence", 0.5), 0.5)
    score_hook = _safe_float(clip.get("score_hook", 0.0), 0.0)
    score_payoff = _safe_float(clip.get("score_payoff", 0.0), 0.0)
    score_clarity = _safe_float(clip.get("score_clarity", 0.0), 0.0)
    score_shareability = _safe_float(clip.get("score_shareability", 0.0), 0.0)

    normalized = {
        "id": clip_id or str(clip.get("id", "")).strip(),
        "start": round(start_sec, 2),
        "end": round(end_sec, 2),
        "duration": round(end_sec - start_sec, 2),
        "reason": reason,
        "hook_type": hook_type,
        "must_include_quote": must_include_quote[:240],
        "confidence": max(0.0, min(1.0, confidence)),
        "score_hook": max(0.0, min(10.0, score_hook)),
        "score_payoff": max(0.0, min(10.0, score_payoff)),
        "score_clarity": max(0.0, min(10.0, score_clarity)),
        "score_shareability": max(0.0, min(10.0, score_shareability)),
    }

    return normalized


def _overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    left = max(a["start"], b["start"])
    right = min(a["end"], b["end"])
    intersection = max(0.0, right - left)
    if intersection <= 0:
        return 0.0
    len_a = max(0.001, a["end"] - a["start"])
    len_b = max(0.001, b["end"] - b["start"])
    return intersection / min(len_a, len_b)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _text_jaccard(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _composite_score(c: Dict[str, Any]) -> float:
    return (
        c.get("score_shareability", 0.0)
        + c.get("score_hook", 0.0)
        + c.get("score_payoff", 0.0)
        + c.get("score_clarity", 0.0)
        + 2.5 * c.get("confidence", 0.0)
    )


def _dedupe_candidates(candidates: List[Dict[str, Any]], keep_max: Optional[int]) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=_composite_score, reverse=True)

    selected: List[Dict[str, Any]] = []
    for clip in ordered:
        is_duplicate = False
        for other in selected:
            overlap = _overlap_ratio(clip, other)
            reason_sim = _text_jaccard(clip.get("reason", ""), other.get("reason", ""))
            quote_sim = _text_jaccard(clip.get("must_include_quote", ""), other.get("must_include_quote", ""))
            if overlap >= 0.70 or (overlap >= 0.45 and max(reason_sim, quote_sim) >= 0.45):
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(clip)
        if keep_max is not None and len(selected) >= keep_max:
            break
    return selected


def _build_pass1_prompt(
    transcript_chunk: str,
    chunk_index: int,
    chunk_start_sec: float,
    chunk_end_sec: float,
    total_duration_seconds: float,
    max_duration: float,
) -> str:
    return f"""
You are selecting viral short-form clip candidates from a transcript chunk.

Goal:
- Maximize standalone clarity, hook strength in first 2 seconds, and payoff by clip end.

Constraints:
- Use only timestamps from this transcript chunk.
- Output time in seconds.
- Every candidate duration must be between {CLIP_MIN_DURATION:.0f} and {max_duration:.0f} seconds.
- Timestamps must stay within [0, {total_duration_seconds:.2f}].
- Prefer moments with conflict, surprise, strong opinion, concrete advice, confession, or clear punchline.

Chunk info:
- chunk_index: {chunk_index}
- chunk_start_sec: {chunk_start_sec:.2f}
- chunk_end_sec: {chunk_end_sec:.2f}

Transcript chunk:
{transcript_chunk}

Return STRICT JSON:
{{
  "candidates": [
    {{
      "start_sec": 0.0,
      "end_sec": 0.0,
      "reason": "Why this is viral-worthy",
      "hook_type": "surprise|conflict|insight|emotion|humor|story",
      "must_include_quote": "short quote from transcript",
      "confidence": 0.0
    }}
  ]
}}

Generate up to {PASS1_MAX_CANDIDATES_PER_CHUNK} candidates for this chunk.
""".strip()


def _build_pass2_prompt(
    transcript_text: str,
    candidates: List[Dict[str, Any]],
    total_duration_seconds: float,
    num_clips: Optional[int],
    max_duration: float,
) -> str:
    candidate_payload = []
    for c in candidates:
        candidate_payload.append({
            "id": c.get("id"),
            "start_sec": c.get("start"),
            "end_sec": c.get("end"),
            "reason": c.get("reason"),
            "hook_type": c.get("hook_type"),
            "must_include_quote": c.get("must_include_quote"),
            "confidence": c.get("confidence"),
        })

    transcript_for_prompt = transcript_text[:MAX_TRANSCRIPT_FOR_RERANK_CHARS]

    if num_clips is None:
        selection_instruction = "- Select all high-quality clips for virality and standalone comprehension."
    else:
        selection_instruction = f"- Select the best {num_clips} clips for virality and standalone comprehension."

    return f"""
You are ranking and refining short-form clip candidates from a full transcript.

Instructions:
{selection_instruction}
- You may refine boundaries by up to +/-5 seconds from each candidate.
- Keep durations between {CLIP_MIN_DURATION:.0f} and {max_duration:.0f} seconds.
- Avoid near-duplicate clips; prioritize topic diversity.
- Do not use timestamps beyond {total_duration_seconds:.2f}.

Full transcript (possibly truncated):
{transcript_for_prompt}

Candidates JSON:
{json.dumps(candidate_payload, ensure_ascii=True)}

Return STRICT JSON:
{{
  "clips": [
    {{
      "id": "candidate id",
      "start_sec": 0.0,
      "end_sec": 0.0,
      "reason": "Short practical reason",
      "score_hook": 0.0,
      "score_payoff": 0.0,
      "score_clarity": 0.0,
      "score_shareability": 0.0,
      "confidence": 0.0,
      "must_include_quote": "short quote"
    }}
  ]
}}
""".strip()


def analyze_audio_with_gemini(
    media_path: str,
    api_key: str = None,
    num_clips: Optional[int] = None,
    max_duration: int = 90,
    gcs_audio_uri: str = None,
    total_duration_seconds: float = None,
) -> list:
    """
    Analyzes the audio/video file using Gemini to find viral moments.
    Returns a list of dicts with at minimum: {'start': seconds, 'end': seconds, 'reason': '...'}.

    Args:
        media_path: Local path to media file
        api_key: Deprecated - now uses Vertex AI service account (ignored if provided)
        num_clips: Number of viral clips to return. If None/<=0, return all high-quality clips.
        max_duration: Max duration per clip in seconds
        gcs_audio_uri: Optional GCS URI (gs://...) of extracted audio to use for Whisper transcription
    """

    # Initialize Vertex AI client (uses service account from GOOGLE_APPLICATION_CREDENTIALS)
    try:
        gemini_client = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
        )
        logger.info(f"Initialized Vertex AI Gemini client: project={VERTEX_PROJECT}, location={VERTEX_LOCATION}, model={MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {e}")
        raise ValueError(f"Vertex AI initialization failed: {e}")

    if total_duration_seconds is None or total_duration_seconds <= 0:
        total_duration_seconds = get_video_duration(media_path) if media_path else 0.0
    total_duration_str = time.strftime('%H:%M:%S', time.gmtime(total_duration_seconds))
    logger.info(f"Video duration: {total_duration_str} ({total_duration_seconds} seconds)")

    temp_audio_path = None

    if not gcs_audio_uri and media_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        logger.info("Extracting audio for faster analysis...")
        extracted = extract_audio(media_path)
        if extracted != media_path:
            temp_audio_path = extracted

    transcript_text = ""

    if gcs_audio_uri:
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

    if not transcript_text.strip():
        logger.warning("Empty transcript. Returning no clips.")
        return []

    try:
        # Create model instance using Vertex AI client
        model = gemini_client.models.generate_content
        model_name = MODEL_NAME
        clip_limit = num_clips if isinstance(num_clips, int) and num_clips > 0 else None

        chunks = _build_transcript_chunks(transcript_text)
        logger.info("Built %d transcript chunks for pass-1 candidate generation", len(chunks))

        all_candidates: List[Dict[str, Any]] = []
        next_id = 1

        for chunk in chunks:
            pass1_prompt = _build_pass1_prompt(
                transcript_chunk=chunk["text"],
                chunk_index=chunk["chunk_index"],
                chunk_start_sec=chunk["start_sec"],
                chunk_end_sec=chunk["end_sec"],
                total_duration_seconds=total_duration_seconds,
                max_duration=max_duration,
            )
            data = _call_gemini_for_json(model, model_name, pass1_prompt, expected_key="candidates")
            raw_candidates = data.get("candidates", [])

            for raw in raw_candidates:
                normalized = _normalize_candidate(
                    raw,
                    total_duration_seconds=total_duration_seconds,
                    max_duration=max_duration,
                    clip_id=f"cand_{next_id}",
                )
                next_id += 1
                if normalized:
                    all_candidates.append(normalized)

        if not all_candidates:
            logger.warning("No candidates from pass-1.")
            return []

        # When clip_limit is None (return all clips), don't limit pass-1 candidates
        # When clip_limit is set, limit pass-1 to MAX_CANDIDATES_FOR_RERANK for efficiency
        pass1_keep_max = None if clip_limit is None else MAX_CANDIDATES_FOR_RERANK
        deduped = _dedupe_candidates(all_candidates, keep_max=pass1_keep_max)
        logger.info("Pass-1 produced %d candidates, %d after dedupe", len(all_candidates), len(deduped))

        pass2_prompt = _build_pass2_prompt(
            transcript_text=transcript_text,
            candidates=deduped,
            total_duration_seconds=total_duration_seconds,
            num_clips=clip_limit,
            max_duration=max_duration,
        )
        pass2_data = _call_gemini_for_json(model, model_name, pass2_prompt, expected_key="clips")
        pass2_raw = pass2_data.get("clips", [])

        candidate_index = {c["id"]: c for c in deduped if c.get("id")}
        final_clips: List[Dict[str, Any]] = []

        for clip in pass2_raw:
            clip_id = str(clip.get("id", "")).strip()
            base = candidate_index.get(clip_id, {})
            merged = {**base, **clip}
            normalized = _normalize_candidate(
                merged,
                total_duration_seconds=total_duration_seconds,
                max_duration=max_duration,
                clip_id=clip_id or base.get("id"),
            )
            if normalized:
                final_clips.append(normalized)

        if not final_clips:
            logger.warning("Pass-2 returned no valid clips. Falling back to pass-1 ranking.")
            final_clips = deduped[:clip_limit] if clip_limit is not None else deduped

        final_clips = _dedupe_candidates(final_clips, keep_max=clip_limit)
        final_clips = sorted(final_clips, key=_composite_score, reverse=True)
        if clip_limit is not None:
            final_clips = final_clips[:clip_limit]

        logger.info("Returning %d final clips", len(final_clips))
        return final_clips
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        raise e
    finally:
        # Cleanup local temp audio if you make extraction path unique in future.
        try:
            pass
        except Exception:
            pass
