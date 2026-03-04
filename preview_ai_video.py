import cv2
import numpy as np
import argparse
import os
import subprocess
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

YOLO_MODEL_PATH = "yolov8s.pt"  # Person detection
DETECTION_WIDTH = 640
FRAME_SKIP = 3
SCENE_THRESHOLD = 0.5

ANALYSIS_FRAMES = 20              # Frames to analyze at scene start
MIN_PERSON_AREA_RATIO = 0.02      # Ignore tiny background persons
MOTION_MIN_PIXELS = 150
SMOOTHING_FACTOR = 0.85           # Crop smoothing (0-1)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_motion_center(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < MOTION_MIN_PIXELS:
        return None
    cy, cx = coords.mean(axis=0)
    return int(cx), int(cy)


def filter_large_persons(results, scale_ratio, width, height):
    persons = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rx1, ry1, rx2, ry2 = x1/scale_ratio, y1/scale_ratio, x2/scale_ratio, y2/scale_ratio

            area = (rx2 - rx1) * (ry2 - ry1)
            if area > width * height * MIN_PERSON_AREA_RATIO:
                cx = int((rx1 + rx2) / 2)
                cy = int((ry1 + ry2) / 2)
                persons.append((cx, cy, area))

    persons = sorted(persons, key=lambda x: x[2], reverse=True)
    return persons[:2]


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_preview(video_path, output_path):

    print(f"Loading model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 9:16 crop
    target_aspect = 9 / 16
    crop_height = height
    crop_width = int(crop_height * target_aspect)
    if crop_width > width:
        crop_width = width
        crop_height = int(width / target_aspect)

    # 9:8 split crop
    split_aspect = 9 / 8
    split_crop_height = height
    split_crop_width = int(split_crop_height * split_aspect)
    if split_crop_width > width:
        split_crop_width = width
        split_crop_height = int(width / split_aspect)

    out_w, out_h = 720, 1280

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    prev_frame_for_scene = None
    prev_gray = None
    frame_idx = 0

    # Scene-level state
    current_mode = None
    analysis_counter = 0
    scene_persons = []
    scene_motion = []
    locked_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        viz_frame = frame.copy()
        scene_cut = False

        # ---------------- SCENE DETECTION ----------------
        if prev_frame_for_scene is not None:
            small_curr = cv2.resize(frame, (160, 90))
            small_prev = cv2.resize(prev_frame_for_scene, (160, 90))

            hist_curr = cv2.calcHist([cv2.cvtColor(small_curr, cv2.COLOR_BGR2HSV)], [0], None, [32], [0, 256])
            hist_prev = cv2.calcHist([cv2.cvtColor(small_prev, cv2.COLOR_BGR2HSV)], [0], None, [32], [0, 256])

            cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)

            score = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)

            if score < SCENE_THRESHOLD:
                scene_cut = True

        prev_frame_for_scene = frame.copy() if frame_idx % 2 == 0 else prev_frame_for_scene

        # Reset scene-level state
        if scene_cut:
            current_mode = None
            analysis_counter = 0
            scene_persons = []
            scene_motion = []
            locked_center = None

        # ---------------- SCENE ANALYSIS WINDOW ----------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_mode is None:
            analysis_counter += 1

            # Person detection
            if frame_idx % FRAME_SKIP == 0:
                scale_ratio = DETECTION_WIDTH / width
                resized = cv2.resize(frame, (DETECTION_WIDTH, int(height * scale_ratio)))
                results = model.predict(resized, verbose=False, conf=0.5)

                persons = filter_large_persons(results, scale_ratio, width, height)
                scene_persons.extend(persons)

            # Motion detection
            if prev_gray is not None:
                motion_center = compute_motion_center(prev_gray, gray)
                if motion_center:
                    scene_motion.append(motion_center)

            # After enough frames, decide mode
            if analysis_counter >= ANALYSIS_FRAMES:
                if len(scene_persons) >= 2:
                    current_mode = "SPLIT"
                elif len(scene_persons) == 1:
                    current_mode = "SINGLE"
                elif len(scene_motion) > 5:
                    current_mode = "MOTION"
                else:
                    current_mode = "DOCUMENT"

        prev_gray = gray.copy()

        # ---------------- RENDERING ----------------
        processed_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        if current_mode == "SPLIT" and len(scene_persons) >= 2:
            scene_persons = sorted(scene_persons, key=lambda x: x[0])
            c1 = scene_persons[0]
            c2 = scene_persons[-1]

            def crop_person(center):
                cx, cy, _ = center
                x = max(0, min(width - split_crop_width, cx - split_crop_width // 2))
                y = max(0, min(height - split_crop_height, cy - split_crop_height // 2))
                return frame[y:y+split_crop_height, x:x+split_crop_width]

            r1 = cv2.resize(crop_person(c1), (out_w, out_h//2))
            r2 = cv2.resize(crop_person(c2), (out_w, out_h//2))
            processed_frame = np.vstack((r1, r2))

        elif current_mode == "SINGLE" and len(scene_persons) >= 1:
            cx, cy, _ = scene_persons[0]

            if locked_center is None:
                locked_center = (cx, cy)
            else:
                lx, ly = locked_center
                cx = int(SMOOTHING_FACTOR*lx + (1-SMOOTHING_FACTOR)*cx)
                cy = int(SMOOTHING_FACTOR*ly + (1-SMOOTHING_FACTOR)*cy)
                locked_center = (cx, cy)

            x = max(0, min(width - crop_width, cx - crop_width // 2))
            y = max(0, min(height - crop_height, cy - crop_height // 2))
            crop_img = frame[y:y+crop_height, x:x+crop_width]
            processed_frame = cv2.resize(crop_img, (out_w, out_h))

        elif current_mode == "MOTION" and len(scene_motion) > 0:
            cx, cy = scene_motion[-1]
            x = max(0, min(width - crop_width, cx - crop_width // 2))
            y = max(0, min(height - crop_height, cy - crop_height // 2))
            crop_img = frame[y:y+crop_height, x:x+crop_width]
            processed_frame = cv2.resize(crop_img, (out_w, out_h))

        else:
            scale = min(out_w / width, out_h / height)
            nw, nh = int(width * scale), int(height * scale)
            resized_full = cv2.resize(frame, (nw, nh))
            y_off, x_off = (out_h - nh) // 2, (out_w - nw) // 2
            processed_frame[y_off:y_off+nh, x_off:x_off+nw] = resized_full

        process.stdin.write(processed_frame.tobytes())

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames}")

        frame_idx += 1

    process.stdin.close()
    process.wait()
    cap.release()
    print("Preview generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.mp4")
    parser.add_argument("--output", default="preview_output.mp4")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Input file not found")
    else:
        generate_preview(args.input, args.output)
