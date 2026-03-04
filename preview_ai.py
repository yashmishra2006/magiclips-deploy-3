import cv2
import numpy as np
import argparse
import os
import torch
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIG (Synchronized with video_processing.py)
# ============================================================
YOLO_FACE_MODEL_PATH = "yolov8n.pt"
DETECTION_WIDTH = 640
FRAME_SKIP = 3
SCENE_THRESHOLD = 0.5
FACE_PADDING_RATIO = 0.15

# ============================================================
# PREVIEW LOGIC
# ============================================================

def preview_video(video_path):
    # Load Model
    print(f"Loading model: {YOLO_FACE_MODEL_PATH}...")
    model = YOLO(YOLO_FACE_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 9:16 crop setup (Single Mode)
    target_aspect = 9 / 16
    crop_height = height
    crop_width = int(crop_height * target_aspect)
    if crop_width > width:
        crop_width = width
        crop_height = int(width / target_aspect)

    # 9:8 crop setup (Split Mode)
    split_aspect = 9 / 8
    split_crop_height = height
    split_crop_width = int(split_crop_height * split_aspect)
    if split_crop_width > width:
        split_crop_width = width
        split_crop_height = int(width / split_aspect)

    # Output window size (Display)
    out_w, out_h = 720, 1280

    history_buffer = deque(maxlen=10)
    prev_frame_for_scene = None
    last_faces = []
    frame_idx = 0
    
    # Smoothing state
    smoothed_center = None  # For single mode
    smoothed_c1 = None      # For split mode top
    smoothed_c2 = None      # For split mode bottom
    prev_mode = None        # To detect mode switches
    ALPHA = 0.2             # Smoothing factor (lower = smoother but slower)

    print("\n--- [MAGICLIPS AI PREVIEW] ---")
    print("Controls:")
    print(" - 'q': Quit Preview")
    print(" - 'p': Pause/Resume")
    print(" - Any key: Step through frames (if paused)")
    print("\nVisual Guide:")
    print(" - Green Circle: Raw Face Detection")
    print(" - White Dot: Stabilized Face Positions (Buffer)")
    print(" - Blue Box/Red Box: Crop Region (Active)")
    print(" - Red Flash: Scene Cut Detected")
    print("-----------------------------\n")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
        
        # We need a copy for visualization to avoid drawing on the "raw" frame used for processing logic
        viz_frame = frame.copy()
        
        scene_cut = False
        # Scene Detection
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
                history_buffer.clear()
                # Reset smoothing on scene cut
                smoothed_center = None
                smoothed_c1 = None
                smoothed_c2 = None

        prev_frame_for_scene = frame.copy() if frame_idx % 2 == 0 else prev_frame_for_scene

        # Face Detection
        if frame_idx % FRAME_SKIP == 0 or scene_cut:
            scale_ratio = DETECTION_WIDTH / width
            resized = cv2.resize(frame, (DETECTION_WIDTH, int(height * scale_ratio)))
            results = model.predict(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), verbose=False, conf=0.5)
            
            current_faces = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Scale back to original coordinates
                    rx1, ry1, rx2, ry2 = x1/scale_ratio, y1/scale_ratio, x2/scale_ratio, y2/scale_ratio
                    cx = int((rx1+rx2)/2)
                    cy = int((ry1+ry2)/2)
                    current_faces.append((cx, cy))
                    
                    # DRAW RAW DETECTION (GREEN CIRCLE)
                    cv2.rectangle(viz_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 2)
                    cv2.circle(viz_frame, (cx, cy), 5, (0, 255, 0), -1)
            
            last_faces = current_faces
        else:
            current_faces = last_faces

        history_buffer.append(current_faces)
        all_faces = [f for faces in history_buffer for f in faces]

        # Mode Selection Logic (Based on current frame for responsiveness)
        current_split_mode = False
        if len(current_faces) >= 2:
            sorted_curr = sorted(current_faces, key=lambda p: p[0])
            if abs(sorted_curr[-1][0] - sorted_curr[0][0]) > width * 0.15:
                current_split_mode = True
        elif not current_faces and len(all_faces) >= 2:
            # Fallback to buffer if current frame is empty but buffer has faces
            sorted_buff = sorted(all_faces, key=lambda p: p[0])
            if abs(sorted_buff[-1][0] - sorted_buff[0][0]) > width * 0.15:
                current_split_mode = True

        # Detect Mode Switch to force reset smoothing (SNAP instead of SLIDE)
        mode_id = "SPLIT" if (current_split_mode and len(all_faces) >= 2) else ("SINGLE" if all_faces else "DOC")
        if mode_id != prev_mode:
            smoothed_center = None
            smoothed_c1 = None
            smoothed_c2 = None
        prev_mode = mode_id

        # Draw Stabilized Faces (White Dots)
        for fx, fy in all_faces:
            cv2.circle(viz_frame, (fx, fy), 2, (255, 255, 255), -1)

        # Rendering and Crop Visualization
        processed_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        if mode_id == "SPLIT":
            mode_text = "SPLIT MODE"
            faces_sorted = sorted(all_faces, key=lambda p: p[0])
            c1_raw = np.array(faces_sorted[0])
            c2_raw = np.array(faces_sorted[-1])

            # Apply EMA Smoothing for Split Mode
            if smoothed_c1 is None: smoothed_c1 = c1_raw
            else: smoothed_c1 = ALPHA * c1_raw + (1 - ALPHA) * smoothed_c1
            
            if smoothed_c2 is None: smoothed_c2 = c2_raw
            else: smoothed_c2 = ALPHA * c2_raw + (1 - ALPHA) * smoothed_c2

            def get_crop_and_draw(center_smoothed, color):
                cx, cy = int(center_smoothed[0]), int(center_smoothed[1])
                x = max(0, min(width - split_crop_width, cx - split_crop_width // 2))
                y = max(0, min(height - split_crop_height, cy - split_crop_height // 2))
                # Draw Blue Crop Box
                cv2.rectangle(viz_frame, (x, y), (x + split_crop_width, y + split_crop_height), color, 3)
                return frame[y:y + split_crop_height, x:x + split_crop_width]

            crop1 = get_crop_and_draw(smoothed_c1, (255, 100, 0)) # Blue
            crop2 = get_crop_and_draw(smoothed_c2, (255, 200, 0)) # Lighter blue

            r1 = cv2.resize(crop1, (out_w, out_h // 2))
            r2 = cv2.resize(crop2, (out_w, out_h // 2))
            processed_frame = np.vstack((r1, r2))
            
        elif mode_id == "SINGLE":
            mode_text = "SINGLE FOCUS"
            avg_x = np.mean([p[0] for p in all_faces])
            avg_y = np.mean([p[1] for p in all_faces])
            center_raw = np.array([avg_x, avg_y])

            # Apply EMA Smoothing for Single Mode
            if smoothed_center is None: smoothed_center = center_raw
            else: smoothed_center = ALPHA * center_raw + (1 - ALPHA) * smoothed_center

            sx, sy = int(smoothed_center[0]), int(smoothed_center[1])
            x = max(0, min(width - crop_width, sx - crop_width // 2))
            y = max(0, min(height - crop_height, sy - crop_height // 2))
            
            # Draw Single Crop Box
            cv2.rectangle(viz_frame, (x, y), (x + crop_width, y + crop_height), (255, 50, 50), 3)
            
            crop_img = frame[y:y + crop_height, x:x + crop_width]
            processed_frame = cv2.resize(crop_img, (out_w, out_h))
        else:
            mode_text = "DOCUMENT MODE"
            scale = min(out_w / width, out_h / height)
            nw, nh = int(width * scale), int(height * scale)
            resized_full = cv2.resize(frame, (nw, nh))
            y_off, x_off = (out_h - nh) // 2, (out_w - nw) // 2
            processed_frame[y_off:y_off+nh, x_off:x_off+nw] = resized_full

        if scene_cut:
            cv2.rectangle(viz_frame, (0, 0), (width, height), (0, 0, 255), 15)
            cv2.putText(viz_frame, "SCENE CUT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Labels
        cv2.putText(viz_frame, f"MODE: {mode_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viz_frame, f"FRAME: {frame_idx}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update display
        display_scale = 800 / width
        display_viz = cv2.resize(viz_frame, (800, int(height * display_scale)))
        display_proc = cv2.resize(processed_frame, (int(out_w * 0.5), int(out_h * 0.5)))

        cv2.imshow("1. AI INPUT ANALYSIS (SOURCE)", display_viz)
        cv2.imshow("2. AI OUTPUT PREVIEW (RESULT)", display_proc)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif paused and key != 255: # If paused and any key pressed
            # Step forward
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
        
        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to video file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
    else:
        preview_video(args.input)
