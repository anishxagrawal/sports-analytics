import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/videos/match.mp4"
OUTPUT_DIR = Path("tmp/raw_frames/match1")
TARGET_FPS = 5
DIFF_THRESHOLD = 25  # controls similarity filtering

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(round(video_fps / TARGET_FPS)))

prev_gray = None
frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            save = True
        else:
            diff = cv2.absdiff(gray, prev_gray)
            score = diff.mean()
            save = score > DIFF_THRESHOLD

        if save:
            out_path = OUTPUT_DIR / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            prev_gray = gray
            saved_idx += 1

    frame_idx += 1

cap.release()

print(f"Saved {saved_idx} diverse frames to {OUTPUT_DIR}")
