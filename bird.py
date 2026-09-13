#!/usr/bin/env python3
"""
Detect 'bird' class in a video using YOLO11l and save frame numbers to CSV.
Usage: python bird.py <video_path>
"""
import sys
import os
import csv
from pathlib import Path
from ultralytics import YOLO
import cv2


def detect_birds(video_path: str):
    model = YOLO("yolo11l.pt")  # YOLO11 large (~25M params)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  Size: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    bird_frame_numbers = []
    detected_count = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect with YOLO (class 14 = 'bird' in COCO)
        results = model.predict(frame, classes=[14], verbose=False)
        if len(results[0].boxes) > 0:
            bird_frame_numbers.append(frame_num)
            detected_count += 1

        frame_num += 1

        if frame_num % 30 == 0:
            print(f"  Progress: {frame_num}/{total_frames} frames", end="\r")

    cap.release()
    print(f"\n  Progress: {frame_num}/{total_frames} frames   ")

    bird_frame_numbers.sort()

    # Save/append CSV in same directory as video
    video_dir = Path(video_path).parent
    csv_path = video_dir / "bird.csv"

    existed = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not existed:
            writer.writerow(["frame_number", "video_path"])
        for fn in bird_frame_numbers:
            writer.writerow([fn, os.path.abspath(video_path)])

    print(f"\nBirds detected in {detected_count}/{total_frames} frames")
    print(f"Frames with birds: {len(bird_frame_numbers)}")
    print(f"Saved to: {csv_path}")

    return bird_frame_numbers


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bird.py <video_path>")
        sys.exit(1)
    detect_birds(sys.argv[1])
