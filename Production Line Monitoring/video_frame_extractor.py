# video_frame_extractor.py

import cv2
import os

def extract_video_frames(video_path, output_folder, frame_interval=1, target_fps=10, resize_dim=(640, 480)):
    """
    Extract frames from a video at specified intervals and save them as images.

    Parameters:
    - video_path: path to the video file
    - output_folder: folder to save extracted frames
    - frame_interval: interval in seconds between frames to extract
    - target_fps: desired frames per second to simulate
    - resize_dim: desired image resolution (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(actual_fps * frame_interval)
    frame_count = 0
    saved_count = 0

    print(f"🎞 Extracting frames every {frame_interval}s from: {video_path}")
    print(f"📁 Saving to: {output_folder}")
    print(f"🎯 Target resolution: {resize_dim}, Target FPS: {target_fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            frame_resized = cv2.resize(frame, resize_dim)
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame_resized)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done. {saved_count} frames saved.")

# Example usage:
if __name__ == "__main__":
    extract_video_frames(
        video_path="data/good_runs/good_run_01.mp4",
        output_folder="data/extracted/good_01",
        frame_interval=0.5,       # extract a frame every 0.5 seconds
        target_fps=10,
        resize_dim=(640, 480)
    )
