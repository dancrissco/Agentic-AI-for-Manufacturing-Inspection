# webcam_frame_extractor.py

import cv2
import os
import time

def extract_from_webcam(output_folder, capture_duration=10, frame_interval=1, resize_dim=(640, 480), camera_index=0):
    """
    Capture frames from a webcam at specified intervals for a set duration.

    Parameters:
    - output_folder: folder to save frames
    - capture_duration: total time to capture in seconds
    - frame_interval: interval in seconds between captures
    - resize_dim: (width, height) of saved frames
    - camera_index: index of webcam (0 is usually default)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        return

    print(f"🎥 Capturing from webcam {camera_index} for {capture_duration}s...")
    print(f"📁 Saving frames every {frame_interval}s to: {output_folder}")

    start_time = time.time()
    saved_count = 0

    while (time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

        frame_resized = cv2.resize(frame, resize_dim)
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame_resized)
        print(f"✅ Saved: {filename}")
        saved_count += 1
        time.sleep(frame_interval)

    cap.release()
    print("📷 Done capturing.")

# Example usage:
if __name__ == "__main__":
    extract_from_webcam(
        output_folder="data/live_capture/conveyor_run_03",
        capture_duration=15,    # Capture for 15 seconds
        frame_interval=1,       # Save one frame every 1 second
        resize_dim=(640, 480),
        camera_index=0          # Default webcam
    )
