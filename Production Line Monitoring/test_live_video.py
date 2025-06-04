import cv2

def live_webcam_test(camera_index=0, resize_dim=(640, 480)):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print(f"📷 Webcam {camera_index} opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        frame_resized = cv2.resize(frame, resize_dim)
        cv2.imshow("Live Webcam Feed", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📴 Webcam released. Window closed.")

# Example usage
if __name__ == "__main__":
    live_webcam_test(camera_index=0, resize_dim=(640, 480))
