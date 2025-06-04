import cv2
import os
import glob

def view_image_folder(image_folder):
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + 
                         glob.glob(os.path.join(image_folder, "*.png")))

    if not image_paths:
        print("❌ No images found in:", image_folder)
        return

    idx = 0
    print(f"🖼 Viewing {len(image_paths)} images in: {image_folder}")
    print("➡ Use LEFT and RIGHT arrow keys to navigate. Press 'q' or ESC to exit.")

    while True:
        img = cv2.imread(image_paths[idx])
        if img is None:
            print(f"⚠️ Failed to load: {image_paths[idx]}")
            break

        cv2.imshow("Image Viewer", img)
        key = cv2.waitKey(0)

        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == 81:  # Left arrow
            idx = max(0, idx - 1)
        elif key == 83:  # Right arrow
            idx = min(len(image_paths) - 1, idx + 1)

    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    view_image_folder("data/live_capture/conveyor_run_03")
