# run_signature_comparison.py

import os
import numpy as np
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt

def compute_visual_signature(folder_path, model, preprocess, device):
    embeddings = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, fname)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embeddings.append(embedding.cpu().numpy())
    if embeddings:
        return np.mean(embeddings, axis=0)
    return None

def compare_runs(reference_folder, test_folders, threshold=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"\n🔍 Computing reference signature from: {reference_folder}")
    reference_embedding = compute_visual_signature(reference_folder, model, preprocess, device)

    if reference_embedding is None:
        print("❌ No reference embedding computed.")
        return

    run_labels = []
    similarities = []

    for label, test_folder in test_folders.items():
        print(f"\n📁 Comparing with {label} ({test_folder})")
        test_embedding = compute_visual_signature(test_folder, model, preprocess, device)

        if test_embedding is None:
            print("⚠️ Skipped (no images found).")
            continue

        cosine_similarity = np.dot(reference_embedding, test_embedding.T) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(test_embedding)
        )
        cosine_similarity = cosine_similarity.item()

        run_labels.append(label)
        similarities.append(cosine_similarity)

        print(f"📈 Cosine Similarity = {cosine_similarity:.4f}")
        if cosine_similarity < threshold:
            print("❗ Potential anomaly detected!")
        else:
            print("✅ Similar to reference.")

    # Plotting
    plt.figure(figsize=(8, 5))
    bar_colors = ['green' if s >= threshold else 'red' for s in similarities]
    plt.bar(run_labels, similarities, color=bar_colors)
    plt.axhline(threshold, color='blue', linestyle='--', label=f'Anomaly Threshold ({threshold})')
    plt.title("Cosine Similarity Comparison Across Conveyor Runs")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = "conveyor_similarity_plot.png"
    plt.savefig(plot_path)
    print(f"📊 Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    reference_folder = "data/live_capture/conveyor_run_01"
    test_folders = {
        "Run 2 - No Parts": "data/live_capture/conveyor_run_02",
        "Run 3 - Spaced Parts": "data/live_capture/conveyor_run_03"
    }

    compare_runs(reference_folder, test_folders)
