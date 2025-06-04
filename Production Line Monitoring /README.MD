# 🧠 Agentic AI for Conveyor Monitoring

This project demonstrates an **agentic AI approach** for visual inspection in manufacturing, using live video of a conveyor system. The system learns a **reference visual signature** from normal operation and flags deviations using **CLIP embeddings + cosine similarity**.

---

## 🔧 System Setup

- 🎥 Input: Webcam/video of parts flowing on a conveyor
- 🤖 Engine: CLIP model (OpenAI) for image-to-vector encoding
- 📐 Method: Cosine similarity vs reference embedding
- 📉 Output: Pass/Fail detection + plot visualization

---

## 🧪 Use Case: Visual Process Monitoring

Three conveyor runs were tested:

| Run | Description               | Cosine Similarity | Status          |
|-----|---------------------------|-------------------|------------------|
| 1   | Normal (reference)        | —                 | ✅ Baseline      |
| 2   | Empty conveyor            | ~0.8198           | ❗ Anomaly       |
| 3   | Spaced parts              | ~0.8996           | ❗ Near-threshold|

---




