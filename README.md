# 🧠 AuraScan

This repository contains an **AI-powered AR Health Scanner and Virtual Try-On System**.  
It combines **computer vision, deep learning, and augmented reality** to detect human health cues, recognize objects, and overlay virtual items in real-time.

---

## 📂 Project Structure

├── model/ # Pre-trained & custom-trained ML/DL models
├── snaps/ # Example screenshots & output snapshots
├── tryon resources/ # Glasses / accessories / virtual objects for AR TryOn
├── error_handling.py # Centralized error handling & logging
├── health_scanner.py # AR Health Scanner (face, posture, nails, skin)
├── obj_recog.py # Object recognition using YOLO / vision models
├── skin_scan.py # Skin condition prediction (CNN)
├── train_labels.py # Dataset training & label mapping
├── tryon.py # AR Virtual Try-On engine
└── README.md # Project documentation


---

## 🚀 Features

- **Real-time AR Health Scanner**  
  - Eye fatigue detection  
  - Posture analysis (spine, shoulders, head tilt)  
  - Nail & skin health checks  

- **Skin Condition Classification**  
  - Uses a CNN trained on medical skin datasets  
  - Detects and classifies conditions with accuracy reports  

- **Object Recognition**  
  - YOLOv8 / Mediapipe-based real-time recognition  
  - Identifies daily objects, living/non-living items  

- **Virtual Try-On (AR)**  
  - Overlay glasses / accessories on user’s face  
  - Uses `tryon resources/` assets  

- **Robust Error Handling**  
  - Centralized logging in `error_handling.py`  

- **Training Utilities**  
  - `train_labels.py` handles dataset preprocessing & label mappings  

---

## 🛠️ Installation

### Requirements
- Python 3.9+
- OpenCV
- Mediapipe
- TensorFlow / Keras
- NumPy
- Ultralytics YOLO (optional for `obj_recog.py`)

### Install dependencies
```bash
pip install -r requirements.txt
