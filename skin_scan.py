import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy

# ✅ Define both custom metrics
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# ✅ Try loading model with both metrics
try:
    model = load_model(
        'skin_condition_model.h5',
        custom_objects={
            'top_2_accuracy': top_2_accuracy,
            'top_3_accuracy': top_3_accuracy
        }
    )
    print("[✅] Model loaded successfully")
except Exception as e:
    print(f"[❌] Error loading model: {e}")
    model = None

    # ✅ This is the function your health_scanner.py must import
def predict_skin_condition(frame, x, y, w, h):
    if model is None:
        print("[❌] Model not loaded")
        return "Model not loaded"

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    roi = frame[y:y+h, x:x+w]
    if roi.size == 0 or w == 0 or h == 0:
        print("[⚠️] Invalid ROI — possibly outside frame")
        return "Invalid ROI"

    try:
        roi = cv2.resize(roi, (224, 224))  # Match model input size
    except Exception as e:
        print(f"[⚠️] Resize error: {e}")
        return "Resize failed"

    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)

    try:
        prediction = model.predict(roi)[0]
        print(f"[🔍] Raw prediction vector: {prediction}")

        labels = ["Acne", "Eczema", "Mole", "Normal", "Other", "Psoriasis", "Rash"]

        if len(prediction) != len(labels):
            return f"Label mismatch: model predicts {len(prediction)} classes, but {len(labels)} labels defined."

        label = labels[np.argmax(prediction)]
        print(f"[✅] Predicted class: {label}")
        return label
    except Exception as e:
        print(f"[❌] Prediction error: {e}")
        return "Prediction error"