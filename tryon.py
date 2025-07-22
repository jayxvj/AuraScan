import cv2
import mediapipe as mp
import numpy as np
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import time
import os

# Load AR images
glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
cap_img = cv2.imread("cap.png", cv2.IMREAD_UNCHANGED)

# Flags
add_glasses = False
add_cap = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Initialize Vosk Model (supports English + Hindi commands)
model = Model("model/vosk-model-small-en-us-0.15/")  # change if your model differs
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def get_voice_command():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        rec = KaldiRecognizer(model, 16000)
        print("🎤 Speak your command (glasses/cap/remove/photo or chashma/topi/hatao/kheecho)...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                command = result.get("text", "")
                print(f"🗣 You said: {command}")
                return command.lower()

# Overlay helper (safe)
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]

    if h <= 0 or w <= 0:
        return background

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return background

def overlay_glasses(frame, landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    face_width = int(abs(right_eye.x - left_eye.x) * frame.shape[1]) + 40
    glasses_resized = cv2.resize(glasses_img, (face_width, int(face_width * 0.4)))
    x = int((left_eye.x + right_eye.x) / 2 * frame.shape[1]) - face_width // 2
    y = int((left_eye.y + right_eye.y) / 2 * frame.shape[0]) - 20
    return overlay_image(frame, glasses_resized, x, y)

def overlay_cap(frame, landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    face_width = int(abs(right_eye.x - left_eye.x) * frame.shape[1]) + 40
    cap_resized = cv2.resize(cap_img, (face_width, int(face_width * 0.6)))
    x = int((left_eye.x + right_eye.x) / 2 * frame.shape[1]) - face_width // 2
    y = int(landmarks[10].y * frame.shape[0]) - int(face_width * 0.6)
    return overlay_image(frame, cap_resized, x, y)

# Save photo
def take_photo(frame, label):
    save_path = "F:\JJ Projects 2025\AIML\AI Model Part -II\snaps"  # or wherever you want to save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(save_path, f"{label}_{timestamp}.png")
    cv2.imwrite(filename, frame)
    print(f"📸 Photo saved as {filename}")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            landmark_list = landmarks.landmark
            if add_glasses:
                frame = overlay_glasses(frame, landmark_list)
            if add_cap:
                frame = overlay_cap(frame, landmark_list)

    cv2.imshow("🕶️ AR Try-On", frame)
    key = cv2.waitKey(1)

    if key == ord('v'):
        command = get_voice_command()

        # English & Hindi Command Support
        if "glasses" in command or "eyeware" in command:
            add_glasses = True
        elif "cap" in command or "headwear" in command:
            add_cap = True
        elif "remove" in command or "erase" in command:
            add_cap = False
            add_glasses = False
        elif "photo" in command or "capture" in command:
            if add_glasses:
                take_photo(frame, "glasses")
            elif add_cap:
                take_photo(frame, "cap")
            else:
                take_photo(frame, "plain")  # If nothing is added

    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()