import cv2
import mediapipe as mp
import numpy as np
from skin_scan import predict_skin_condition  # 🔍 NEW

# Initialize Mediapipe modules
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh()
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_point(frame, lm):
    return int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

def get_skin_color(frame, landmarks):
    p = get_point(frame, landmarks[234])  # Left cheek
    b, g, r = map(int, frame[p[1], p[0]])
    return (r, g, b)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face = face_mesh.process(rgb)
    pose_result = pose.process(rgb)
    hand_result = hands.process(rgb)

    # --------------- FACE ANALYSIS ---------------
    if face.multi_face_landmarks:
        for landmarks in face.multi_face_landmarks:
            lm = landmarks.landmark

            # Eye fatigue
            left_top = get_point(frame, lm[159])
            left_bottom = get_point(frame, lm[145])
            left_h = euclidean(left_top, left_bottom)
            left_w = euclidean(get_point(frame, lm[33]), get_point(frame, lm[133]))
            ear = left_h / (left_w + 1e-5)

            if ear < 0.18:
                cv2.putText(frame, "😴 Eye fatigue detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Skin tone estimation
            skin_color = get_skin_color(frame, lm)
            cv2.putText(frame, f"Skin tone RGB: {skin_color}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (skin_color[2], skin_color[1], skin_color[0]), 2)

            # Face symmetry
            left = get_point(frame, lm[234])
            right = get_point(frame, lm[454])
            mid = get_point(frame, lm[1])
            dist_l = euclidean(left, mid)
            dist_r = euclidean(right, mid)
            if abs(dist_l - dist_r) > 15:
                cv2.putText(frame, "⚠️ Face may be asymmetric", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


            # --------------- SKIN CONDITION SCANNER ---------------
            cheek_x, cheek_y = get_point(frame, lm[234])
            size = 50  # Reduced size to stay within bounds

            x = max(0, cheek_x - size)
            y = max(0, cheek_y - size)
            w = min(size * 2, frame.shape[1] - x)
            h = min(size * 2, frame.shape[0] - y)
            
            label = predict_skin_condition(frame, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(frame, f"Skin: {label}", (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    # --------------- POSTURE ANALYSIS ---------------
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        left_shoulder = get_point(frame, landmarks[11])
        right_shoulder = get_point(frame, landmarks[12])
        nose = get_point(frame, landmarks[0])
        left_hip = get_point(frame, landmarks[23])
        right_hip = get_point(frame, landmarks[24])

        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_diff > 20:
            cv2.putText(frame, "⚠️ Uneven Shoulders", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

        spine_top = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2])
        spine_bottom = np.array([(left_hip[0] + right_hip[0]) / 2,
                                 (left_hip[1] + right_hip[1]) / 2])
        spine_angle = np.degrees(np.arctan2(
            spine_bottom[1] - spine_top[1],
            spine_bottom[0] - spine_top[0]
        ))
        if abs(spine_angle - 90) > 15:
            cv2.putText(frame, "⚠️ Slouching Detected", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "✅ Posture Good", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        head_angle = np.degrees(np.arctan2(
            nose[1] - spine_top[1],
            nose[0] - spine_top[0]
        ))
        if abs(head_angle) > 10:
            cv2.putText(frame, "🤕 Head Tilted", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

    # --------------- HAND / NAIL ANALYSIS ---------------
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            nail_points = [lm[8], lm[12], lm[16]]
            nail_colors = []

            for pt in nail_points:
                x, y = get_point(frame, pt)
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    b, g, r = map(int, frame[y, x])
                    nail_colors.append((r, g, b))
                    cv2.circle(frame, (x, y), 4, (r, g, b), -1)

            if nail_colors:
                avg_r = int(np.mean([c[0] for c in nail_colors]))
                avg_g = int(np.mean([c[1] for c in nail_colors]))
                avg_b = int(np.mean([c[2] for c in nail_colors]))

                if avg_r < 130 and avg_g < 130 and avg_b < 130:
                    cv2.putText(frame, "⚠️ Pale Nails - Check Anemia", (30, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                elif avg_r > avg_g and avg_r > avg_b:
                    cv2.putText(frame, "✅ Nails Healthy Color", (30, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "🧴 Dry/Discolored Nails", (30, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            base = get_point(frame, lm[5])
            tip = get_point(frame, lm[8])
            dist = euclidean(base, tip)
            if dist < 40:
                cv2.putText(frame, "⚠️ Possible Finger Clubbing", (30, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --------------- DISPLAY ---------------
    cv2.imshow("🧠 Health Scanner - AR Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()