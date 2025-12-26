import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading
import time
import os

# -----------------------------
# Fatigue timing control
# -----------------------------
last_fatigue_update = time.time()
FATIGUE_UPDATE_INTERVAL = 2  # seconds


# -----------------------------
# Utility functions
# -----------------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def play_alert():
    sound_path = os.path.join(os.path.dirname(__file__), "alert.wav")
    playsound(sound_path)


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -----------------------------
# Thresholds & counters
# -----------------------------
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20
counter = 0
alarm_on = False

# Blink & fatigue variables
blink_history = []
blink_start = None
EYE_CLOSED = False
fatigue_score = 0

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye, right_eye = [], []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            # -----------------------------
            # Blink detection
            # -----------------------------
            if ear < EAR_THRESHOLD and not EYE_CLOSED:
                EYE_CLOSED = True
                blink_start = current_time

            elif ear >= EAR_THRESHOLD and EYE_CLOSED:
                EYE_CLOSED = False
                blink_duration = current_time - blink_start
                if blink_duration < 0.5:
                    blink_history.append(current_time)

            blink_history = [b for b in blink_history if current_time - b <= 60]
            blink_rate = len(blink_history)

            # -----------------------------
            # Fatigue score (time-based)
            # -----------------------------
            if current_time - last_fatigue_update >= FATIGUE_UPDATE_INTERVAL:
                if blink_rate < 10:
                    fatigue_score += 2
                elif blink_rate > 20:
                    fatigue_score -= 1

                if ear < EAR_THRESHOLD:
                    fatigue_score += 2
                else:
                    fatigue_score -= 1

                fatigue_score = max(0, min(100, fatigue_score))
                last_fatigue_update = current_time

            # -----------------------------
            # Status
            # -----------------------------
            status = "DROWSY" if fatigue_score > 60 else "FOCUSED"
            status_color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)

            cv2.putText(frame, f"Status: {status}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # -----------------------------
            # Drowsiness alert
            # -----------------------------
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CLOSED_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alert, daemon=True).start()

                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (100, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
            else:
                counter = 0
                alarm_on = False

            # -----------------------------
            # Display info
            # -----------------------------
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Blink Rate: {blink_rate}/min", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Fatigue Score: {fatigue_score}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
