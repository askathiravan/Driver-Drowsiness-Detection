# Driver Drowsiness Detection System 🚗💤

A real-time computer vision system that detects driver fatigue using
eye aspect ratio (EAR), blink rate analysis, and a fatigue scoring mechanism.
The system triggers an audio alert when drowsiness is detected.

---

## 🔍 Features
- Real-time webcam-based face and eye detection
- Eye Aspect Ratio (EAR) based eye-closure detection
- Blink rate calculation (blinks per minute)
- Fatigue score (0–100) to represent driver alertness
- Audio alert using multithreading
- Live status display: FOCUSED / DROWSY

---

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

---

## ▶ How to Run the Project

### Option 1: Run from Source (Recommended for Developers)

```bash
pip install -r requirements.txt
python driver_drowsiness.py
