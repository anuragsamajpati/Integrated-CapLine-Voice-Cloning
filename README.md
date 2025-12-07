# Integrated CapLine + AKF Voice Cloning

This repository combines:

- **CapLine_TazzoX** – real-time speech translation dashboard (Whisper + M2M100).
- **Voice Cloning** – Adaptive Kalman Filter (AKF) based voice cloning.

## How to run

```bash
# from repo root
python3.11 -m venv venv311
source venv311/bin/activate

pip install --upgrade pip
pip install -r backend/requirements_integrated.txt
pip install openai-whisper sentencepiece flask flask-cors flask-socketio

cd CapLine_TazzoX/backend
python app.py
```

Then open:

```text
http://localhost:5000/dashboard
```

Speak into the mic: translated audio is passed through the AKF voice cloning pipeline.
