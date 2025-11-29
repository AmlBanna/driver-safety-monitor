# integrated_dashboard.py
# FULLY INTEGRATED DRIVER SAFETY SYSTEM
# Front Cam: Drowsiness (MRL) | Side Cam: Distraction (StateFarm) | Live Dashboard

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import json
from datetime import datetime
from collections import deque
import threading
import queue
import os

# ====================== CONFIG ======================
st.set_page_config(page_title="Driver Safety Monitor", layout="wide")
st.title("Real-Time Driver Safety Monitoring System")
st.markdown("**Front Cam: Drowsiness | Side Cam: Distraction | Live Dashboard**")

# Model Paths (Update if needed)
DROWSINESS_MODEL_PATH = "improved_cnn_best.keras"
DISTRACTION_MODELS = {
    "EfficientNet": "model_EfficientNet_B0_bs_16_best_model",
    "DenseNet": "model_Dense_bs_16_best_model",
    "MobileNet": "model_MobileNet_V3_bs_16_best_model"
}
CLASS_INDICES_PATH = "class_indices.json"

# Distraction Labels
DISTRACTION_LABELS = {
    "c0": "Safe Driving", "c1": "Texting (Right)", "c2": "Talking on Phone (Right)",
    "c3": "Texting (Left)", "c4": "Talking on Phone (Left)", "c5": "Adjusting Radio",
    "c6": "Drinking", "c7": "Reaching Behind", "c8": "Hair/Makeup", "c9": "Talking to Passenger"
}

RISK_LEVELS = {
    "c0": "SAFE", "c1": "CRITICAL", "c2": "HIGH", "c3": "CRITICAL", "c4": "HIGH",
    "c5": "MEDIUM", "c6": "MEDIUM", "c7": "HIGH", "c8": "MEDIUM", "c9": "LOW"
}

# Transform for PyTorch models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== MODEL LOADER ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Loading models on {device}...")

    # Load Drowsiness Model (TensorFlow)
    drowsiness_model = load_model(DROWSINESS_MODEL_PATH)

    # Load Distraction Models (PyTorch)
    from model_builder import EfficientNet_B0, DenseNet, MobileNet_V3
    models = {}
    for name, path in DISTRACTION_MODELS.items():
        if name == "EfficientNet":
            model = EfficientNet_B0().to(device)
        elif name == "DenseNet":
            model = DenseNet().to(device)
        else:
            model = MobileNet_V3().to(device)
        
        state = torch.load(path, map_location=device)
        state_dict = state["model"] if "model" in state else state
        state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        models[name] = model

    # Load class indices
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    st.success("All models loaded!")
    return drowsiness_model, models, device, idx_to_class

drowsiness_model, distraction_models, device, idx_to_class = load_models()

# Face & Eye Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ====================== DETECTION FUNCTIONS ======================
def detect_drowsiness(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
    if len(faces) == 0:
        return frame, False, 0.0, False

    eyes_closed = 0
    eyes_detected = 0
    for (x, y, w, h) in faces:
        roi = gray[y:y+int(h*0.7), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi, 1.05, 3, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eyes_detected += 1
            eye = roi[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (128, 128)) / 255.0
            eye = np.expand_dims(eye, axis=(0, -1))
            pred = model.predict(eye, verbose=0)[0][0]
            is_open = pred > 0.5
            label = "OPEN" if is_open else "CLOSED"
            color = (0, 255, 0) if is_open else (0, 0, 255)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            cv2.putText(frame, f"{label} {pred:.2f}", (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if not is_open:
                eyes_closed += 1

    return frame, eyes_detected > 0, eyes_closed >= 2, eyes_detected > 0

def detect_distraction(frame, models, device, idx_to_class):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = [m(input_tensor) for m in models.values()]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        probs = torch.softmax(avg_output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    
    class_code = idx_to_class.get(pred_idx, f"c{pred_idx}")
    label = DISTRACTION_LABELS.get(class_code, "Unknown")
    risk = RISK_LEVELS.get(class_code, "UNKNOWN")
    
    color = (0, 255, 0) if class_code == "c0" else (0, 0, 255)
    cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Risk: {risk} ({confidence:.1%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, class_code, label, confidence, risk

# ====================== STREAMLIT DASHBOARD ======================
col1, col2, col3 = st.columns([1, 1, 1])

front_placeholder = col1.empty()
side_placeholder = col2.empty()
dashboard_placeholder = col3.empty()

# Queues for thread-safe frame passing
front_queue = queue.Queue(maxsize=1)
side_queue = queue.Queue(maxsize=1)

# State
alerts = deque(maxlen=10)
drowsy_counter = 0
distraction_counter = 0
DROWSY_THRESH = 5
DISTRACT_THRESH = 3

def front_cam_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (480, 360))
        try:
            front_queue.put_nowait(frame)
        except:
            pass
        time.sleep(0.03)
    cap.release()

def side_cam_thread():
    cap = cv2.VideoCapture(1)  # Change to 2 if needed
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (480, 360))
        try:
            side_queue.put_nowait(frame)
        except:
            pass
        time.sleep(0.03)
    cap.release()

# Start threads
threading.Thread(target=front_cam_thread, daemon=True).start()
threading.Thread(target=side_cam_thread, daemon=True).start()

st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Live Monitoring", value=True)
show_demo = st.sidebar.checkbox("Use Demo Mode (No Camera)", value=False)

if show_demo:
    st.warning("Demo Mode: Using sample frames")

# Main Loop
frame_count = 0
start_time = time.time()

while run:
    frame_count += 1
    current_time = datetime.now().strftime("%H:%M:%S")

    # Get frames
    front_frame = front_queue.get() if not front_queue.empty() else None
    side_frame = side_queue.get() if not side_queue.empty() else None

    if show_demo:
        front_frame = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        side_frame = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)

    if front_frame is not None:
        front_frame, eyes_detected, is_drowsy, _ = detect_drowsiness(front_frame.copy(), drowsiness_model)
        if is_drowsy:
            drowsy_counter += 1
            if drowsy_counter >= DROWSY_THRESH:
                alerts.appendleft(f"[{current_time}] DROWSINESS ALERT!")
        elif eyes_detected:
            drowsy_counter = 0
        front_placeholder.image(front_frame, channels="BGR", caption="Front Camera (Drowsiness)")

    if side_frame is not None:
        side_frame, class_code, label, conf, risk = detect_distraction(side_frame.copy(), distraction_models, device, idx_to_class)
        if class_code != "c0":
            distraction_counter += 1
            if distraction_counter >= DISTRACT_THRESH and risk in ["CRITICAL", "HIGH"]:
                alerts.appendleft(f"[{current_time}] DISTRACTION: {label}")
        else:
            distraction_counter = 0
        side_placeholder.image(side_frame, channels="BGR", caption="Side Camera (Distraction)")

    # Dashboard
    with dashboard_placeholder.container():
        st.subheader("Live Alerts")
        if alerts:
            for alert in alerts:
                st.error(alert)
        else:
            st.success("All Clear")

        st.write(f"**Drowsy Frames**: {drowsy_counter}/{DROWSY_THRESH}")
        st.write(f"**Distraction Frames**: {distraction_counter}/{DISTRACT_THRESH}")
        st.write(f"**FPS**: {frame_count / (time.time() - start_time):.1f}")

    time.sleep(0.03)
    if not st.session_state.get("run", True):
        break

st.success("Monitoring Stopped")