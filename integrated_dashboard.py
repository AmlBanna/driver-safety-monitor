# integrated_dashboard.py - FINAL FIXED: No packages.txt, Compatible Requirements

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time
from datetime import datetime
import threading
import queue
import urllib.request
import os

# ====================== DOWNLOAD MODEL ======================
@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}... (1-2 min)"):
            urllib.request.urlretrieve(url, filename)
        st.success(f"{filename} ready!")
    return filename

EFFNET_URL = "https://drive.google.com/uc?export=download&id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"
effnet_path = download_model(EFFNET_URL, "effnet.pth")

# ====================== LOAD DISTRACTION MODEL ======================
@st.cache_resource
def load_distraction_model():
    st.info("Loading distraction model...")
    device = torch.device("cpu")
    from torchvision import models
    import torch.nn as nn

    class EfficientNet_B0(nn.Module):
        def __init__(self): 
            super().__init__(); 
            self.net = models.efficientnet_b0(pretrained=False); 
            self.net.classifier = nn.Linear(1280, 10)
        def forward(self, x): 
            return self.net(x)

    model = EfficientNet_B0().to(device)
    state = torch.load(effnet_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    # Fix common key mismatches
    fixed_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("net.", "").replace("module.", "")
        fixed_state[new_k] = v
    model.load_state_dict(fixed_state)
    model.eval()
    st.success("Distraction model loaded!")
    return model, device

distraction_model, device = load_distraction_model()

# ====================== DISTRACTION DETECTION ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LABELS = {
    0: "Safe Driving", 1: "Texting (Right)", 2: "Talking (Right)",
    3: "Texting (Left)", 4: "Talking (Left)", 5: "Radio",
    6: "Drinking", 7: "Reaching", 8: "Hair/Makeup", 9: "Passenger"
}

def detect_distraction(frame, model, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        idx = torch.argmax(probs).item()
        conf = probs[idx].item()
    label = LABELS[idx]
    color = (0, 255, 0) if idx == 0 else (0, 0, 255)
    cv2.putText(frame, f"{label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, idx != 0, label

# ====================== DROWSINESS (OpenCV - Simple & Fast) ======================
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
    closed = 0
    eyes_detected = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
        eyes_detected += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            # Simple closed eye detection using variance (low = closed)
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                variance = np.var(eye_region)
                if variance < 50:  # Threshold for closed eyes
                    closed += 1
                    cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "OPEN", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    is_drowsy = (closed >= 2) and (eyes_detected > 0)
    return frame, is_drowsy

# ====================== UI DASHBOARD ======================
st.title("Driver Safety Monitor - Live & Safe!")
col1, col2, col3 = st.columns([1, 1, 1])

front_placeholder = col1.empty()
side_placeholder = col2.empty()
dashboard_placeholder = col3.empty()

run = st.checkbox("Start Monitoring", value=True)

front_queue = queue.Queue(maxsize=1)
side_queue = queue.Queue(maxsize=1)

def front_cam_thread():
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            try:
                front_queue.put_nowait(frame)
            except queue.Full:
                pass
        time.sleep(0.033)
    cap.release()

def side_cam_thread():
    cap = cv2.VideoCapture(1)
    while run:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            try:
                side_queue.put_nowait(frame)
            except queue.Full:
                pass
        time.sleep(0.033)
    cap.release()

if run:
    threading.Thread(target=front_cam_thread, daemon=True).start()
    threading.Thread(target=side_cam_thread, daemon=True).start()

drowsy_counter = 0
distract_counter = 0
alerts = []

while run:
    front_frame = front_queue.get() if not front_queue.empty() else None
    side_frame = side_queue.get() if not side_queue.empty() else None

    if front_frame is not None:
        front_annotated, is_drowsy = detect_drowsiness(front_frame.copy())
        if is_drowsy:
            drowsy_counter += 1
        else:
            drowsy_counter = 0
        if drowsy_counter >= 5:
            alert = f"[{datetime.now().strftime('%H:%M:%S')}] DROWSINESS ALERT!"
            alerts.append(alert)
            drowsy_counter = 0
        front_placeholder.image(front_annotated, channels="BGR", caption="Front Cam: Drowsiness")

    if side_frame is not None:
        side_annotated, is_distracted, label = detect_distraction(side_frame.copy(), distraction_model, device)
        if is_distracted:
            distract_counter += 1
        else:
            distract_counter = 0
        if distract_counter >= 3:
            alert = f"[{datetime.now().strftime('%H:%M:%S')}] DISTRACTION: {label}!"
            alerts.append(alert)
            distract_counter = 0
        side_placeholder.image(side_annotated, channels="BGR", caption="Side Cam: Distraction")

    with dashboard_placeholder.container():
        st.subheader("Live Alerts & Stats")
        if alerts:
            for a in alerts[-3:]:
                st.error(a)
        else:
            st.success("All Clear - Safe Driving!")
        col_a, col_b = st.columns(2)
        col_a.metric("Drowsy Frames", drowsy_counter)
        col_b.metric("Distract Frames", distract_counter)

    time.sleep(0.033)

st.success("Monitoring stopped. Drive safe!")
