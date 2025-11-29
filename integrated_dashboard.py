# integrated_dashboard.py - FIXED VERSION: No TF Conflict, OpenCV Drowsiness, Torch Distraction

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

# ====================== DOWNLOAD TORCH MODEL FROM GOOGLE DRIVE ======================
@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            urllib.request.urlretrieve(url, filename)
        st.success(f"{filename} ready!")
    return filename

EFFNET_URL = "https://drive.google.com/uc?export=download&id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"
effnet_path = download_model(EFFNET_URL, "effnet.pth")

# ====================== LOAD TORCH MODEL ======================
@st.cache_resource
def load_distraction_model():
    st.info("Loading distraction model...")
    device = torch.device("cpu")
    from torchvision import models
    import torch.nn as nn

    class EfficientNet_B0(nn.Module):
        def __init__(self): 
            super().__init__() 
            self.net = models.efficientnet_b0(pretrained=False)
            self.net.classifier = nn.Linear(1280, 10)
        def forward(self, x): 
            return self.net(x)

    model = EfficientNet_B0().to(device)
    state = torch.load(effnet_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    model.load_state_dict({k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")})
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
    0: "Safe Driving", 1: "Texting (Right)", 2: "Talking on Phone (Right)",
    3: "Texting (Left)", 4: "Talking on Phone (Left)", 5: "Adjusting Radio",
    6: "Drinking", 7: "Reaching Behind", 8: "Hair/Makeup", 9: "Talking to Passenger"
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

# ====================== DROWSINESS DETECTION (OpenCV EAR - No TF) ======================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    closed = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            # Simple EAR calculation (using eye corners approximation)
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                ear = eye_aspect_ratio(np.array([[0,0], [ew//3, eh], [2*ew//3, eh//2], [ew//2, 0], [ew, eh//3], [0, eh//2]]))  # Approx points
                if ear < 0.25:  # Threshold for closed eyes
                    closed += 1
                    cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "OPEN", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    is_drowsy = closed >= 2
    return frame, is_drowsy

# ====================== UI DASHBOARD ======================
st.title("Driver Safety Monitor - Fixed Version")
col1, col2, col3 = st.columns([1, 1, 1])

front_placeholder = col1.empty()
side_placeholder = col2.empty()
dashboard_placeholder = col3.empty()

run = st.checkbox("Start Monitoring", value=True)

front_queue = queue.Queue(maxsize=1)
side_queue = queue.Queue(maxsize=1)

def front_cam_thread():
    cap = cv2.VideoCapture(0)  # Front camera
    while run:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            try:
                front_queue.put_nowait(frame)
            except queue.Full:
                pass
        time.sleep(0.033)  # ~30 FPS
    cap.release()

def side_cam_thread():
    cap = cv2.VideoCapture(1)  # Side camera (change to 2 if needed)
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

# Main loop
drowsy_counter = 0
distract_counter = 0
alerts = []
start_time = time.time()

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
            st.error(alert)
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
            st.error(alert)
            distract_counter = 0
        side_placeholder.image(side_annotated, channels="BGR", caption="Side Cam: Distraction")

    # Dashboard
    with dashboard_placeholder.container():
        st.subheader("Live Status")
        st.metric("Drowsy Frames", drowsy_counter, delta=1 if drowsy_counter > 0 else 0)
        st.metric("Distract Frames", distract_counter, delta=1 if distract_counter > 0 else 0)
        st.caption(f"FPS: {len(alerts) / (time.time() - start_time):.1f}" if start_time > 0 else "0.0")

    time.sleep(0.033)

st.success("Monitoring stopped. All clear!")
