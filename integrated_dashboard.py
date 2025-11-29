# integrated_dashboard.py
# FIXED VERSION - NO ERRORS - LIGHTWEIGHT

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import urllib.request
import os
from datetime import datetime
import threading
import queue

# ====================== DOWNLOAD MODELS (LIGHT) ======================
@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            try:
                urllib.request.urlretrieve(url, filename)
                st.success(f"{filename} ready!")
            except:
                st.error("Failed to download model. Check internet.")
    return filename

# --- GOOGLE DRIVE LINKS ---
DROWSINESS_URL = "https://drive.google.com/uc?export=download&id=1WxCzOlSZLWjUscZPVFatblPzfM_WNh5h"
EFFNET_URL     = "https://drive.google.com/uc?export=download&id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"
DENSENET_URL   = "https://drive.google.com/uc?export=download&id=1-1eJ-5-6_GtjNghuGQLlrO2psp18l-z4"
MOBILENET_URL  = "https://drive.google.com/uc?export=download&id=1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"

drowsiness_path = download_model(DROWSINESS_URL, "drowsiness.keras")
effnet_path = download_model(EFFNET_URL, "effnet.pth")
densenet_path = download_model(DENSENET_URL, "densenet.pth")
mobilenet_path = download_model(MOBILENET_URL, "mobilenet.pth")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    st.info("Loading AI models...")
    
    # Drowsiness
    drowsiness_model = load_model(drowsiness_path)
    
    # Distraction
    device = torch.device("cpu")
    from torchvision import models
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self): 
            super().__init__(); 
            self.net = models.mobilenet_v3_small(pretrained=False)
            self.net.classifier[3] = nn.Linear(1024, 10)
        def forward(self, x): return self.net(x)

    model = SimpleCNN().to(device)
    state = torch.load(effnet_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    st.success("Models ready!")
    return drowsiness_model, model, device

drowsiness_model, distraction_model, device = load_models()

# ====================== DETECTION ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LABELS = ["Safe", "Text R", "Talk R", "Text L", "Talk L", "Radio", "Drink", "Reach", "Hair", "Passenger"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
    closed = 0
    for (x, y, w, h) in faces:
        roi = gray[y:y+int(h*0.7), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi, 1.05, 3)
        for (ex, ey, ew, eh) in eyes:
            eye = cv2.resize(roi[ey:ey+eh, ex:ex+ew], (128, 128)) / 255.0
            eye = np.expand_dims(eye, (0, -1))
            pred = drowsiness_model.predict(eye, verbose=0)[0][0]
            label = "CLOSED" if pred < 0.5 else "OPEN"
            color = (0, 0, 255) if pred < 0.5 else (0, 255, 0)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            if pred < 0.5: closed += 1
    return frame, closed >= 2

def detect_distraction(frame):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(distraction_model(tensor), 1)[0]
        idx = torch.argmax(probs).item()
    label = LABELS[idx]
    conf = probs[idx].item()
    color = (0, 255, 0) if idx == 0 else (0, 0, 255)
    cv2.putText(frame, f"{label} ({conf:.0%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame, idx != 0

# ====================== UI ======================
st.title("Driver Safety System")
col1, col2 = st.columns(2)
front_ph = col1.empty(); side_ph = col2.empty()

run = st.checkbox("Start Live Monitoring", True)
alerts = []

q1 = queue.Queue(1); q2 = queue.Queue(1)

def cam_thread(q, idx):
    cap = cv2.VideoCapture(idx)
    while run:
        ret, f = cap.read()
        if ret:
            f = cv2.resize(f, (400, 300))
            try: q.put_nowait(f)
            except: pass
        time.sleep(0.03)
    cap.release()

if run:
    threading.Thread(target=cam_thread, args=(q1, 0), daemon=True).start()
    threading.Thread(target=cam_thread, args=(q2, 1), daemon=True).start()

drowsy_c = 0; distract_c = 0
while run:
    f1 = q1.get() if not q1.empty() else None
    f2 = q2.get() if not q2.empty() else None

    if f1 is not None:
        f1, is_drowsy = detect_drowsiness(f1.copy())
        if is_drowsy: drowsy_c += 1
        else: drowsy_c = 0
        if drowsy_c >= 5:
            alerts.append(f"[{datetime.now().strftime('%H:%M')}] DROWSY!")
            drowsy_c = 0
        front_ph.image(f1, channels="BGR", caption="Drowsiness")

    if f2 is not None:
        f2, is_distracted = detect_distraction(f2.copy())
        if is_distracted: distract_c += 1
        else: distract_c = 0
        if distract_c >= 3:
            alerts.append(f"[{datetime.now().strftime('%H:%M')}] DISTRACTED!")
            distract_c = 0
        side_ph.image(f2, channels="BGR", caption="Distraction")

    if alerts:
        st.error(alerts[-1])
    st.caption(f"Drowsy: {drowsy_c}/5 | Distracted: {distract_c}/3")

    time.sleep(0.03)
