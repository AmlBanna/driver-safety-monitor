# integrated_dashboard.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import json
from datetime import datetime
import threading
import queue
import urllib.request
import os

# ====================== DOWNLOAD MODELS ======================
@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            urllib.request.urlretrieve(url, filename)
        st.success(f"{filename} downloaded!")
    return filename

# --- GOOGLE DRIVE DIRECT LINKS ---
DROWSINESS_URL = "https://drive.google.com/uc?export=download&id=1WxCzOlSZLWjUscZPVFatblPzfM_WNh5h"
EFFNET_URL     = "https://drive.google.com/uc?export=download&id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"
DENSENET_URL   = "https://drive.google.com/uc?export=download&id=1-1eJ-5-6_GtjNghuGQLlrO2psp18l-z4"
MOBILENET_URL  = "https://drive.google.com/uc?export=download&id=1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"

drowsiness_path = download_model(DROWSINESS_URL, "improved_cnn_best.keras")
effnet_path = download_model(EFFNET_URL, "model_EfficientNet_B0_bs_16_best_model")
densenet_path = download_model(DENSENET_URL, "model_Dense_bs_16_best_model")
mobilenet_path = download_model(MOBILENET_URL, "model_MobileNet_V3_bs_16_best_model")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    st.info("Loading models...")
    
    drowsiness_model = load_model(drowsiness_path)
    
    device = torch.device("cpu")
    from torchvision import models
    import torch.nn as nn

    class EfficientNet_B0(nn.Module):
        def __init__(self): super().__init__(); self.net = models.efficientnet_b0(pretrained=False); self.net.classifier = nn.Linear(1280, 10)
        def forward(self, x): return self.net(x)
    
    class DenseNet(nn.Module):
        def __init__(self): super().__init__(); self.net = models.densenet121(pretrained=False); self.net.classifier = nn.Linear(1024, 10)
        def forward(self, x): return self.net(x)
    
    class MobileNet_V3(nn.Module):
        def __init__(self): super().__init__(); self.net = models.mobilenet_v3_small(pretrained=False); self.net.classifier[3] = nn.Linear(1024, 10)
        def forward(self, x): return self.net(x)

    effnet = EfficientNet_B0().to(device)
    densenet = DenseNet().to(device)
    mobilenet = MobileNet_V3().to(device)

    for path, model in zip([effnet_path, densenet_path, mobilenet_path], [effnet, densenet, mobilenet]):
        state = torch.load(path, map_location=device)
        state_dict = state["model"] if "model" in state else state
        model.load_state_dict(state_dict)
        model.eval()

    st.success("All models loaded!")
    return drowsiness_model, {"EfficientNet": effnet, "DenseNet": densenet, "MobileNet": mobilenet}, device

drowsiness_model, distraction_models, device = load_models()

# ====================== DETECTION ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LABELS = {f"c{i}": name for i, name in enumerate([
    "Safe Driving", "Texting (Right)", "Talking on Phone (Right)",
    "Texting (Left)", "Talking on Phone (Left)", "Adjusting Radio",
    "Drinking", "Reaching Behind", "Hair/Makeup", "Talking to Passenger"
])}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_drowsiness(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
    closed = 0
    for (x, y, w, h) in faces:
        roi = gray[y:y+int(h*0.7), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi, 1.05, 3)
        for (ex, ey, ew, eh) in eyes:
            eye = cv2.resize(roi[ey:ey+eh, ex:ex+ew], (128, 128)) / 255.0
            eye = np.expand_dims(eye, (0, -1))
            pred = model.predict(eye, verbose=0)[0][0]
            label = "CLOSED" if pred < 0.5 else "OPEN"
            color = (0, 0, 255) if pred < 0.5 else (0, 255, 0)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            if pred < 0.5: closed += 1
    return frame, closed >= 2

def detect_distraction(frame, models, device):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = [m(tensor) for m in models.values()]
        probs = torch.softmax(torch.mean(torch.stack(outputs), 0), 1)[0]
        idx = torch.argmax(probs).item()
    label = LABELS[f"c{idx}"]
    conf = probs[idx].item()
    color = (0, 255, 0) if idx == 0 else (0, 0, 255)
    cv2.putText(frame, f"{label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame, idx == 0

# ====================== UI ======================
st.title("Driver Safety Monitor")
col1, col2, col3 = st.columns(3)
front_ph = col1.empty(); side_ph = col2.empty(); dash_ph = col3.empty()

run = st.checkbox("Start Monitoring", True)
alerts = []

front_q = queue.Queue(1); side_q = queue.Queue(1)

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
    threading.Thread(target=cam_thread, args=(front_q, 0), daemon=True).start()
    threading.Thread(target=cam_thread, args=(side_q, 1), daemon=True).start()

drowsy_count = 0; distract_count = 0
while run:
    front = front_q.get() if not front_q.empty() else None
    side = side_q.get() if not side_q.empty() else None

    if front is not None:
        front, is_drowsy = detect_drowsiness(front.copy(), drowsiness_model)
        if is_drowsy: drowsy_count += 1
        else: drowsy_count = 0
        if drowsy_count >= 5:
            alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] DROWSY ALERT!")
            drowsy_count = 0
        front_ph.image(front, channels="BGR", caption="Front Cam (Drowsiness)")

    if side is not None:
        side, is_safe = detect_distraction(side.copy(), distraction_models, device)
        if not is_safe: distract_count += 1
        else: distract_count = 0
        if distract_count >= 3:
            alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] DISTRACTION ALERT!")
            distract_count = 0
        side_ph.image(side, channels="BGR", caption="Side Cam (Distraction)")

    with dash_ph.container():
        st.subheader("Alerts")
        for a in alerts[-5:]: st.error(a)
        st.write(f"Drowsy: {drowsy_count}/5 | Distracted: {distract_count}/3")

    time.sleep(0.03)
