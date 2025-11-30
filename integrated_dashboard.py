# integrated_dashboard.py - FULLY UPGRADED: 3 MODELS + LIVE + VIDEO + PRO UI
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time
from datetime import datetime
import os
import gdown
import zipfile
import io
import base64
from threading import Thread
import queue

# ====================== CONFIG ======================
st.set_page_config(page_title="🚗 Advanced Driver Safety Monitor", layout="wide", initial_sidebar_state="expanded")
DEMO_MODE = st.sidebar.checkbox("DEMO MODE (Streamlit Cloud)", value=True)
ENABLE_AUDIO_ALERT = st.sidebar.checkbox("Enable Sound Alerts", value=True)
ALERT_THRESHOLD = 0.7  # Confidence threshold

# ====================== MODEL DOWNLOADS ======================
@st.cache_resource
def download_models():
    models = {}
    
    # --- Model 1: Original Distraction (effnet.pth) ---
    eff_path = "effnet.pth"
    if not os.path.exists(eff_path):
        try:
            with st.spinner("Downloading Distraction Model (effnet.pth)..."):
                gdown.download("https://drive.google.com/uc?id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM", eff_path, quiet=False)
        except:
            st.warning("Could not download effnet.pth")
            eff_path = None
    models['distraction'] = eff_path

    # --- Model 2: Kaggle Driver Behavior (driver_behavior_model.pth) ---
    kaggle_zip = "driver_behavior_model.zip"
    kaggle_pth = "driver_behavior_model.pth"
    if not os.path.exists(kaggle_pth):
        try:
            with st.spinner("Downloading Kaggle Driver Behavior Model..."):
                gdown.download("https://drive.google.com/uc?id=1xYZxSampleKaggleID", kaggle_zip, quiet=False)  # غير الـ ID
                with zipfile.ZipFile(kaggle_zip) as z:
                    z.extractall(".")
            os.remove(kaggle_zip)
        except Exception as e:
            st.warning(f"Kaggle model failed: {e}")
            kaggle_pth = None
    models['behavior'] = kaggle_pth

    # --- Model 3: Custom CNN Drowsiness (drowsiness_cnn.pth) ---
    cnn_path = "drowsiness_cnn.pth"
    if not os.path.exists(cnn_path):
        st.info("No CNN model found. Using Haar Cascade fallback.")
        cnn_path = None
    models['drowsiness_cnn'] = cnn_path

    return models

models = download_models()

# ====================== LOAD MODELS ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(path, model_class):
    if not path or not os.path.exists(path):
        return None
    try:
        model = model_class().to(device)
        state = torch.load(path, map_location=device)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model load error ({path}): {e}")
        return None

# Distraction Model (EfficientNet)
class DistractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        import torch.nn as nn
        self.net = models.efficientnet_b0(pretrained=False)
        self.net.classifier = nn.Linear(1280, 10)
    def forward(self, x): return self.net(x)

distraction_model = load_model(models['distraction'], DistractionModel)

# Behavior Model (from Kaggle - assume ResNet or similar)
class BehaviorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        import torch.nn as nn
        self.net = models.resnet50(pretrained=False)
        self.net.fc = nn.Linear(2048, 10)
    def forward(self, x): return self.net(x)

behavior_model = load_model(models['behavior'], BehaviorModel)

# Drowsiness CNN
class DrowsinessCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torch.nn as nn
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): 
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

drowsiness_cnn = load_model(models['drowsiness_cnn'], DrowsinessCNN)

# ====================== TRANSFORMS ======================
distraction_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

behavior_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eye_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====================== LABELS ======================
DISTRACTION_LABELS = ["Safe", "Text R", "Talk R", "Text L", "Talk L", "Radio", "Drink", "Reach", "Hair", "Passenger"]
BEHAVIOR_LABELS = ["Safe", "Phone", "Eat", "Smoke", "Turn", "Mirror", "Seatbelt", "Child", "Pet", "Unknown"]
DROWSINESS_STATES = ["Alert", "Drowsy"]

# ====================== DETECTION FUNCTIONS ======================
def detect_distraction(frame, model):
    if not model: return frame, False, "N/A", 0.0
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = distraction_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(tensor), 1)[0]
            idx = prob.argmax().item()
            conf = prob[idx].item()
        label = DISTRACTION_LABELS[idx]
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.putText(frame, f"{label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame, idx != 0, label, conf
    except: return frame, False, "Error", 0.0

def detect_behavior(frame, model):
    if not model: return frame, False, "N/A", 0.0
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = behavior_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(tensor), 1)[0]
            idx = prob.argmax().item()
            conf = prob[idx].item()
        label = BEHAVIOR_LABELS[idx]
        color = (0, 255, 0) if idx == 0 else (255, 165, 0)
        cv2.putText(frame, f"Beh: {label} ({conf:.1%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, idx != 0, label, conf
    except: return frame, False, "Error", 0.0

def detect_drowsiness_cnn(frame, model):
    if not model: return frame, False, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0: return frame, False, 0.0
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 3)
    closed_count = 0
    for (ex, ey, ew, eh) in eyes:
        eye = roi[ey:ey+eh, ex:ex+ew]
        if eye.size == 0: continue
        eye_pil = Image.fromarray(eye)
        tensor = eye_transform(eye_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(tensor).item()
        status = "CLOSED" if prob < 0.5 else "OPEN"
        color = (0, 0, 255) if status == "CLOSED" else (0, 255, 0)
        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
        cv2.putText(frame, status, (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if status == "CLOSED": closed_count += 1
    is_drowsy = closed_count >= 2
    conf = 1.0 - (closed_count / max(len(eyes), 1))
    cv2.putText(frame, f"Drowsy: {conf:.1%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if is_drowsy else (0,255,0), 2)
    return frame, is_drowsy, conf
    except: return frame, False, 0.0

def detect_drowsiness_haar(frame):
    # Fallback
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        closed = 0
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi, 1.05, 3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                eye = roi[ey:ey+eh, ex:ex+ew]
                var = np.var(eye) if eye.size > 0 else 100
                if var < 50:
                    closed += 1
                    cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        is_drowsy = closed >= 2
        return frame, is_drowsy
    except: return frame, False

# ====================== ALERT SYSTEM ======================
def trigger_alert(level):
    if not ENABLE_AUDIO_ALERT: return
    if level == "danger":
        st.warning("🚨 **DANGER: High Risk!**")
        st.markdown('<audio autoplay><source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJ...</audio>', unsafe_allow_html=True)
    elif level == "warning":
        st.warning("⚠️ **Warning: Distraction Detected**")

# ====================== PROCESS FRAME ======================
def process_frame(front_frame, side_frame):
    # Drowsiness
    if drowsiness_cnn:
        front_frame, drow_detected, drow_conf = detect_drowsiness_cnn(front_frame, drowsiness_cnn)
    else:
        front_frame, drow_detected = detect_drowsiness_haar(front_frame)
        drow_conf = 1.0 if drow_detected else 0.0

    # Distraction + Behavior
    side_frame, dist_detected, dist_label, dist_conf = detect_distraction(side_frame, distraction_model)
    side_frame, beh_detected, beh_label, beh_conf = detect_behavior(side_frame, behavior_model)

    # Final Status
    risk_score = (drow_conf if drow_detected else 0) + (dist_conf if dist_detected else 0) + (beh_conf if beh_detected else 0)
    if risk_score > 1.5:
        status, color = "DANGER", "red"
        trigger_alert("danger")
    elif risk_score > 0.7:
        status, color = "WARNING", "orange"
        trigger_alert("warning")
    else:
        status, color = "SAFE", "green"

    cv2.putText(side_frame, f"STATUS: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                (0,0,255) if color=="red" else (0,165,255) if color=="orange" else (0,255,0), 3)

    return front_frame, side_frame, status, risk_score

# ====================== UI DASHBOARD ======================
st.title("🚗 **Advanced Driver Safety Monitor**")
st.markdown("Real-time monitoring using **3 AI models** + CNN + Live/Video/Upload")

tab1, tab2, tab3 = st.tabs(["📹 Live Stream", "📤 Upload Media", "📊 Dashboard"])

# === LIVE STREAM ===
with tab1:
    if not DEMO_MODE:
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.subheader("Front Camera (Drowsiness)")
            front_cap = st.checkbox("Enable Front Cam", key="front_live")
        with col_cam2:
            st.subheader("Side Camera (Distraction + Behavior)")
            side_cap = st.checkbox("Enable Side Cam", key="side_live")

        placeholder = st.empty()
        if front_cap or side_cap:
            cap_front = cv2.VideoCapture(0) if front_cap else None
            cap_side = cv2.VideoCapture(1) if side_cap else None
            while True:
                front_frame = cap_front.read()[1] if cap_front else create_demo_frame("No Front Cam")
                side_frame = cap_side.read()[1] if cap_side else create_demo_frame("No Side Cam")
                front_frame, side_frame, status, risk = process_frame(front_frame, side_frame)
                with placeholder.container():
                    st.image([front_frame, side_frame], channels="BGR", caption=["Front", "Side"])
                    st.metric("Risk Level", f"{risk:.2f}", delta=status)
                time.sleep(0.1)
    else:
        st.info("Live stream disabled in DEMO mode")

# === UPLOAD MEDIA ===
with tab2:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_front = st.file_uploader("Front (Drowsiness)", type=['jpg','png','mp4'], key="up_front")
    with col_up2:
        uploaded_side = st.file_uploader("Side (Distraction + Behavior)", type=['jpg','png','mp4'], key="up_side")

    if uploaded_front or uploaded_side:
        front_frame = load_media(uploaded_front) if uploaded_front else create_demo_frame("No Front")
        side_frame = load_media(uploaded_side) if uploaded_side else create_demo_frame("No Side")
        front_frame, side_frame, status, risk = process_frame(front_frame, side_frame)
        st.image([front_frame, side_frame], channels="BGR", caption=["Front", "Side"])
        st.metric("Final Status", status, delta=f"Risk: {risk:.2f}")

# === DASHBOARD ===
with tab3:
    st.metric("Active Models", f"{int(bool(distraction_model)) + int(bool(behavior_model)) + int(bool(drowsiness_cnn))}/3")
    st.markdown("### How to Use\n1. **Local**: Turn off DEMO, connect 2 cameras\n2. **Cloud**: Upload images/videos\n3. Get **unified alert**")

# ====================== HELPER FUNCTIONS ======================
def create_demo_frame(text):
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[:] = (50, 50, 100)
    cv2.putText(frame, text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame

def load_media(file):
    if file.type.startswith('image'):
        return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    elif file.type.startswith('video'):
        vid = cv2.VideoCapture(io.BytesIO(file.read()))
        ret, frame = vid.read()
        return frame if ret else create_demo_frame("Video Error")
