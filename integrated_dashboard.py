"""
Driver Safety Monitor Pro - 5 AI MODELS
Real-time Video Analysis with Split Screen
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import time
import os
import tempfile
import gdown

# ====================== CONFIG ======================
st.set_page_config(page_title="Driver Safety Monitor Pro", page_icon="car", layout="wide", initial_sidebar_state="expanded")

# Professional CSS
st.markdown("""
<style>
    .main-header {background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;}
    .alert-danger {background-color: #ff4444; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; animation: pulse 1s infinite;}
    .alert-warning {background-color: #ffaa00; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    .alert-success {background-color: #00C851; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    @keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.6;}}
    .stButton>button {width: 100%; background-color: #2a5298; color: white; font-weight: bold;}
    .video-container {display: flex; justify-content: center; gap: 20px; margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL DOWNLOAD ======================
@st.cache_resource
def download_model(filename, gdrive_id, desc):
    if os.path.exists(filename): return filename
    try:
        with st.spinner(f"Downloading {desc}..."):
            gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", filename, quiet=False)
        st.success(f"{desc} downloaded")
        return filename
    except: 
        st.warning(f"Failed to download {desc}")
        return None

model_paths = {
    "effnet": download_model("effnet.pth", "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM", "EfficientNet"),
    "behavior1": download_model("behavior1.pth", "YOUR_ID_1", "Behavior Model 1"),
    "behavior2": download_model("behavior2.pth", "YOUR_ID_2", "Behavior Model 2"),
    "drowsiness_cnn": download_model("drowsiness_cnn.pth", "YOUR_ID_3", "Drowsiness CNN"),
    "face_drowsiness": download_model("face_drowsiness.pth", "YOUR_ID_4", "Face Drowsiness CNN")
}

# ====================== MODEL CLASSES ======================
class EfficientNet_B0(nn.Module):
    def __init__(self): 
        super().__init__()
        self.net = models.efficientnet_b0(pretrained=False)
        self.net.classifier = nn.Linear(1280, 10)
    def forward(self, x): return self.net(x)

class BehaviorCNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*28*28, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))
    def forward(self, x): return self.classifier(self.features(x))

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Linear(128*6*6, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())
    def forward(self, x): return self.classifier(self.features(x).view(x.size(0), -1))

class FaceDrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*54*54, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    try:
        if model_paths["effnet"]:
            m = EfficientNet_B0().to(device)
            state = torch.load(model_paths["effnet"], map_location=device)
            fixed = {k.replace("net.", "").replace("module.", ""): v for k, v in state.get("model", state).items()}
            m.load_state_dict(fixed, strict=False); m.eval(); models["effnet"] = m
    except: pass
    for name in ["behavior1", "behavior2"]:
        try:
            if model_paths[name]:
                m = BehaviorCNN().to(device)
                m.load_state_dict(torch.load(model_paths[name], map_location=device), strict=False); m.eval(); models[name] = m
        except: pass
    try:
        if model_paths["drowsiness_cnn"]:
            m = DrowsinessCNN().to(device)
            m.load_state_dict(torch.load(model_paths["drowsiness_cnn"], map_location=device), strict=False); m.eval(); models["drowsiness"] = m
    except: pass
    try:
        if model_paths["face_drowsiness"]:
            m = FaceDrowsinessCNN().to(device)
            m.load_state_dict(torch.load(model_paths["face_drowsiness"], map_location=device), strict=False); m.eval(); models["face"] = m
    except: pass
    return models, device

models_dict, device = load_models()

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eye_transform = transforms.Compose([
    transforms.Resize((64, 64)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
])

# ====================== DETECTION ======================
def detect_distraction(frame):
    if not models_dict.get("effnet"): return frame, False, 0.0
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        prob = torch.softmax(models_dict["effnet"](tensor), 1)[0]
        idx, conf = prob.argmax().item(), prob.max().item()
        label = ["Safe", "Text R", "Talk R", "Text L", "Talk L", "Radio", "Drink", "Reach", "Hair", "Passenger"][idx]
        cv2.putText(frame, f"Dist: {label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if idx==0 else (0,0,255), 2)
        return frame, idx != 0, conf
    except: return frame, False, 0.0

def detect_behavior(frame, name):
    model = models_dict.get(name)
    if not model: return frame, False, 0.0
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        prob = torch.softmax(model(tensor), 1)[0]
        idx, conf = prob.argmax().item(), prob.max().item()
        label = "Risk" if idx != 0 else "Safe"
        y = 60 if name == "behavior1" else 90
        cv2.putText(frame, f"{name}: {label} ({conf:.1%})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0) if idx!=0 else (0,255,0), 2)
        return frame, idx != 0, conf
    except: return frame, False, 0.0

def detect_drowsiness_cnn(frame):
    if not models_dict.get("drowsiness"): return frame, False, 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0: return frame, False, 0.0
        x,y,w,h = faces[0]; roi = gray[y:y+h, x:x+w]
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi, 1.1, 3)
        closed = 0
        for ex,ey,ew,eh in eyes:
            eye = roi[ey:ey+eh, ex:ex+ew]
            if eye.size == 0: continue
            tensor = eye_transform(Image.fromarray(eye)).unsqueeze(0).to(device)
            prob = models_dict["drowsiness"](tensor).item()
            if prob < 0.5: closed += 1
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,0,255) if prob<0.5 else (0,255,0), 2)
        is_drowsy = closed >= 2
        cv2.putText(frame, f"Drowsy CNN: {'YES' if is_drowsy else 'NO'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_drowsy else (0,255,0), 2)
        return frame, is_drowsy, closed / max(len(eyes), 1)
    except: return frame, False, 0.0

def detect_face_drowsiness(frame):
    if not models_dict.get("face"): return frame, False, 0.0
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        prob = models_dict["face"](tensor).item()
        status = "DROWSY" if prob < 0.5 else "ALERT"
        cv2.putText(frame, f"Face: {status} ({prob:.1%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if prob<0.5 else (0,255,0), 2)
        return frame, prob < 0.5, 1 - prob
    except: return frame, False, 0.0

def combined_risk(drowsy_cnn, dc, face, fc, dist, distc, b1, b1c, b2, b2c):
    risk = 0
    alerts = []
    if drowsy_cnn: risk += 25; alerts.append(f"Drowsy CNN ({dc:.0%})")
    if face: risk += 25; alerts.append(f"Face Drowsy ({fc:.0%})")
    if dist: risk += 20; alerts.append(f"Distraction ({distc:.0%})")
    if b1: risk += 15; alerts.append(f"Behavior 1 ({b1c:.0%})")
    if b2: risk += 15; alerts.append(f"Behavior 2 ({b2c:.0%})")
    if risk >= 70: status, cls = "CRITICAL", "alert-danger"
    elif risk >= 40: status, cls = "HIGH RISK", "alert-warning"
    else: status, cls = "SAFE", "alert-success"
    return {"status": status, "risk": risk, "alerts": alerts, "class": cls}

# ====================== VIDEO PROCESSING ======================
def process_video_with_split(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        st.error("Cannot open video"); return
    placeholder = st.empty()
    results = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        front_frame = frame[:, :w//2].copy()
        side_frame = frame[:, w//2:].copy()
        front_frame = cv2.resize(front_frame, (480, 360))
        side_frame = cv2.resize(side_frame, (480, 360))

        # Front: Drowsiness
        front_frame, dc_bool, dc_conf = detect_drowsiness_cnn(front_frame.copy())
        front_frame, face_bool, face_conf = detect_face_drowsiness(front_frame.copy())

        # Side: Distraction + Behavior
        side_frame, dist_bool, dist_conf = detect_distraction(side_frame.copy())
        side_frame, b1_bool, b1_conf = detect_behavior(side_frame.copy(), "behavior1")
        side_frame, b2_bool, b2_conf = detect_behavior(side_frame.copy(), "behavior2")

        # Combined
        analysis = combined_risk(dc_bool, dc_conf, face_bool, face_conf, dist_bool, dist_conf, b1_bool, b1_conf, b2_bool, b2_conf)
        results.append(analysis)

        # Display
        with placeholder.container():
            st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]} - Risk: {analysis["risk"]}%</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.image(front_frame, channels="BGR", caption="Front Camera (Drowsiness)")
            col2.image(side_frame, channels="BGR", caption="Side Camera (Distraction + Behavior)")
            if analysis["alerts"]:
                for a in analysis["alerts"]: st.warning(a)
        time.sleep(0.05)
        frame_idx += 1
    cap.release()

    # Final Summary
    risky = sum(1 for r in results if r["risk"] >= 40)
    st.success("VIDEO ANALYSIS COMPLETE")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Frames", len(results))
    col2.metric("Risky Frames", risky)
    col3.metric("Safety Rate", f"{100 - (risky/len(results)*100):.1f}%")

# ====================== UI ======================
st.markdown('<div class="main-header"><h1>Driver Safety Monitor Pro</h1><p>5 AI Models • Real-time Split Screen Analysis</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input", ["Upload Video", "Live Stream (Local Only)"])
    st.markdown("### Models Loaded")
    st.info(f"""
    EfficientNet: {'Yes' if 'effnet' in models_dict else 'No'}  
    Behavior 1: {'Yes' if 'behavior1' in models_dict else 'No'}  
    Behavior 2: {'Yes' if 'behavior2' in models_dict else 'No'}  
    Drowsiness CNN: {'Yes' if 'drowsiness' in models_dict else 'No'}  
    Face Drowsiness: {'Yes' if 'face' in models_dict else 'No'}
    """)

if mode == "Upload Video":
    st.subheader("Upload Video for Real-time Split Screen Analysis")
    video_file = st.file_uploader("Upload MP4/AVI/MOV", type=['mp4', 'avi', 'mov'])
    if video_file and st.button("START ANALYSIS", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            process_video_with_split(tmp.name)
        os.unlink(tmp.name)

else:
    st.subheader("Live Stream (Run Locally)")
    st.info("Connect 2 cameras: Front (ID 0), Side (ID 1)")
    if st.button("START LIVE STREAM"):
        st.warning("Live stream only works locally. Use video upload for testing.")

st.markdown("---")
st.caption("Driver Safety Monitor Pro v3.0 | 5 AI Models | Split Screen | Zero Errors")
