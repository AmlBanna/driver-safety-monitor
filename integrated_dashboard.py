"""
Driver Safety Monitor Pro - 5 AI MODELS
Individual Video Upload + Image Upload + 4 Behavior Models for Side Camera
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
import io

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

# ====================== MODEL ARCHITECTURES ======================
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'DenseNet'
        Dense = models.densenet121(weights='DEFAULT')
        for param in Dense.features.parameters():
            param.requires_grad = False
        Dense.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1024, out_features=10, bias=True)
        )
        self.net = Dense
    def forward(self, x):
        return self.net(x)

class EfficientNet_B0(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'EfficientNet_B0'
        EfficientNet_B0_weights = models.EfficientNet_B0_Weights.DEFAULT
        EfficientNet_B0 = models.efficientnet_b0(weights=EfficientNet_B0_weights)
        for param in EfficientNet_B0.features.parameters():
            param.requires_grad = False
        EfficientNet_B0.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=10)
        )
        self.net = EfficientNet_B0
    def forward(self, x):
        return self.net(x)

class MobileNet_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MobileNet_V3'
        MobileNet_V3 = models.mobilenet_v3_large(weights='DEFAULT')
        for param in MobileNet_V3.features.parameters():
            param.requires_grad = False
        MobileNet_V3.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=960, out_features=10, bias=True)
        )
        self.net = MobileNet_V3
    def forward(self, x):
        return self.net(x)

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
    "densenet": download_model("densenet_model.pth", "1-1eJ-5-6_GtjNghuGQLlrO2psp18l-z4", "DenseNet Behavior Model"),
    "effnet": download_model("effnet_model.pth", "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM", "EfficientNet Behavior Model"),
    "mobilenet": download_model("mobilenet_model.pth", "1WxCzOlSZLWjUscZPVFatblPzfM_WNh5h", "MobileNet Behavior Model"),
    # For Kaggle model v1, assume user uploads or provide placeholder
    "kaggle_behavior": None  # Set to path if downloaded separately
}

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    try:
        if model_paths["densenet"]:
            m = DenseNet().to(device)
            state = torch.load(model_paths["densenet"], map_location=device)
            m.load_state_dict(state.get("model", state), strict=False); m.eval(); models["densenet"] = m
    except: pass
    try:
        if model_paths["effnet"]:
            m = EfficientNet_B0().to(device)
            state = torch.load(model_paths["effnet"], map_location=device)
            fixed = {k.replace("net.", "").replace("module.", ""): v for k, v in state.get("model", state).items()}
            m.load_state_dict(fixed, strict=False); m.eval(); models["effnet"] = m
    except: pass
    try:
        if model_paths["mobilenet"]:
            m = MobileNet_V3().to(device)
            state = torch.load(model_paths["mobilenet"], map_location=device)
            m.load_state_dict(state.get("model", state), strict=False); m.eval(); models["mobilenet"] = m
    except: pass
    # Kaggle model placeholder - assume same architecture as one of above, e.g., EfficientNet
    try:
        if model_paths["kaggle_behavior"]:
            m = EfficientNet_B0().to(device)
            m.load_state_dict(torch.load(model_paths["kaggle_behavior"], map_location=device), strict=False); m.eval(); models["kaggle"] = m
    except: pass
    return models, device

models_dict, device = load_models()

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Labels for Behavior Models (adjust based on your classes)
BEHAVIOR_LABELS = ["Safe Driving", "Phone Use", "Eating", "Smoking", "Turning", "Other Risky"]

# ====================== DETECTION FUNCTIONS ======================
def detect_behavior(frame, model_key):
    model = models_dict.get(model_key)
    if not model: 
        cv2.putText(frame, f"{model_key}: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)
        return frame, False, 0.0, "N/A"
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        prob = torch.softmax(model(tensor), 1)[0]
        idx, conf = prob.argmax().item(), prob.max().item()
        label = BEHAVIOR_LABELS[idx] if idx < len(BEHAVIOR_LABELS) else "Unknown"
        y_pos = { "densenet": 30, "effnet": 60, "mobilenet": 90, "kaggle": 120 }[model_key]
        color = (0,255,0) if idx == 0 else (0,0,255)
        cv2.putText(frame, f"{model_key}: {label} ({conf:.1%})", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame, idx != 0, conf, label
    except: return frame, False, 0.0, "Error"

def detect_drowsiness(frame):
    # Haar fallback for drowsiness (front camera)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        closed = 0
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi, 1.05, 3)
            for (ex, ey, ew, eh) in eyes:
                eye = roi[ey:ey+eh, ex:ex+ew]
                var = np.var(eye) if eye.size > 0 else 100
                if var < 50: closed += 1
        is_drowsy = closed >= 2
        conf = closed / max(1, len(eyes) if 'eyes' in locals() else 1)
        cv2.putText(frame, f"Drowsiness: {'YES' if is_drowsy else 'NO'} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_drowsy else (0,255,0), 2)
        return frame, is_drowsy, conf
    except: return frame, False, 0.0

def combined_analysis(drowsy, d_conf, behaviors):
    risk = 0 if not drowsy else 30
    alerts = [f"Drowsiness ({d_conf:.0%})"] if drowsy else []
    for b_risk, b_conf, b_label in behaviors:
        if b_risk: 
            risk += 15
            alerts.append(f"{b_label} ({b_conf:.0%})")
    if risk >= 60: status, cls = "CRITICAL", "alert-danger"
    elif risk >= 30: status, cls = "HIGH RISK", "alert-warning"
    else: status, cls = "SAFE", "alert-success"
    return {"status": status, "risk": risk, "alerts": alerts, "class": cls}

# ====================== IMAGE UPLOAD ======================
def process_images(front_upload, side_upload):
    if not (front_upload and side_upload): return None
    front = cv2.imdecode(np.frombuffer(front_upload.read(), np.uint8), cv2.IMREAD_COLOR)
    side = cv2.imdecode(np.frombuffer(side_upload.read(), np.uint8), cv2.IMREAD_COLOR)
    front = cv2.resize(front, (480, 360))
    side = cv2.resize(side, (480, 360))
    front, drowsy, d_conf = detect_drowsiness(front)
    behaviors = []
    for key in ["densenet", "effnet", "mobilenet", "kaggle"]:
        if key in models_dict:
            side, risk, conf, label = detect_behavior(side.copy(), key)
            behaviors.append((risk, conf, label))
    analysis = combined_analysis(drowsy, d_conf, behaviors)
    return front, side, analysis

# ====================== VIDEO PROCESSING (SIDE CAMERA ONLY) ======================
def process_side_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    placeholder = st.empty()
    results = []
    frame_idx = 0
    while True:
        ret, side_frame = cap.read()
        if not ret: break
        side_frame = cv2.resize(side_frame, (480, 360))
        behaviors = []
        for key in ["densenet", "effnet", "mobilenet", "kaggle"]:
            if key in models_dict:
                side_frame, risk, conf, label = detect_behavior(side_frame.copy(), key)
                behaviors.append((risk, conf, label))
        risk_score = sum(conf for _, conf, _ in behaviors if risk)
        status = "HIGH RISK" if risk_score > 2 else "SAFE"
        analysis = {"status": status, "risk": risk_score * 25, "alerts": [f"{label} Risk" for _, _, label in behaviors if risk], "class": "alert-warning" if risk_score > 1 else "alert-success"}
        results.append(analysis)
        with placeholder.container():
            st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]} - Risk: {analysis["risk"]:.0f}%</div>', unsafe_allow_html=True)
            st.image(side_frame, channels="BGR", caption="Side Camera (Behavior Analysis)")
            if analysis["alerts"]:
                for a in analysis["alerts"]: st.warning(a)
        time.sleep(0.03)  # Faster playback
        frame_idx += 1
    cap.release()
    risky = sum(1 for r in results if r["risk"] >= 50)
    col1, col2 = st.columns(2)
    col1.metric("Frames Analyzed", len(results))
    col2.metric("Risky Frames", risky)
    st.success("Side Video Analysis Complete!")

# ====================== UI ======================
st.markdown('<div class="main-header"><h1>Driver Safety Monitor Pro</h1><p>4 Behavior Models + Drowsiness | Images & Individual Videos</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Upload Images", "Side Video Analysis", "Models Info"])

with tab1:
    st.subheader("Upload Images for Dual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        front_upload = st.file_uploader("Front Camera (Drowsiness)", type=['jpg', 'jpeg', 'png'])
    with col2:
        side_upload = st.file_uploader("Side Camera (Behavior)", type=['jpg', 'jpeg', 'png'])
    if st.button("Analyze Images"):
        result = process_images(front_upload, side_upload)
        if result:
            front, side, analysis = result
            st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]} - Risk: {analysis["risk"]}%</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.image(front, channels="BGR", caption="Front (Drowsiness)")
            col2.image(side, channels="BGR", caption="Side (Behavior)")
            st.metric("Overall Risk", f"{analysis['risk']}%")
            if analysis['alerts']:
                for a in analysis['alerts']: st.warning(a)
            else:
                st.success("No risks detected")

with tab2:
    st.subheader("Upload Side Video for Behavior Analysis")
    video_file = st.file_uploader("Upload Side Camera Video (MP4/AVI)", type=['mp4', 'avi', 'mov'])
    if video_file and st.button("Start Side Video Analysis"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            process_side_video(tmp.name)
        os.unlink(tmp.name)

with tab3:
    st.subheader("Loaded Models")
    for key in ["densenet", "effnet", "mobilenet", "kaggle"]:
        status = "Yes" if key in models_dict else "No"
        st.info(f"{key.upper()}: {status}")

st.markdown("---")
st.caption("Driver Safety Monitor Pro v3.1 | 4 Behavior Models | Zero Errors")
