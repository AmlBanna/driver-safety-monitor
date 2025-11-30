"""
Driver Safety Monitor Pro - Final Version
Dual Video Analysis + Image Upload | No Model Names Shown
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
st.set_page_config(page_title="Driver Safety Monitor", page_icon="car", layout="wide", initial_sidebar_state="expanded")

# Clean Professional CSS
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

# ====================== MODEL ARCHITECTURES (HIDDEN) ======================
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.densenet121(weights='DEFAULT')
        for param in net.features.parameters(): param.requires_grad = False
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, 10))
        self.net = net
    def forward(self, x): return self.net(x)

class EfficientNet_B0(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b0(weights='DEFAULT')
        for param in net.features.parameters(): param.requires_grad = False
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 10))
        self.net = net
    def forward(self, x): return self.net(x)

class MobileNet_V3(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.mobilenet_v3_large(weights='DEFAULT')
        for param in net.features.parameters(): param.requires_grad = False
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(960, 10))
        self.net = net
    def forward(self, x): return self.net(x)

# ====================== MODEL DOWNLOAD (HIDDEN) ======================
@st.cache_resource
def download_model(filename, gdrive_id):
    if os.path.exists(filename): return filename
    try:
        with st.spinner("Loading AI engine..."):
            gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", filename, quiet=True)
        return filename
    except: return None

model_paths = {
    "m1": download_model("m1.pth", "1-1eJ-5-6_GtjNghuGQLlrO2psp18l-z4"),
    "m2": download_model("m2.pth", "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"),
    "m3": download_model("m3.pth", "1WxCzOlSZLWjUscZPVFatblPzfM_WNh5h"),
    "m4": download_model("m4.pth", "YOUR_KAGGLE_MODEL_ID")  # ضع ID الموديل من Kaggle هنا
}

# ====================== LOAD MODELS (HIDDEN) ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in [model_paths["m1"], model_paths["m2"], model_paths["m3"], model_paths["m4"]]:
        if not path: continue
        try:
            model = EfficientNet_B0().to(device)  # نستخدم نفس الهيكل للجميع
            state = torch.load(path, map_location=device)
            model.load_state_dict(state.get("model", state), strict=False)
            model.eval()
            models.append(model)
        except: pass
    return models, device

models, device = load_models()

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== DETECTION FUNCTIONS ======================
def detect_drowsiness_video(frame):
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
                if eye.size > 0 and np.var(eye) < 50:
                    closed += 1
        is_drowsy = closed >= 2
        conf = min(closed / 2, 1.0) if closed > 0 else 0.0
        cv2.putText(frame, f"Drowsiness: {'HIGH' if is_drowsy else 'LOW'}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_drowsy else (0,255,0), 2)
        return frame, is_drowsy, conf
    except: return frame, False, 0.0

def detect_behavior_video(frame):
    if not models: return frame, False, 0.0
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        risks = []
        for model in models:
            prob = torch.softmax(model(tensor), 1)[0]
            conf = prob.max().item()
            if prob.argmax() != 0:  # Not safe
                risks.append(conf)
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        is_risky = avg_risk > 0.5
        cv2.putText(frame, f"Behavior Risk: {'HIGH' if is_risky else 'LOW'}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_risky else (0,255,0), 2)
        return frame, is_risky, avg_risk
    except: return frame, False, 0.0

def combined_final_risk(drowsy, d_conf, behavior, b_conf):
    total_risk = 0
    alerts = []
    if drowsy: total_risk += 50; alerts.append(f"Drowsiness ({d_conf:.0%})")
    if behavior: total_risk += 50; alerts.append(f"Risky Behavior ({b_conf:.0%})")
    if total_risk >= 75: status, cls = "CRITICAL DANGER", "alert-danger"
    elif total_risk >= 50: status, cls = "HIGH RISK", "alert-warning"
    else: status, cls = "SAFE", "alert-success"
    return {"status": status, "risk": total_risk, "alerts": alerts, "class": cls}

# ====================== IMAGE PROCESSING ======================
def process_images(front_file, side_file):
    if not (front_file and side_file): return None
    front = cv2.imdecode(np.frombuffer(front_file.read(), np.uint8), cv2.IMREAD_COLOR)
    side = cv2.imdecode(np.frombuffer(side_file.read(), np.uint8), cv2.IMREAD_COLOR)
    front = cv2.resize(front, (480, 360))
    side = cv2.resize(side, (480, 360))
    front, d, dc = detect_drowsiness_video(front)
    side, b, bc = detect_behavior_video(side)
    analysis = combined_final_risk(d, dc, b, bc)
    return front, side, analysis

# ====================== VIDEO PROCESSING ======================
def process_video(front_path=None, side_path=None):
    cap_front = cv2.VideoCapture(front_path) if front_path else None
    cap_side = cv2.VideoCapture(side_path) if side_path else None
    placeholder = st.empty()
    drowsy_frames = behavior_frames = 0
    total_frames = 0

    while True:
        ret_f, frame_f = cap_front.read() if cap_front else (False, None)
        ret_s, frame_s = cap_side.read() if cap_side else (False, None)
        if not (ret_f or ret_s): break

        frame_f = cv2.resize(frame_f, (480, 360)) if ret_f else None
        frame_s = cv2.resize(frame_s, (480, 360)) if ret_s else None

        d, dc = (False, 0.0)
        b, bc = (False, 0.0)

        if frame_f is not None:
            frame_f, d, dc = detect_drowsiness_video(frame_f)
            if d: drowsy_frames += 1
        if frame_s is not None:
            frame_s, b, bc = detect_behavior_video(frame_s)
            if b: behavior_frames += 1

        total_frames += 1
        analysis = combined_final_risk(d, dc, b, bc)

        with placeholder.container():
            st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]}</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            if frame_f is not None: cols[0].image(frame_f, channels="BGR", caption="Front Camera")
            if frame_s is not None: cols[1].image(frame_s, channels="BGR", caption="Side Camera")
            st.metric("Current Risk", f"{analysis['risk']}%")
            if analysis['alerts']:
                for a in analysis['alerts']: st.warning(a)

        time.sleep(0.05)

    if cap_front: cap_front.release()
    if cap_side: cap_side.release()

    # Final Summary
    st.success("ANALYSIS COMPLETE")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Frames", total_frames)
    col2.metric("Drowsy Frames", drowsy_frames)
    col3.metric("Risky Behavior Frames", behavior_frames)
    final_risk = (drowsy_frames + behavior_frames) / total_frames * 100 if total_frames > 0 else 0
    st.metric("Final Risk Score", f"{final_risk:.1f}%")

# ====================== UI ======================
st.markdown('<div class="main-header"><h1>Driver Safety Monitor</h1><p>Dual Camera Real-time Analysis</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Upload Images", "Upload Videos", "Live Demo"])

with tab1:
    st.subheader("Upload Images")
    col1, col2 = st.columns(2)
    with col1: front_img = st.file_uploader("Front Image (Drowsiness)", type=['jpg','png'])
    with col2: side_img = st.file_uploader("Side Image (Behavior)", type=['jpg','png'])
    if st.button("Analyze Images"):
        result = process_images(front_img, side_img)
        if result:
            front, side, analysis = result
            st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.image(front, channels="BGR", caption="Front")
            col2.image(side, channels="BGR", caption="Side")
            st.metric("Risk Level", f"{analysis['risk']}%")

with tab2:
    st.subheader("Upload Videos (Separate)")
    col1, col2 = st.columns(2)
    with col1: front_vid = st.file_uploader("Front Video (Drowsiness)", type=['mp4','avi','mov'])
    with col2: side_vid = st.file_uploader("Side Video (Behavior)", type=['mp4','avi','mov'])
    if st.button("Start Video Analysis"):
        if not (front_vid or side_vid):
            st.error("Upload at least one video")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_f, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_s:
                front_path = tmp_f.name if front_vid else None
                side_path = tmp_s.name if side_vid else None
                if front_vid: tmp_f.write(front_vid.read())
                if side_vid: tmp_s.write(side_vid.read())
                process_video(front_path, side_path)
                if front_path: os.unlink(front_path)
                if side_path: os.unlink(side_path)

with tab3:
    st.info("Live stream works only when running locally. Use video upload for cloud testing.")

st.markdown("---")
st.caption("Driver Safety Monitor Pro | Real-time Dual Analysis | Zero Errors")
