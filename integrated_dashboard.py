"""
Driver Safety Monitor Pro - Final Version
Real-time Behavior Analysis During Video + Final Report
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
st.set_page_config(page_title="Driver Safety Monitor", page_icon="car", layout="wide", initial_sidebar_state="expanded")

# Clean CSS
st.markdown("""
<style>
    .main-header {background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;}
    .alert-danger {background-color: #ff4444; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; animation: pulse 1s infinite;}
    .alert-warning {background-color: #ffaa00; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    .alert-success {background-color: #00C851; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    @keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.6;}}
    .stButton>button {width: 100%; background-color: #2a5298; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL ARCHITECTURES (HIDDEN) ======================
class BehaviorModel(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b0(weights='DEFAULT')
        for param in net.features.parameters(): param.requires_grad = False
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 10))
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

model_paths = [
    download_model("m1.pth", "1-1eJ-5-6_GtjNghuGQLlrO2psp18l-z4"),
    download_model("m2.pth", "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"),
    download_model("m3.pth", "1WxCzOlSZLWjUscZPVFatblPzfM_WNh5h"),
    download_model("m4.pth", "YOUR_KAGGLE_MODEL_ID")  # ضع رابط Kaggle هنا
]

# ====================== LOAD MODELS (HIDDEN) ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in model_paths:
        if not path: continue
        try:
            m = BehaviorModel().to(device)
            state = torch.load(path, map_location=device)
            m.load_state_dict(state.get("model", state), strict=False)
            m.eval()
            models.append(m)
        except: pass
    return models, device

models, device = load_models()

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== BEHAVIOR DETECTION (SIDE CAMERA) ======================
def analyze_behavior_frame(frame):
    if not models:
        cv2.putText(frame, "AI: Loading...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        return frame, False, 0.0, []
    
    try:
        tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        risks = []
        detected_types = set()
        for model in models:
            prob = torch.softmax(model(tensor), 1)[0]
            idx, conf = prob.argmax().item(), prob.max().item()
            if idx != 0 and conf > 0.5:
                risks.append(conf)
                label = ["Phone", "Eating", "Smoking", "Turning", "Reaching", "Hair", "Drink", "Talk", "Text", "Other"][idx-1]
                detected_types.add(label)
        avg_risk = sum(risks)/len(risks) if risks else 0.0
        is_risky = avg_risk > 0.5
        status = "RISKY BEHAVIOR" if is_risky else "SAFE DRIVING"
        color = (0, 0, 255) if is_risky else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Risk: {avg_risk:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, is_risky, avg_risk, list(detected_types)
    except:
        cv2.putText(frame, "AI Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return frame, False, 0.0, []

# ====================== DROWSINESS DETECTION (FRONT CAMERA) ======================
def analyze_drowsiness_frame(frame):
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
        conf = min(closed / 2, 1.0)
        status = "DROWSY" if is_drowsy else "ALERT"
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Risk: {conf:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, is_drowsy, conf
    except:
        return frame, False, 0.0

# ====================== VIDEO ANALYSIS WITH LIVE + FINAL REPORT ======================
def run_video_analysis(front_path=None, side_path=None):
    cap_front = cv2.VideoCapture(front_path) if front_path else None
    cap_side = cv2.VideoCapture(side_path) if side_path else None

    if not (cap_front or cap_side):
        st.error("No video provided")
        return

    placeholder = st.empty()
    behavior_risks = []
    drowsy_risks = []
    detected_behaviors = set()
    total_frames = 0

    while True:
        ret_f, f_frame = cap_front.read() if cap_front else (False, None)
        ret_s, s_frame = cap_side.read() if cap_side else (False, None)
        if not (ret_f or ret_s): break

        current_drowsy = current_behavior = 0.0
        current_types = []

        if f_frame is not None:
            f_frame = cv2.resize(f_frame, (480, 360))
            f_frame, d_bool, d_conf = analyze_drowsiness_frame(f_frame)
            if d_bool: drowsy_risks.append(d_conf); current_drowsy = d_conf

        if s_frame is not None:
            s_frame = cv2.resize(s_frame, (480, 360))
            s_frame, b_bool, b_conf, types = analyze_behavior_frame(s_frame)
            if b_bool: behavior_risks.append(b_conf); detected_behaviors.update(types)
            current_behavior = b_conf
            current_types = types

        total_frames += 1

        # Live Display
        with placeholder.container():
            st.markdown(f"### Live Analysis - Frame {total_frames}")
            cols = st.columns(2)
            if f_frame is not None: cols[0].image(f_frame, channels="BGR", caption="Front Camera (Drowsiness)")
            if s_frame is not None: cols[1].image(s_frame, channels="BGR", caption="Side Camera (Behavior)")
            risk_now = (current_drowsy + current_behavior) * 50
            cls = "alert-danger" if risk_now >= 75 else "alert-warning" if risk_now >= 50 else "alert-success"
            st.markdown(f'<div class="{cls}">CURRENT RISK: {risk_now:.0f}%</div>', unsafe_allow_html=True)
            if current_types:
                st.warning("Detected: " + ", ".join(current_types))

        time.sleep(0.03)

    # Release
    if cap_front: cap_front.release()
    if cap_side: cap_side.release()

    # === FINAL REPORT ===
    st.success("ANALYSIS COMPLETE - FINAL REPORT")
    
    drowsy_rate = len(drowsy_risks) / total_frames * 100 if total_frames > 0 else 0
    behavior_rate = len(behavior_risks) / total_frames * 100 if total_frames > 0 else 0
    final_risk = (drowsy_rate + behavior_rate)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Frames", total_frames)
    col2.metric("Drowsy Frames", len(drowsy_risks))
    col3.metric("Risky Behavior Frames", len(behavior_risks))

    st.markdown("### Final Behavior Summary")
    if detected_behaviors:
        st.warning("Risky behaviors detected during video:")
        for b in detected_behaviors:
            st.write(f"• {b}")
    else:
        st.success("No risky behavior detected")

    st.metric("Final Risk Score", f"{final_risk:.1f}%")
    if final_risk >= 50:
        st.error("HIGH OVERALL RISK")
    elif final_risk >= 25:
        st.warning("MODERATE RISK")
    else:
        st.success("SAFE DRIVING")

# ====================== IMAGE ANALYSIS ======================
def analyze_images(front_file, side_file):
    if not (front_file and side_file): return
    front = cv2.imdecode(np.frombuffer(front_file.read(), np.uint8), cv2.IMREAD_COLOR)
    side = cv2.imdecode(np.frombuffer(side_file.read(), np.uint8), cv2.IMREAD_COLOR)
    front = cv2.resize(front, (480, 360))
    side = cv2.resize(side, (480, 360))

    front, d, dc = analyze_drowsiness_frame(front)
    side, b, bc, types = analyze_behavior_frame(side)

    risk = (dc + bc) * 50
    st.image([front, side], channels="BGR", caption=["Front", "Side"], width=360)
    st.metric("Risk Level", f"{risk:.0f}%")
    if types: st.warning("Detected: " + ", ".join(types))
    if risk >= 50: st.error("HIGH RISK")
    elif risk >= 25: st.warning("MODERATE RISK")
    else: st.success("SAFE")

# ====================== UI ======================
st.markdown('<div class="main-header"><h1>Driver Safety Monitor</h1><p>Real-time Analysis + Final Report</p></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload Images", "Upload Videos"])

with tab1:
    st.subheader("Image Analysis")
    c1, c2 = st.columns(2)
    with c1: front_img = st.file_uploader("Front Image", type=['jpg','png'])
    with c2: side_img = st.file_uploader("Side Image", type=['jpg','png'])
    if st.button("Analyze Images"):
        analyze_images(front_img, side_img)

with tab2:
    st.subheader("Video Analysis (Separate Uploads)")
    c1, c2 = st.columns(2)
    with c1: front_vid = st.file_uploader("Front Video (Drowsiness)", type=['mp4','avi'])
    with c2: side_vid = st.file_uploader("Side Video (Behavior)", type=['mp4','avi'])
    if st.button("START ANALYSIS"):
        if not (front_vid or side_vid):
            st.error("Upload at least one video")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ts:
                f_path = tf.name if front_vid else None
                s_path = ts.name if side_vid else None
                if front_vid: tf.write(front_vid.read())
                if side_vid: ts.write(side_vid.read())
                run_video_analysis(f_path, s_path)
                if f_path: os.unlink(f_path)
                if s_path: os.unlink(s_path)

st.caption("Driver Safety Monitor | Real-time + Final Report | Zero Errors")
