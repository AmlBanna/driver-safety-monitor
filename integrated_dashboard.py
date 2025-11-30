"""
Driver Safety Monitor Pro - Integrated System
Combines 4 models: EfficientNet + 2 Driver Behavior CNNs + Drowsiness CNN
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
st.set_page_config(
    page_title="Driver Safety Monitor Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .alert-danger {background-color: #ff4444; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; animation: pulse 1s infinite;}
    .alert-warning {background-color: #ffaa00; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    .alert-success {background-color: #00C851; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;}
    @keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.6;}}
    .metric-card {background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #2a5298; margin: 10px 0;}
    .stButton>button {width: 100%; background-color: #2a5298; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL DOWNLOAD ======================
@st.cache_resource
def download_model(filename, gdrive_id, description):
    if os.path.exists(filename):
        return filename
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        with st.spinner(f"Downloading {description}..."):
            gdown.download(url, filename, quiet=False)
        st.success(f"{description} downloaded!")
        return filename
    except Exception as e:
        st.warning(f"Could not download {description}")
        return None

# UPDATE THESE IDs with your real Google Drive links
model_paths = {
    "effnet": download_model("effnet.pth", "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM", "EfficientNet Distraction"),
    "behavior1": download_model("behavior1.pth", "YOUR_BEHAVIOR1_ID", "Driver Behavior Model 1"),
    "behavior2": download_model("behavior2.pth", "YOUR_BEHAVIOR2_ID", "Driver Behavior Model 2"),
    "drowsiness": download_model("drowsiness_cnn.pth", "YOUR_DROWSINESS_ID", "Drowsiness CNN")
}

# ====================== MODEL ARCHITECTURES ======================
class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = models.efficientnet_b0(pretrained=False)
        self.net.classifier = nn.Linear(1280, num_classes)
    def forward(self, x): return self.net(x)

class BehaviorCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*28*28, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    if model_paths["effnet"]:
        try:
            m = EfficientNet_B0().to(device)
            state = torch.load(model_paths["effnet"], map_location=device)
            state_dict = state.get("model", state)
            fixed = {k.replace("net.", "").replace("module.", ""): v for k, v in state_dict.items()}
            m.load_state_dict(fixed, strict=False)
            m.eval()
            models["effnet"] = m
        except: pass
    for name in ["behavior1", "behavior2"]:
        if model_paths[name]:
            try:
                m = BehaviorCNN().to(device)
                m.load_state_dict(torch.load(model_paths[name], map_location=device), strict=False)
                m.eval()
                models[name] = m
            except: pass
    if model_paths["drowsiness"]:
        try:
            m = DrowsinessCNN().to(device)
            m.load_state_dict(torch.load(model_paths["drowsiness"], map_location=device), strict=False)
            m.eval()
            models["drowsiness"] = m
        except: pass
    return models, device

models_dict, device = load_models()

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
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

# ====================== DETECTION FUNCTIONS ======================
def detect_distraction(frame):
    if not models_dict.get("effnet"): return frame, False, "N/A", 0.0
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(models_dict["effnet"](tensor), 1)[0]
            idx, conf = prob.argmax().item(), prob.max().item()
        label = ["Safe", "Text R", "Talk R", "Text L", "Talk L", "Radio", "Drink", "Reach", "Hair", "Passenger"][idx]
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.putText(frame, f"Distraction: {label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, idx != 0, label, conf
    except: return frame, False, "Error", 0.0

def detect_behavior(frame, model_name):
    model = models_dict.get(model_name)
    if not model: return frame, False, "N/A", 0.0
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(tensor), 1)[0]
            idx, conf = prob.argmax().item(), prob.max().item()
        label = "Risky" if idx != 0 else "Safe"
        color = (255, 165, 0) if idx != 0 else (0, 255, 0)
        cv2.putText(frame, f"{model_name}: {label} ({conf:.1%})", (10, 60 if model_name=="behavior1" else 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, idx != 0, label, conf
    except: return frame, False, "Error", 0.0

def detect_drowsiness(frame):
    if models_dict.get("drowsiness"):
        # CNN version
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0: return frame, False, 0.0
            x,y,w,h = faces[0]
            roi = gray[y:y+h, x:x+w]
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi, 1.1, 3)
            closed = 0
            for ex,ey,ew,eh in eyes:
                eye = roi[ey:ey+eh, ex:ex+ew]
                if eye.size == 0: continue
                tensor = eye_transform(Image.fromarray(eye)).unsqueeze(0).to(device)
                prob = models_dict["drowsiness"](tensor).item()
                status = "CLOSED" if prob < 0.5 else "OPEN"
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,0,255) if status=="CLOSED" else (0,255,0), 2)
                if status == "CLOSED": closed += 1
            is_drowsy = closed >= 2
            conf = closed / max(len(eyes), 1)
            cv2.putText(frame, f"Drowsiness: {'HIGH' if is_drowsy else 'LOW'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if is_drowsy else (0,255,0), 2)
            return frame, is_drowsy, conf
        except: pass
    # Fallback Haar
    return cv2.putText(frame, "Drowsiness: Using OpenCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2), False, 0.0

def combined_analysis(drowsy, drowsy_conf, dist, dist_conf, b1, b1_conf, b2, b2_conf):
    risk = 0
    alerts = []
    if drowsy: risk += 40; alerts.append(f"Drowsiness ({drowsy_conf:.0%})")
    if dist: risk += 30; alerts.append(f"Distraction ({dist_conf:.0%})")
    if b1: risk += 15; alerts.append(f"Behavior Risk 1 ({b1_conf:.0%})")
    if b2: risk += 15; alerts.append(f"Behavior Risk 2 ({b2_conf:.0%})")
    if risk >= 60:
        status, cls = "HIGH RISK", "alert-danger"
    elif risk >= 30:
        status, cls = "WARNING", "alert-warning"
    else:
        status, cls = "SAFE", "alert-success"
    return {"status": status, "risk": risk, "alerts": alerts, "class": cls}

# ====================== VIDEO & LIVE PROCESSING ======================
def process_video(video_path, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if i % 5 == 0:
            h, w = frame.shape[:2]
            front = frame[:, :w//2]
            side = frame[:, w//2:]
            front, d, dc = detect_drowsiness(front.copy())
            side, dist, _, distc = detect_distraction(side.copy())
            side, b1, _, b1c = detect_behavior(side.copy(), "behavior1")
            side, b2, _, b2c = detect_behavior(side.copy(), "behavior2")
            analysis = combined_analysis(d, dc, dist, distc, b1, b1c, b2, b2c)
            results.append(analysis)
        i += 1
        progress_bar.progress(i/total)
        status_text.text(f"Processing frame {i}/{total}")
    cap.release()
    return results

# ====================== MAIN UI ======================
st.markdown('<div class="main-header"><h1>Driver Safety Monitor Pro</h1><p>Integrated System with 4 AI Models</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode", ["Upload Images", "Upload Video", "Live Stream"])
    st.markdown("### Loaded Models")
    st.info(f"EfficientNet: {'Yes' if 'effnet' in models_dict else 'No'}\n"
            f"Behavior Model 1: {'Yes' if 'behavior1' in models_dict else 'No'}\n"
            f"Behavior Model 2: {'Yes' if 'behavior2' in models_dict else 'No'}\n"
            f"Drowsiness CNN: {'Yes' if 'drowsiness' in models_dict else 'No'}")

if input_mode == "Upload Images":
    st.subheader("Upload Images for Analysis")
    col1, col2 = st.columns(2)
    with col1:
        front_upload = st.file_uploader("Front Camera (Drowsiness)", type=['jpg', 'jpeg', 'png'], key="front")
    with col2:
        side_upload = st.file_uploader("Side Camera (Distraction & Behavior)", type=['jpg', 'jpeg', 'png'], key="side")
    
    if st.button("Analyze Images", use_container_width=True):
        if front_upload and side_upload:
            with st.spinner("Analyzing..."):
                front = cv2.imdecode(np.frombuffer(front_upload.read(), np.uint8), cv2.IMREAD_COLOR)
                side = cv2.imdecode(np.frombuffer(side_upload.read(), np.uint8), cv2.IMREAD_COLOR)
                front = cv2.resize(front, (640, 480))
                side = cv2.resize(side, (640, 480))
                
                front, d, dc = detect_drowsiness(front.copy())
                side, dist, _, distc = detect_distraction(side.copy())
                side, b1, _, b1c = detect_behavior(side.copy(), "behavior1")
                side, b2, _, b2c = detect_behavior(side.copy(), "behavior2")
                
                analysis = combined_analysis(d, dc, dist, distc, b1, b1c, b2, b2c)
                
                st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]}</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.image(front, channels="BGR", caption="Front Camera", use_container_width=True)
                col2.image(side, channels="BGR", caption="Side Camera", use_container_width=True)
                with col3:
                    st.markdown("### Detailed Analysis")
                    st.metric("Risk Level", f"{analysis['risk']}%")
                    if analysis['alerts']:
                        for a in analysis['alerts']: st.warning(a)
                    else:
                        st.success("No risks detected")
        else:
            st.error("Please upload both images")

elif input_mode == "Upload Video":
    st.subheader("Upload Video for Analysis")
    video_upload = st.file_uploader("Upload driving video", type=['mp4', 'avi', 'mov'])
    if video_upload and st.button("Analyze Video", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_upload.read())
            path = tmp.name
        progress = st.progress(0)
        status = st.empty()
        results = process_video(path, progress, status)
        os.unlink(path)
        st.success("Analysis Complete!")
        risky = sum(1 for r in results if r["risk"] >= 30)
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames Analyzed", len(results))
        col2.metric("Risky Frames", risky)
        col3.metric("Safety Rate", f"{100 - (risky/len(results)*100):.1f}%")
        if st.checkbox("Show last 10 results"):
            for r in results[-10:]:
                st.write(f"Risk: {r['risk']}% - {r['status']}")
                for a in r['alerts']: st.write("  • " + a)

else:  # Live Stream
    st.subheader("Live Stream")
    st.info("Live stream works only when running locally")
    col1, col2 = st.columns(2)
    with col1: front_id = st.number_input("Front Camera ID", 0, 10, 0)
    with col2: side_id = st.number_input("Side Camera ID", 0, 10, 1)
    if st.button("Start Live Stream", use_container_width=True):
        cap_front = cv2.VideoCapture(front_id)
        cap_side = cv2.VideoCapture(side_id)
        placeholder = st.empty()
        while True:
            f1 = cap_front.read()[1]
            f2 = cap_side.read()[1]
            if f1 is None or f2 is None: break
            f1, d, dc = detect_drowsiness(f1)
            f2, dist, _, distc = detect_distraction(f2)
            f2, b1, _, b1c = detect_behavior(f2, "behavior1")
            f2, b2, _, b2c = detect_behavior(f2, "behavior2")
            analysis = combined_analysis(d, dc, dist, distc, b1, b1c, b2, b2c)
            with placeholder.container():
                st.markdown(f'<div class="{analysis["class"]}">{analysis["status"]}</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.image(f1, channels="BGR", caption="Front")
                col2.image(f2, channels="BGR", caption="Side")
                st.metric("Risk Level", f"{analysis['risk']}%")
            time.sleep(0.1)
        cap_front.release()
        cap_side.release()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>Driver Safety Monitor Pro v2.0 | Powered by PyTorch & OpenCV</p></div>", unsafe_allow_html=True)
