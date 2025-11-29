# integrated_dashboard.py - FIXED FOR GITHUB/STREAMLIT CLOUD

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time
from datetime import datetime
import os

# ====================== CONFIG ======================
# تحذير: الكاميرات مش هتشتغل على Streamlit Cloud
DEMO_MODE = True  # غير لـ False لو شغال locally

st.set_page_config(page_title="Driver Safety Monitor", layout="wide")

# ====================== DOWNLOAD MODEL (FIXED) ======================
@st.cache_resource
def download_model():
    """تحميل الموديل - مع fallback للـ demo mode"""
    filename = "effnet.pth"
    
    if os.path.exists(filename):
        st.success(f"✅ Model found: {filename}")
        return filename
    
    try:
        # جرب تحمل من Google Drive
        import gdown
        url = "https://drive.google.com/uc?id=1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM"
        with st.spinner("Downloading model... (1-2 min)"):
            gdown.download(url, filename, quiet=False)
        st.success("✅ Model downloaded!")
        return filename
    except Exception as e:
        st.warning(f"⚠️ Could not download model: {e}")
        st.info("Running in DEMO mode without distraction detection")
        return None

model_path = download_model()

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_distraction_model(path):
    if path is None or not os.path.exists(path):
        return None, None
    
    try:
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
        state = torch.load(path, map_location=device)
        state_dict = state.get("model", state)
        
        # Fix key names
        fixed_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("net.", "").replace("module.", "")
            fixed_state[new_k] = v
        
        model.load_state_dict(fixed_state, strict=False)
        model.eval()
        st.success("✅ Distraction model loaded!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

distraction_model, device = load_distraction_model(model_path)

# ====================== DETECTION FUNCTIONS ======================
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
    if model is None:
        # Demo mode
        cv2.putText(frame, "Demo Mode - No Model", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame, False, "Demo"
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item()
        label = LABELS[idx]
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.putText(frame, f"{label} ({conf:.1%})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, idx != 0, label
    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)[:20]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, False, "Error"

def detect_drowsiness(frame):
    try:
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
                eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                if eye_region.size > 0:
                    variance = np.var(eye_region)
                    if variance < 50:
                        closed += 1
                        cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "OPEN", (x+ex, y+ey-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        is_drowsy = (closed >= 2) and (eyes_detected > 0)
        return frame, is_drowsy
    except Exception as e:
        cv2.putText(frame, f"Detection Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, False

# ====================== DEMO FRAMES ======================
def create_demo_frame(text, color=(100, 100, 100)):
    """إنشاء frame تجريبي"""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = color
    cv2.putText(frame, text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    return frame

# ====================== UI DASHBOARD ======================
st.title("🚗 Driver Safety Monitor")
st.markdown("---")

if DEMO_MODE:
    st.warning("⚠️ **DEMO MODE** - Cameras not available on Streamlit Cloud. Upload images to test!")

# File uploaders for testing
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    front_upload = st.file_uploader("Upload Front Camera Image (for drowsiness)", 
                                   type=['jpg', 'jpeg', 'png'], key="front")
with col_upload2:
    side_upload = st.file_uploader("Upload Side Camera Image (for distraction)", 
                                  type=['jpg', 'jpeg', 'png'], key="side")

# Display columns
col1, col2, col3 = st.columns([1, 1, 1])

# Process uploaded images or show demo
with col1:
    st.subheader("📹 Front Camera")
    if front_upload:
        file_bytes = np.asarray(bytearray(front_upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (320, 240))
        annotated, is_drowsy = detect_drowsiness(frame.copy())
        st.image(annotated, channels="BGR", use_container_width=True)
        if is_drowsy:
            st.error("⚠️ DROWSINESS DETECTED!")
        else:
            st.success("✅ Driver Alert")
    else:
        demo = create_demo_frame("Upload Image", (50, 50, 100))
        st.image(demo, channels="BGR", use_container_width=True)

with col2:
    st.subheader("📹 Side Camera")
    if side_upload:
        file_bytes = np.asarray(bytearray(side_upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (320, 240))
        annotated, is_dist, label = detect_distraction(frame.copy(), distraction_model, device)
        st.image(annotated, channels="BGR", use_container_width=True)
        if is_dist:
            st.error(f"⚠️ DISTRACTION: {label}")
        else:
            st.success("✅ Safe Driving")
    else:
        demo = create_demo_frame("Upload Image", (50, 100, 50))
        st.image(demo, channels="BGR", use_container_width=True)

with col3:
    st.subheader("📊 Dashboard")
    if front_upload or side_upload:
        st.metric("Status", "Analyzing...")
        if front_upload and side_upload:
            st.success("✅ Both cameras active")
    else:
        st.info("💡 Upload images to test the system")
        st.markdown("""
        **How to use:**
        1. Upload a front-facing driver image
        2. Upload a side-view driver image
        3. System will detect drowsiness & distraction
        """)

# Instructions
st.markdown("---")
st.markdown("""
### 🔧 **For Local Development:**
1. Install requirements: `pip install streamlit opencv-python torch torchvision pillow gdown`
2. Change `DEMO_MODE = False` in code
3. Run: `streamlit run integrated_dashboard.py`
4. Connect cameras (0 and 1)

### ☁️ **On Streamlit Cloud:**
- Use file upload feature above
- Model downloads automatically (if available)
- No camera access needed
""")
