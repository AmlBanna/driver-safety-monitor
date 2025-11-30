"""
Driver Safety Monitor Pro - Integrated System
دمج 4 موديلز: EfficientNet + Driver Behavior CNN + Drowsiness + Combined Analysis
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
from datetime import datetime
import os
import tempfile

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Driver Safety Monitor Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS للواجهة الاحترافية
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
    .alert-danger {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .alert-success {
        background-color: #00C851;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ====================== MODELS DOWNLOAD ======================
@st.cache_resource
def download_model(filename, gdrive_id, description):
    """تحميل الموديل من Google Drive"""
    if os.path.exists(filename):
        return filename
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        with st.spinner(f"⏳ Downloading {description}..."):
            gdown.download(url, filename, quiet=False)
        st.success(f"✅ {description} downloaded!")
        return filename
    except Exception as e:
        st.warning(f"⚠️ Could not download {description}: {e}")
        return None

# تحميل الموديلات
MODELS_CONFIG = {
    "effnet": {
        "filename": "effnet.pth",
        "gdrive_id": "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM",
        "description": "EfficientNet Distraction Model"
    },
    "driver_behavior": {
        "filename": "driver_behavior.pth",
        "gdrive_id": "YOUR_DRIVER_BEHAVIOR_MODEL_ID",  # ضع ID الموديل من Kaggle
        "description": "Driver Behavior CNN Model"
    }
}

model_paths = {}
for key, config in MODELS_CONFIG.items():
    model_paths[key] = download_model(
        config["filename"],
        config["gdrive_id"],
        config["description"]
    )

# ====================== MODEL ARCHITECTURES ======================
class EfficientNet_B0(nn.Module):
    """EfficientNet للكشف عن التشتيت"""
    def __init__(self, num_classes=10): 
        super().__init__()
        self.net = models.efficientnet_b0(pretrained=False)
        self.net.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x): 
        return self.net(x)

class DriverBehaviorCNN(nn.Module):
    """CNN للكشف عن سلوكيات القيادة"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_all_models():
    """تحميل جميع الموديلات"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict = {}
    
    # 1. EfficientNet Model
    if model_paths.get("effnet") and os.path.exists(model_paths["effnet"]):
        try:
            model = EfficientNet_B0(num_classes=10).to(device)
            state = torch.load(model_paths["effnet"], map_location=device)
            state_dict = state.get("model", state)
            
            fixed_state = {}
            for k, v in state_dict.items():
                new_k = k.replace("net.", "").replace("module.", "")
                fixed_state[new_k] = v
            
            model.load_state_dict(fixed_state, strict=False)
            model.eval()
            models_dict["effnet"] = model
            st.success("✅ EfficientNet loaded!")
        except Exception as e:
            st.error(f"❌ EfficientNet error: {e}")
    
    # 2. Driver Behavior CNN Model
    if model_paths.get("driver_behavior") and os.path.exists(model_paths["driver_behavior"]):
        try:
            model = DriverBehaviorCNN(num_classes=5).to(device)
            state = torch.load(model_paths["driver_behavior"], map_location=device)
            model.load_state_dict(state, strict=False)
            model.eval()
            models_dict["behavior_cnn"] = model
            st.success("✅ Driver Behavior CNN loaded!")
        except Exception as e:
            st.error(f"❌ Behavior CNN error: {e}")
    
    return models_dict, device

models_dict, device = load_all_models()

# ====================== LABELS & TRANSFORMS ======================
DISTRACTION_LABELS = {
    0: "Safe Driving", 1: "Texting (Right)", 2: "Talking (Right)",
    3: "Texting (Left)", 4: "Talking (Left)", 5: "Radio",
    6: "Drinking", 7: "Reaching", 8: "Hair/Makeup", 9: "Passenger"
}

BEHAVIOR_LABELS = {
    0: "Normal Driving",
    1: "Aggressive Driving",
    2: "Distracted Driving",
    3: "Drowsy Driving",
    4: "Drunk Driving"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== DETECTION FUNCTIONS ======================
def detect_distraction_effnet(frame, model, device):
    """كشف التشتيت باستخدام EfficientNet"""
    if model is None:
        return frame, False, "Model Not Loaded", 0.0
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item()
        
        label = DISTRACTION_LABELS[idx]
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        
        cv2.putText(frame, f"Distraction: {label}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Confidence: {conf:.1%}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, idx != 0, label, conf
    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, False, "Error", 0.0

def detect_behavior_cnn(frame, model, device):
    """كشف السلوك باستخدام CNN"""
    if model is None:
        return frame, False, "Model Not Loaded", 0.0
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item()
        
        label = BEHAVIOR_LABELS[idx]
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        
        cv2.putText(frame, f"Behavior: {label}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Confidence: {conf:.1%}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, idx != 0, label, conf
    except Exception as e:
        return frame, False, "Error", 0.0

def detect_drowsiness(frame):
    """كشف النعاس باستخدام OpenCV"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        closed_eyes = 0
        total_eyes = 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
            total_eyes += len(eyes)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                
                if eye_region.size > 0:
                    variance = np.var(eye_region)
                    if variance < 50:
                        closed_eyes += 1
                        cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        cv2.putText(frame, "OPEN", (x+ex, y+ey-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        is_drowsy = (closed_eyes >= 2) and (total_eyes > 0)
        confidence = (closed_eyes / max(total_eyes, 1)) if total_eyes > 0 else 0.0
        
        status = "DROWSY!" if is_drowsy else "ALERT"
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"Drowsiness: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, is_drowsy, confidence
    except Exception as e:
        return frame, False, 0.0

def combined_analysis(front_result, side_result):
    """تحليل مدمج من الكاميرتين"""
    drowsy, drowsy_conf = front_result
    distracted, dist_label, dist_conf = side_result[:3]
    behavior_risk, behavior_label, behavior_conf = side_result[3:] if len(side_result) > 3 else (False, "N/A", 0.0)
    
    # حساب مستوى الخطر الإجمالي
    risk_score = 0
    alerts = []
    
    if drowsy:
        risk_score += 40
        alerts.append(f"⚠️ النعاس ({drowsy_conf:.0%})")
    
    if distracted:
        risk_score += 35
        alerts.append(f"⚠️ التشتيت: {dist_label} ({dist_conf:.0%})")
    
    if behavior_risk:
        risk_score += 25
        alerts.append(f"⚠️ سلوك خطر: {behavior_label} ({behavior_conf:.0%})")
    
    # تحديد مستوى الخطر
    if risk_score >= 60:
        status = "🔴 خطر عالي"
        alert_class = "alert-danger"
    elif risk_score >= 30:
        status = "🟡 تحذير"
        alert_class = "alert-warning"
    else:
        status = "🟢 آمن"
        alert_class = "alert-success"
    
    return {
        "status": status,
        "risk_score": risk_score,
        "alerts": alerts,
        "alert_class": alert_class,
        "drowsy": drowsy,
        "distracted": distracted,
        "behavior_risk": behavior_risk
    }

# ====================== VIDEO PROCESSING ======================
def process_video(video_path, progress_bar, status_text):
    """معالجة الفيديو"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    results = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # معالجة كل 5 فريمات لتوفير الوقت
        if frame_idx % 5 == 0:
            # كاميرا أمامية (نصف الإطار الأيسر)
            h, w = frame.shape[:2]
            front_frame = frame[:, :w//2]
            side_frame = frame[:, w//2:]
            
            # التحليل
            _, drowsy, drowsy_conf = detect_drowsiness(front_frame.copy())
            
            _, dist, dist_label, dist_conf = detect_distraction_effnet(
                side_frame.copy(),
                models_dict.get("effnet"),
                device
            )
            
            _, behavior, behavior_label, behavior_conf = detect_behavior_cnn(
                side_frame.copy(),
                models_dict.get("behavior_cnn"),
                device
            )
            
            analysis = combined_analysis(
                (drowsy, drowsy_conf),
                (dist, dist_label, dist_conf, behavior, behavior_label, behavior_conf)
            )
            
            results.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "analysis": analysis
            })
        
        frame_idx += 1
        progress = frame_idx / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx}/{frame_count}")
    
    cap.release()
    return results

# ====================== MAIN UI ======================
st.markdown('<div class="main-header"><h1>🚗 Driver Safety Monitor Pro</h1><p>نظام متكامل لمراقبة سلامة القيادة</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    input_mode = st.radio(
        "اختر طريقة الإدخال:",
        ["📸 رفع صور", "🎥 رفع فيديو", "📹 بث مباشر"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 معلومات النظام")
    st.info(f"""
    **الموديلات المحملة:**
    - ✅ EfficientNet: {'نعم' if 'effnet' in models_dict else 'لا'}
    - ✅ Behavior CNN: {'نعم' if 'behavior_cnn' in models_dict else 'لا'}
    - ✅ Drowsiness: نعم (OpenCV)
    """)

# Main Content
if input_mode == "📸 رفع صور":
    st.subheader("📸 رفع الصور للتحليل")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📹 الكاميرا الأمامية")
        front_upload = st.file_uploader(
            "ارفع صورة السائق من الأمام (للكشف عن النعاس)",
            type=['jpg', 'jpeg', 'png'],
            key="front_img"
        )
    
    with col2:
        st.markdown("#### 📹 الكاميرا الجانبية")
        side_upload = st.file_uploader(
            "ارفع صورة السائق من الجانب (للكشف عن التشتت والسلوك)",
            type=['jpg', 'jpeg', 'png'],
            key="side_img"
        )
    
    if st.button("🔍 تحليل الصور", use_container_width=True):
        if front_upload and side_upload:
            with st.spinner("⏳ جاري التحليل..."):
                # معالجة الصورة الأمامية
                front_bytes = np.asarray(bytearray(front_upload.read()), dtype=np.uint8)
                front_frame = cv2.imdecode(front_bytes, cv2.IMREAD_COLOR)
                front_frame = cv2.resize(front_frame, (640, 480))
                
                # معالجة الصورة الجانبية
                side_bytes = np.asarray(bytearray(side_upload.read()), dtype=np.uint8)
                side_frame = cv2.imdecode(side_bytes, cv2.IMREAD_COLOR)
                side_frame = cv2.resize(side_frame, (640, 480))
                
                # التحليل
                front_annotated, drowsy, drowsy_conf = detect_drowsiness(front_frame.copy())
                
                side_annotated1, dist, dist_label, dist_conf = detect_distraction_effnet(
                    side_frame.copy(),
                    models_dict.get("effnet"),
                    device
                )
                
                side_annotated2, behavior, behavior_label, behavior_conf = detect_behavior_cnn(
                    side_annotated1.copy(),
                    models_dict.get("behavior_cnn"),
                    device
                )
                
                # التحليل المدمج
                analysis = combined_analysis(
                    (drowsy, drowsy_conf),
                    (dist, dist_label, dist_conf, behavior, behavior_label, behavior_conf)
                )
                
                # عرض النتائج
                st.markdown("---")
                st.markdown(f'<div class="{analysis["alert_class"]}">{analysis["status"]}</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(front_annotated, channels="BGR", caption="الكاميرا الأمامية", use_container_width=True)
                
                with col2:
                    st.image(side_annotated2, channels="BGR", caption="الكاميرا الجانبية", use_container_width=True)
                
                with col3:
                    st.markdown("### 📊 التحليل التفصيلي")
                    st.metric("مستوى الخطر", f"{analysis['risk_score']}%")
                    
                    if analysis['alerts']:
                        st.markdown("**⚠️ التنبيهات:**")
                        for alert in analysis['alerts']:
                            st.warning(alert)
                    else:
                        st.success("✅ لا توجد مخاطر")
        else:
            st.error("⚠️ يرجى رفع الصورتين أولاً!")

elif input_mode == "🎥 رفع فيديو":
    st.subheader("🎥 رفع الفيديو للتحليل")
    
    video_upload = st.file_uploader(
        "ارفع فيديو القيادة",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video"
    )
    
    if video_upload and st.button("🔍 تحليل الفيديو", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_upload.read())
            video_path = tmp_file.name
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = process_video(video_path, progress_bar, status_text)
        
        st.success("✅ اكتمل التحليل!")
        
        # عرض ملخص النتائج
        st.markdown("### 📊 ملخص التحليل")
        
        total_frames = len(results)
        risky_frames = sum(1 for r in results if r['analysis']['risk_score'] >= 30)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("إجمالي الإطارات المحللة", total_frames)
        col2.metric("إطارات بها مخاطر", risky_frames)
        col3.metric("نسبة الأمان", f"{100 - (risky_frames/total_frames*100):.1f}%")
        
        # عرض التفاصيل
        if st.checkbox("عرض التفاصيل الكاملة"):
            for result in results[-10:]:  # آخر 10 نتائج
                time_str = f"{int(result['time']//60):02d}:{int(result['time']%60):02d}"
                st.markdown(f"**الوقت {time_str}** - {result['analysis']['status']}")
                if result['analysis']['alerts']:
                    for alert in result['analysis']['alerts']:
                        st.write(f"  {alert}")
        
        os.unlink(video_path)

else:  # بث مباشر
    st.subheader("📹 البث المباشر")
    
    st.info("💡 لاستخدام البث المباشر، تأكد من تشغيل التطبيق محلياً (لا يعمل على Streamlit Cloud)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        front_cam_id = st.number_input("رقم الكاميرا الأمامية", value=0, min_value=0, max_value=10)
    
    with col2:
        side_cam_id = st.number_input("رقم الكاميرا الجانبية", value=1, min_value=0, max_value=10)
    
    if st.button("▶️ ابدأ البث", use_container_width=True):
        st.warning("⚠️ البث المباشر يتطلب تشغيل محلي. استخدم رفع الصور/الفيديو للاختبار على Streamlit Cloud")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Driver Safety Monitor Pro v2.0 | Powered by PyTorch & OpenCV</p>
    <p>For support: contact@example.com</p>
</div>
""", unsafe_allow_html=True)
