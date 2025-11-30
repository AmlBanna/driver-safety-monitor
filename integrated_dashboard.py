"""
Driver Safety Monitor Pro - Complete System
4 Models: EfficientNet + 3 Behavior Models (CNN, VGG, ResNet) + Drowsiness
Full Support: Images + Videos + Live Stream
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import tempfile
import os

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Driver Safety Monitor Pro",
    page_icon="🚗",
    layout="wide"
)

# CSS احترافي
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .alert-danger {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        animation: pulse 1.5s infinite;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,136,0,0.3);
    }
    .alert-success {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,200,81,0.3);
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .video-container {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ====================== MODELS CONFIG ======================
MODELS_CONFIG = {
    "effnet": {
        "filename": "effnet.pth",
        "gdrive_id": "1GvL1w3UmOeMRISBWdKGGeeKNR2oH0MZM",
        "description": "EfficientNet Distraction Model"
    },
    "behavior_cnn": {
        "filename": "behavior_cnn.pth",
        "gdrive_id": "YOUR_CNN_MODEL_ID",  # ضع الـ ID هنا
        "description": "Behavior Detection CNN"
    },
    "behavior_vgg": {
        "filename": "behavior_vgg.pth",
        "gdrive_id": "YOUR_VGG_MODEL_ID",  # ضع الـ ID هنا
        "description": "Behavior Detection VGG16"
    },
    "behavior_resnet": {
        "filename": "behavior_resnet.pth",
        "gdrive_id": "YOUR_RESNET_MODEL_ID",  # ضع الـ ID هنا
        "description": "Behavior Detection ResNet"
    }
}

# ====================== DOWNLOAD MODELS ======================
@st.cache_resource
def download_model(filename, gdrive_id, description):
    """تحميل الموديل من Google Drive"""
    if os.path.exists(filename):
        return filename
    
    if gdrive_id.startswith("YOUR_"):
        st.warning(f"⚠️ {description}: ID not configured yet")
        return None
    
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

# تحميل جميع الموديلات
model_paths = {}
with st.spinner("🔄 Loading models..."):
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
        self.net = models.efficientnet_b0(weights=None)
        self.net.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x): 
        return self.net(x)

class BehaviorCNN(nn.Module):
    """CNN للكشف عن سلوكيات القيادة"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BehaviorVGG16(nn.Module):
    """VGG16 للكشف عن سلوكيات القيادة"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.vgg = models.vgg16(weights=None)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.vgg(x)

class BehaviorResNet(nn.Module):
    """ResNet للكشف عن سلوكيات القيادة"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_all_models():
    """تحميل جميع الموديلات الأربعة"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = {}
    
    # 1. EfficientNet for Distraction
    if model_paths.get("effnet") and os.path.exists(model_paths["effnet"]):
        try:
            model = EfficientNet_B0(num_classes=10).to(device)
            state = torch.load(model_paths["effnet"], map_location=device, weights_only=False)
            state_dict = state.get("model", state)
            
            fixed_state = {}
            for k, v in state_dict.items():
                new_k = k.replace("net.", "").replace("module.", "")
                fixed_state[new_k] = v
            
            model.load_state_dict(fixed_state, strict=False)
            model.eval()
            loaded_models["effnet"] = model
            st.success("✅ EfficientNet Model Loaded!")
        except Exception as e:
            st.error(f"❌ EfficientNet Error: {e}")
    
    # 2. Behavior CNN
    if model_paths.get("behavior_cnn") and os.path.exists(model_paths["behavior_cnn"]):
        try:
            model = BehaviorCNN(num_classes=5).to(device)
            state = torch.load(model_paths["behavior_cnn"], map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            model.eval()
            loaded_models["behavior_cnn"] = model
            st.success("✅ Behavior CNN Model Loaded!")
        except Exception as e:
            st.error(f"❌ Behavior CNN Error: {e}")
    
    # 3. Behavior VGG16
    if model_paths.get("behavior_vgg") and os.path.exists(model_paths["behavior_vgg"]):
        try:
            model = BehaviorVGG16(num_classes=5).to(device)
            state = torch.load(model_paths["behavior_vgg"], map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            model.eval()
            loaded_models["behavior_vgg"] = model
            st.success("✅ Behavior VGG16 Model Loaded!")
        except Exception as e:
            st.error(f"❌ Behavior VGG Error: {e}")
    
    # 4. Behavior ResNet
    if model_paths.get("behavior_resnet") and os.path.exists(model_paths["behavior_resnet"]):
        try:
            model = BehaviorResNet(num_classes=5).to(device)
            state = torch.load(model_paths["behavior_resnet"], map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            model.eval()
            loaded_models["behavior_resnet"] = model
            st.success("✅ Behavior ResNet Model Loaded!")
        except Exception as e:
            st.error(f"❌ Behavior ResNet Error: {e}")
    
    return loaded_models, device

models_dict, device = load_all_models()

# ====================== LABELS ======================
DISTRACTION_LABELS = {
    0: "Safe Driving", 1: "Texting Right", 2: "Talking Right",
    3: "Texting Left", 4: "Talking Left", 5: "Operating Radio",
    6: "Drinking", 7: "Reaching Behind", 8: "Hair/Makeup", 9: "Talking to Passenger"
}

BEHAVIOR_LABELS = {
    0: "Normal Driving",
    1: "Aggressive Driving", 
    2: "Distracted Driving",
    3: "Drowsy Driving",
    4: "Drunk Driving"
}

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== DETECTION FUNCTIONS ======================
def detect_distraction_effnet(frame, model, device):
    """كشف التشتيت - EfficientNet"""
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
        
        label = DISTRACTION_LABELS.get(idx, "Unknown")
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        
        cv2.putText(frame, f"EfficientNet: {label}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Conf: {conf:.1%}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, idx != 0, label, conf
    except Exception as e:
        return frame, False, f"Error: {e}", 0.0

def detect_behavior_model(frame, model, device, model_name):
    """كشف السلوك - CNN/VGG/ResNet"""
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
        
        label = BEHAVIOR_LABELS.get(idx, "Unknown")
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        
        y_pos = 90 if "CNN" in model_name else (120 if "VGG" in model_name else 150)
        cv2.putText(frame, f"{model_name}: {label}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{conf:.1%}", (10, y_pos + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame, idx != 0, label, conf
    except Exception as e:
        return frame, False, f"Error", 0.0

def detect_drowsiness(frame):
    """كشف النعاس - OpenCV Cascade"""
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
            total_eyes += len(eyes)
            
            for (ex, ey, ew, eh) in eyes:
                eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                
                if eye_region.size > 0:
                    variance = np.var(eye_region)
                    if variance < 50:
                        closed_eyes += 1
                        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
                        cv2.putText(frame, "CLOSED", (x+ex, y+ey-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                        cv2.putText(frame, "OPEN", (x+ex, y+ey-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        is_drowsy = (closed_eyes >= 2) and (total_eyes > 0)
        confidence = (closed_eyes / max(total_eyes, 1)) if total_eyes > 0 else 0.0
        
        status = "⚠️ DROWSY!" if is_drowsy else "✓ ALERT"
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Eyes: {total_eyes} (Closed: {closed_eyes})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, is_drowsy, confidence
    except Exception as e:
        cv2.putText(frame, f"Detection Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame, False, 0.0

def combined_analysis(drowsy_result, side_results):
    """التحليل المدمج من جميع الموديلات"""
    drowsy, drowsy_conf = drowsy_result
    
    # استخراج النتائج من الموديلات الأربعة
    distraction_risk = side_results.get("distraction", (False, "N/A", 0.0))
    cnn_risk = side_results.get("cnn", (False, "N/A", 0.0))
    vgg_risk = side_results.get("vgg", (False, "N/A", 0.0))
    resnet_risk = side_results.get("resnet", (False, "N/A", 0.0))
    
    risk_score = 0
    alerts = []
    
    # النعاس (وزن 35%)
    if drowsy:
        risk_score += 35
        alerts.append(f"⚠️ نعاس مكتشف ({drowsy_conf:.0%})")
    
    # التشتيت - EfficientNet (وزن 25%)
    if distraction_risk[0]:
        risk_score += 25
        alerts.append(f"⚠️ تشتيت: {distraction_risk[1]} ({distraction_risk[2]:.0%})")
    
    # السلوك - CNN (وزن 15%)
    if cnn_risk[0]:
        risk_score += 15
        alerts.append(f"⚠️ CNN: {cnn_risk[1]} ({cnn_risk[2]:.0%})")
    
    # السلوك - VGG (وزن 13%)
    if vgg_risk[0]:
        risk_score += 13
        alerts.append(f"⚠️ VGG: {vgg_risk[1]} ({vgg_risk[2]:.0%})")
    
    # السلوك - ResNet (وزن 12%)
    if resnet_risk[0]:
        risk_score += 12
        alerts.append(f"⚠️ ResNet: {resnet_risk[1]} ({resnet_risk[2]:.0%})")
    
    # تحديد مستوى الخطر
    if risk_score >= 50:
        status = "🔴 خطر عالي جداً - توقف فوراً!"
        alert_class = "alert-danger"
    elif risk_score >= 30:
        status = "🟡 تحذير - انتبه للطريق"
        alert_class = "alert-warning"
    else:
        status = "🟢 قيادة آمنة"
        alert_class = "alert-success"
    
    return {
        "status": status,
        "risk_score": risk_score,
        "alerts": alerts,
        "alert_class": alert_class,
        "details": {
            "drowsy": drowsy,
            "distraction": distraction_risk,
            "cnn": cnn_risk,
            "vgg": vgg_risk,
            "resnet": resnet_risk
        }
    }

# ====================== VIDEO PROCESSING ======================
def process_video_file(video_path, progress_bar, status_text, result_container):
    """معالجة ملف فيديو كامل"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ لا يمكن فتح الفيديو!")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
    
    results = []
    frame_idx = 0
    process_every = 10  # معالجة كل 10 إطارات لتوفير الوقت
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % process_every == 0:
            try:
                # تقسيم الإطار: نصف أمامي + نصف جانبي
                h, w = frame.shape[:2]
                front_frame = cv2.resize(frame[:, :w//2], (640, 480))
                side_frame = cv2.resize(frame[:, w//2:], (640, 480))
                
                # تحليل الكاميرا الأمامية (النعاس)
                _, drowsy, drowsy_conf = detect_drowsiness(front_frame.copy())
                
                # تحليل الكاميرا الجانبية (الموديلات الأربعة)
                side_results = {}
                
                # EfficientNet
                _, dist_risk, dist_label, dist_conf = detect_distraction_effnet(
                    side_frame.copy(), models_dict.get("effnet"), device
                )
                side_results["distraction"] = (dist_risk, dist_label, dist_conf)
                
                # Behavior CNN
                _, cnn_risk, cnn_label, cnn_conf = detect_behavior_model(
                    side_frame.copy(), models_dict.get("behavior_cnn"), device, "CNN"
                )
                side_results["cnn"] = (cnn_risk, cnn_label, cnn_conf)
                
                # Behavior VGG
                _, vgg_risk, vgg_label, vgg_conf = detect_behavior_model(
                    side_frame.copy(), models_dict.get("behavior_vgg"), device, "VGG"
                )
                side_results["vgg"] = (vgg_risk, vgg_label, vgg_conf)
                
                # Behavior ResNet
                _, resnet_risk, resnet_label, resnet_conf = detect_behavior_model(
                    side_frame.copy(), models_dict.get("behavior_resnet"), device, "ResNet"
                )
                side_results["resnet"] = (resnet_risk, resnet_label, resnet_conf)
                
                # التحليل المدمج
                analysis = combined_analysis((drowsy, drowsy_conf), side_results)
                
                results.append({
                    "frame": frame_idx,
                    "time": frame_idx / fps,
                    "analysis": analysis
                })
                
                # عرض تحديث مباشر
                if frame_idx % (process_every * 10) == 0:
                    with result_container:
                        st.write(f"⏱️ معالجة الثانية {frame_idx // fps} - {analysis['status']}")
            
            except Exception as e:
                st.warning(f"⚠️ خطأ في الإطار {frame_idx}: {e}")
        
        frame_idx += 1
        progress = min(frame_idx / frame_count, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"معالجة الإطار {frame_idx}/{frame_count}")
    
    cap.release()
    return results

# ====================== LIVE STREAM PROCESSING ======================
def process_live_stream(front_cam_id, side_cam_id, stop_button_placeholder):
    """معالجة البث المباشر من كاميرتين"""
    front_cap = cv2.VideoCapture(front_cam_id)
    side_cap = cv2.VideoCapture(side_cam_id)
    
    if not front_cap.isOpened() or not side_cap.isOpened():
        st.error("❌ لا يمكن فتح الكاميرات! تأكد من توصيلها بشكل صحيح")
        return
    
    # إعداد واجهة البث
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📹 الكاميرا الأمامية")
        front_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📹 الكاميرا الجانبية")
        side_placeholder = st.empty()
    
    with col3:
        st.markdown("### 📊 التحليل المباشر")
        analysis_placeholder = st.empty()
    
    # زر الإيقاف
    stop_clicked = stop_button_placeholder.button("⏹️ إيقاف البث", use_container_width=True, type="primary")
    
    frame_count = 0
    
    try:
        while not stop_clicked:
            # قراءة الإطارات
            ret1, front_frame = front_cap.read()
            ret2, side_frame = side_cap.read()
            
            if not ret1 or not ret2:
                st.warning("⚠️ فقد الاتصال بالكاميرا")
                break
            
            # تغيير الحجم
            front_frame = cv2.resize(front_frame, (640, 480))
            side_frame = cv2.resize(side_frame, (640, 480))
            
            # التحليل (كل 5 إطارات لتحسين الأداء)
            if frame_count % 5 == 0:
                # تحليل النعاس
                front_annotated, drowsy, drowsy_conf = detect_drowsiness(front_frame.copy())
                
                # تحليل الكاميرا الجانبية
                side_annotated = side_frame.copy()
                side_results = {}
                
                # EfficientNet
                side_annotated, dist_risk, dist_label, dist_conf = detect_distraction_effnet(
                    side_annotated, models_dict.get("effnet"), device
                )
                side_results["distraction"] = (dist_risk, dist_label, dist_conf)
                
                # Behavior Models
                side_annotated, cnn_risk, cnn_label, cnn_conf = detect_behavior_model(
                    side_annotated, models_dict.get("behavior_cnn"), device, "CNN"
                )
                side_results["cnn"] = (cnn_risk, cnn_label, cnn_conf)
                
                side_annotated, vgg_risk, vgg_label, vgg_conf = detect_behavior_model(
                    side_annotated, models_dict.get("behavior_vgg"), device, "VGG"
                )
                side_results["vgg"] = (vgg_risk, vgg_label, vgg_conf)
                
                side_annotated, resnet_risk, resnet_label, resnet_conf = detect_behavior_model(
                    side_annotated, models_dict.get("behavior_resnet"), device, "ResNet"
                )
                side_results["resnet"] = (resnet_risk, resnet_label, resnet_conf)
                
                # التحليل المدمج
                analysis = combined_analysis((drowsy, drowsy_conf), side_results)
                
                # عرض النتائج
                front_placeholder.image(front_annotated, channels="BGR", use_container_width=True)
                side_placeholder.image(side_annotated, channels="BGR", use_container_width=True)
                
                with analysis_placeholder.container():
                    st.markdown(f'<div class="{analysis["alert_class"]}">{analysis["status"]}</div>', 
                               unsafe_allow_html=True)
                    st.metric("مستوى الخطر", f"{analysis['risk_score']}%")
                    
                    if analysis['alerts']:
                        st.markdown("**⚠️ التنبيهات:**")
                        for alert in analysis['alerts']:
                            st.warning(alert)
            
            frame_count += 1
            
            # تحديث حالة زر الإيقاف
            stop_clicked = stop_button_placeholder.button("⏹️ إيقاف البث", 
                                                          use_container_width=True, 
                                                          type="primary",
                                                          key=f"stop_{frame_count}")
    
    finally:
        front_cap.release()
        side_cap.release()
        st.success("✅ تم إيقاف البث بنجاح")

# ====================== MAIN UI ======================
st.markdown('<div class="main-header"><h1>🚗 Driver Safety Monitor Pro</h1><p>نظام متكامل لمراقبة السلامة - 4 موديلات ذكاء اصطناعي</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات والتحكم")
    
    input_mode = st.radio(
        "اختر طريقة الإدخال:",
        ["📸 رفع صور", "🎥 رفع فيديو", "📹 بث مباشر"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 حالة الموديلات")
    
    models_status = {
        "EfficientNet": "effnet" in models_dict,
        "Behavior CNN": "behavior_cnn" in models_dict,
        "Behavior VGG16": "behavior_vgg" in models_dict,
        "Behavior ResNet": "behavior_resnet" in models_dict,
        "Drowsiness (OpenCV)": True
    }
    
    for model_name, status in models_status.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {model_name}")
    
    total_models = sum(models_status.values())
    st.metric("الموديلات الجاهزة", f"{total_models}/5")
    
    st.markdown("---")
    st.info("💡 **نصيحة:** للحصول على أفضل النتائج، تأكد من إضاءة جيدة ووضوح الكاميرا")

# Main Content
st.markdown("---")

if input_mode == "📸 رفع صور":
    st.subheader("📸 تحليل الصور")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("#### 📹 الكاميرا الأمامية")
        front_upload = st.file_uploader(
            "ارفع صورة السائق من الأمام",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="front_img",
            help="للكشف عن النعاس"
        )
    
    with col_upload2:
        st.markdown("#### 📹 الكاميرا الجانبية")
        side_upload = st.file_uploader(
            "ارفع صورة السائق من الجانب",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="side_img",
            help="للكشف عن التشتت والسلوك"
        )
    
    if st.button("🔍 تحليل الصور الآن", use_container_width=True, type="primary"):
        if front_upload and side_upload:
            with st.spinner("⏳ جاري التحليل باستخدام 5 موديلات..."):
                try:
                    # قراءة الصور
                    front_bytes = np.asarray(bytearray(front_upload.read()), dtype=np.uint8)
                    front_frame = cv2.imdecode(front_bytes, cv2.IMREAD_COLOR)
                    front_frame = cv2.resize(front_frame, (640, 480))
                    
                    side_bytes = np.asarray(bytearray(side_upload.read()), dtype=np.uint8)
                    side_frame = cv2.imdecode(side_bytes, cv2.IMREAD_COLOR)
                    side_frame = cv2.resize(side_frame, (640, 480))
                    
                    # تحليل الكاميرا الأمامية
                    front_annotated, drowsy, drowsy_conf = detect_drowsiness(front_frame.copy())
                    
                    # تحليل الكاميرا الجانبية (4 موديلات)
                    side_annotated = side_frame.copy()
                    side_results = {}
                    
                    # 1. EfficientNet
                    side_annotated, dist_risk, dist_label, dist_conf = detect_distraction_effnet(
                        side_annotated, models_dict.get("effnet"), device
                    )
                    side_results["distraction"] = (dist_risk, dist_label, dist_conf)
                    
                    # 2. Behavior CNN
                    side_annotated, cnn_risk, cnn_label, cnn_conf = detect_behavior_model(
                        side_annotated, models_dict.get("behavior_cnn"), device, "CNN"
                    )
                    side_results["cnn"] = (cnn_risk, cnn_label, cnn_conf)
                    
                    # 3. Behavior VGG
                    side_annotated, vgg_risk, vgg_label, vgg_conf = detect_behavior_model(
                        side_annotated, models_dict.get("behavior_vgg"), device, "VGG"
                    )
                    side_results["vgg"] = (vgg_risk, vgg_label, vgg_conf)
                    
                    # 4. Behavior ResNet
                    side_annotated, resnet_risk, resnet_label, resnet_conf = detect_behavior_model(
                        side_annotated, models_dict.get("behavior_resnet"), device, "ResNet"
                    )
                    side_results["resnet"] = (resnet_risk, resnet_label, resnet_conf)
                    
                    # التحليل المدمج
                    analysis = combined_analysis((drowsy, drowsy_conf), side_results)
                    
                    # عرض النتائج
                    st.markdown("---")
                    st.markdown(f'<div class="{analysis["alert_class"]}">{analysis["status"]}</div>', 
                               unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        st.image(front_annotated, channels="BGR", caption="📹 الكاميرا الأمامية", 
                                use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        st.image(side_annotated, channels="BGR", caption="📹 الكاميرا الجانبية", 
                                use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("### 📊 التحليل التفصيلي")
                        st.metric("مستوى الخطر الإجمالي", f"{analysis['risk_score']}%",
                                 delta=f"{analysis['risk_score']-50}%" if analysis['risk_score'] > 50 else None,
                                 delta_color="inverse")
                        
                        st.markdown("---")
                        
                        if analysis['alerts']:
                            st.markdown("**⚠️ التنبيهات المكتشفة:**")
                            for i, alert in enumerate(analysis['alerts'], 1):
                                st.warning(f"{i}. {alert}")
                        else:
                            st.success("✅ لا توجد مخاطر - قيادة آمنة!")
                        
                        st.markdown("---")
                        st.markdown("**📋 تفاصيل كل موديل:**")
                        
                        details = analysis['details']
                        st.write(f"🔹 نعاس: {'نعم ⚠️' if details['drowsy'] else 'لا ✓'}")
                        st.write(f"🔹 تشتيت: {details['distraction'][1]} ({details['distraction'][2]:.0%})")
                        st.write(f"🔹 CNN: {details['cnn'][1]} ({details['cnn'][2]:.0%})")
                        st.write(f"🔹 VGG: {details['vgg'][1]} ({details['vgg'][2]:.0%})")
                        st.write(f"🔹 ResNet: {details['resnet'][1]} ({details['resnet'][2]:.0%})")
                
                except Exception as e:
                    st.error(f"❌ حدث خطأ أثناء التحليل: {e}")
        else:
            st.error("⚠️ يرجى رفع الصورتين أولاً!")

elif input_mode == "🎥 رفع فيديو":
    st.subheader("🎥 تحليل الفيديو")
    
    st.info("📌 **ملاحظة:** يجب أن يحتوي الفيديو على كاميرتين جنباً إلى جنب (أمامية على اليسار، جانبية على اليمين)")
    
    video_upload = st.file_uploader(
        "ارفع فيديو القيادة",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video",
        help="حجم الفيديو يجب أن يكون أقل من 200 ميجابايت"
    )
    
    if video_upload:
        st.video(video_upload)
        
        if st.button("🔍 تحليل الفيديو الآن", use_container_width=True, type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_upload.read())
                video_path = tmp_file.name
            
            st.markdown("---")
            st.subheader("⏳ جاري المعالجة...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_container = st.container()
            
            try:
                results = process_video_file(video_path, progress_bar, status_text, result_container)
                
                st.success("✅ اكتمل التحليل بنجاح!")
                
                # ملخص النتائج
                st.markdown("---")
                st.markdown("### 📊 ملخص تحليل الفيديو")
                
                if results:
                    total_frames = len(results)
                    high_risk_frames = sum(1 for r in results if r['analysis']['risk_score'] >= 50)
                    medium_risk_frames = sum(1 for r in results if 30 <= r['analysis']['risk_score'] < 50)
                    safe_frames = sum(1 for r in results if r['analysis']['risk_score'] < 30)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("إجمالي الإطارات", total_frames)
                    col2.metric("خطر عالي 🔴", high_risk_frames)
                    col3.metric("تحذير 🟡", medium_risk_frames)
                    col4.metric("آمن 🟢", safe_frames)
                    
                    safety_percentage = (safe_frames / total_frames * 100) if total_frames > 0 else 0
                    st.progress(safety_percentage / 100)
                    st.write(f"**نسبة السلامة الإجمالية: {safety_percentage:.1f}%**")
                    
                    # عرض التفاصيل
                    if st.checkbox("📋 عرض التفاصيل الكاملة"):
                        st.markdown("### 📝 سجل التحليل")
                        for result in results:
                            time_str = f"{int(result['time']//60):02d}:{int(result['time']%60):02d}"
                            with st.expander(f"⏱️ الوقت {time_str} - {result['analysis']['status']}"):
                                st.write(f"**مستوى الخطر:** {result['analysis']['risk_score']}%")
                                if result['analysis']['alerts']:
                                    st.write("**التنبيهات:**")
                                    for alert in result['analysis']['alerts']:
                                        st.write(f"- {alert}")
                else:
                    st.warning("⚠️ لم يتم العثور على نتائج")
            
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة الفيديو: {e}")
            
            finally:
                os.unlink(video_path)

else:  # بث مباشر
    st.subheader("📹 البث المباشر من الكاميرات")
    
    st.warning("⚠️ **تنبيه:** البث المباشر يعمل فقط عند التشغيل المحلي. لا يعمل على Streamlit Cloud!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        front_cam_id = st.number_input(
            "رقم الكاميرا الأمامية",
            value=0,
            min_value=0,
            max_value=10,
            help="عادة 0 أو 1"
        )
    
    with col2:
        side_cam_id = st.number_input(
            "رقم الكاميرا الجانبية",
            value=1,
            min_value=0,
            max_value=10,
            help="عادة 1 أو 2"
        )
    
    st.markdown("---")
    
    stop_button_placeholder = st.empty()
    
    if st.button("▶️ بدء البث المباشر", use_container_width=True, type="primary"):
        st.markdown("---")
        try:
            process_live_stream(int(front_cam_id), int(side_cam_id), stop_button_placeholder)
        except Exception as e:
            st.error(f"❌ خطأ في البث: {e}")
            st.info("💡 تأكد من توصيل الكاميرات وتشغيل التطبيق محلياً")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;'>
    <h3>🚗 Driver Safety Monitor Pro v2.0</h3>
    <p><strong>Powered by:</strong> PyTorch • OpenCV • Streamlit</p>
    <p><strong>Models:</strong> EfficientNet B0 | Custom CNN | VGG16 | ResNet50</p>
    <p style='color: #667eea;'><strong>Developed with ❤️ for Safer Roads</strong></p>
</div>
""", unsafe_allow_html=True)
            "
