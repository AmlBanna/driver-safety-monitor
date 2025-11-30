"""
Driver Safety Monitor Pro - Complete System
4 Models: EfficientNet + 3 Behavior Models + Drowsiness Detection
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
import time
from datetime import datetime
import os
import tempfile
from io import BytesIO

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius:
