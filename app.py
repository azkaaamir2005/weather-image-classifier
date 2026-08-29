import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. CORE ML INTEGRATION CORE
# ==========================================
CLASSES = ['fogsmog', 'lightning', 'rain', 'sandstorm', 'snow']
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weather_mobilenet.pth"
CONFUSION_MATRIX_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"

@st.cache_resource
def load_production_model():
    if not WEIGHTS_PATH.exists():
        return None
    # Rebuilding MobileNetV3 small backbone
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_production_model()

inference_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_inference(pil_image):
    img_rgb = pil_image.convert('RGB')
    tensor_img = inference_transforms(img_rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
    return probabilities

# ==========================================
# 2. PREMIUM UI HEADER & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Intel Weather AI Portal", layout="wide", initial_sidebar_state="expanded")

# Injecting Custom CSS styling for a cleaner look
st.markdown("""
    <style>
        .reportview-container { background: #0e1117; }
        h1 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; color: #ffffff !important; }
        h2, h3 { color: #00E5FF !important; font-weight: 600; }
        
        /* Premium Metric Box Overrides */
        .stMetric { 
            background-color: #1e222b; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #00E5FF; 
            min-height: 120px; /* Ensures all boxes stay uniform in height */
        }
        /* Fix truncation and allow text wrapping */
        div[data-testid="stMetricValue"] { 
            color: #ffffff !important; 
            font-size: 20px !important; /* Slightly smaller to fit text completely */
            font-weight: 600 !important;
            white-space: normal !important; /* Forces text to wrap instead of truncating */
            word-break: break-word !important; 
        }
        div[data-testid="stMetricLabel"] { 
            color: #8a99ad !important; 
            font-size: 12px !important; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
""", unsafe_allow_html=True)



# Navigation Control
st.sidebar.markdown("<h2 style='text-align: center; color: #00E5FF;'>AI CONTROL CENTER</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("CHOOSE DASHBOARD VIEW:", ["📊 Model Architecture & Diagnostics", "⚡ Live Predictive Engine"])

# ==========================================
# VIEW 1: DIAGNOSTICS (Window 1)
# ==========================================
if page == "📊 Model Architecture & Diagnostics":
    st.title("🛰️ Edge-Optimized Weather Classification Engine")
    st.write("Production-ready diagnostic analysis interface validating deep layer feature maps.")
    st.markdown("---")
    
    # Structural KPI Blocks
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="CORE BACKBONE", value="MobileNetV3 Small")
    with col2:
        st.metric(label="INPUT RESOLUTION", value="160 x 160 px")
    with col3:
        st.metric(label="TARGET CLASSIFICATIONS", value="5 Classes")
    with col4:
        st.metric(label="OPTIMIZATION PIPELINE", value="Adam (LR=0.001)")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("Model Validation: Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Evaluated across isolated validation splits", use_container_width=True)
        else:
            st.info("Confusion matrix file was not found under 'outputs/confusion_matrix.png'. Run your main.py file completely first to generate it automatically.")
            
    with col_right:
        st.subheader("Presentation Technical Notes")
        with st.expander("🔍 Why MobileNetV3-Small?", expanded=True):
            st.write("""
            * **Hardware Efficiency:** Designed explicitly for high accuracy with low resource consumption on edge systems.
            * **Neural Architecture Search (NAS):** Employs hardware-aware NAS to fine-tune network configurations dynamically.
            * **Advanced Layers:** Integrates Hard-Swish activation mechanisms reducing processing cost.
            """)
        with st.expander("📈 Training Constraints Overview"):
            st.write("""
            * **Loss Framework:** Sparse Categorical Cross-Entropy.
            * **Batch Sizing Structure:** 32 allocations per weight updating calculation step.
            * **Augmentation Policies:** Image channel standardization using ImageNet criteria.
            """)

# ==========================================
# VIEW 2: INFERENCE PREDICTOR (Window 2)
# ==========================================
elif page == "⚡ Live Predictive Engine":
    st.title("🎯 Real-Time Predictive Analysis Terminal")
    st.write("Provide raw atmospheric photography vectors to calculate probability distributions instantly.")
    st.markdown("---")
    
    if model is None:
        st.error(f"System Check Error: Target parameter matrix file '{WEIGHTS_PATH.name}' is missing inside this directory. Ensure training is finished.")
    else:
        uploaded_file = st.file_uploader("DROP ATMOSPHERIC IMAGE RECORD TO PROCESS...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col_display_img, col_display_chart = st.columns([1, 1.3])
            user_img = Image.open(uploaded_file)
            
            with col_display_img:
                st.markdown("<h3 style='margin-bottom:15px;'>📸 Analysis Target</h3>", unsafe_allow_html=True)
                st.image(user_img, use_container_width=True)
                
            with col_display_chart:
                st.markdown("<h3 style='margin-bottom:15px;'>🧠 AI Output Vectors</h3>", unsafe_allow_html=True)
                
                with st.spinner("Processing deep tensor array..."):
                    prob_array = run_inference(user_img)
                    max_idx = np.argmax(prob_array)
                    predicted_label = CLASSES[max_idx]
                    confidence_pct = prob_array[max_idx] * 100
                    
                    # Enterprise Highlight Banner
                    st.markdown(f"""
                        <div style="background-color:#1e222b; padding:20px; border-radius:8px; border-left:6px solid #00E5FF; margin-bottom:25px;">
                            <span style="color:#8a99ad; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1.5px;">Top Predicted Class</span>
                            <h2 style="margin:5px 0 0 0; color:#ffffff !important; font-size:36px; font-weight:700;">{predicted_label.upper()}</h2>
                            <span style="color:#00E5FF; font-size:16px; font-weight:600;">System Confidence: {confidence_pct:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Clean Enterprise Distribution Progress Bars
                    st.markdown("<span style='color:#8a99ad; font-size:14px; font-weight:600;'>Probability Breakdown</span>", unsafe_allow_html=True)
                    
                    # Display metrics horizontally using native sleek progress indicators instead of basic matplotlib charts
                    for name, prob in zip(CLASSES, prob_array):
                        col_label, col_bar = st.columns([1, 4])
                        with col_label:
                            st.markdown(f"<p style='color:#ffffff; margin:0; line-height:30px; font-weight:500;'>{name.capitalize()}</p>", unsafe_allow_html=True)
                        with col_bar:
                            st.progress(float(prob))
                            st.markdown(f"<p style='color:#8a99ad; font-size:12px; margin:-10px 0 10px 0; text-align:right;'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
