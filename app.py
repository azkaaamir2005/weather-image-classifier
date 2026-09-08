import os
import time
import base64
from io import BytesIO
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
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_production_model()

MODEL_SIZE_MB = round(os.path.getsize(WEIGHTS_PATH) / (1024 * 1024), 2) if WEIGHTS_PATH.exists() else 9.1

inference_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_inference(pil_image):
    img_rgb = pil_image.convert('RGB')
    tensor_img = inference_transforms(img_rgb).unsqueeze(0)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    
    return probabilities, latency_ms

def img_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ==========================================
# 2. UI CONFIGURATION & CSS INJECTION
# ==========================================
st.set_page_config(page_title="Intel Weather AI Portal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* Hide sidebar and Streamlit top toolbar */
        [data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        
        /* Page padding */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
        }
        
        .reportview-container { background: #0e1117; }
        
        h1 { 
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
            font-weight: 700; 
            color: #ffffff !important; 
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.3 !important;
            font-size: 1.8rem !important;
        }
        
        h2, h3 { color: #00E5FF !important; font-weight: 600; }

        /* Generous spacing around File Uploader */
        div[data-testid="stFileUploader"] {
            margin-top: 10px !important;
            margin-bottom: 45px !important;
        }

        /* Top Predicted Class Card Box */
        .prediction-card {
            background-color: #1e222b;
            padding: 18px 20px;
            border-radius: 8px;
            border-left: 5px solid #00E5FF;
            height: 220px !important;
            box-sizing: border-box !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        /* Probability Breakdown Card Box */
        .prob-card {
            background-color: #1e222b;
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid #2e3545;
            border-left: 5px solid #377ACC;
            height: 220px !important;
            box-sizing: border-box !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        /* Metric Box Styling */
        .stMetric { 
            background-color: #1e222b; 
            padding: 12px 15px; 
            border-radius: 10px; 
            border-left: 5px solid #00E5FF; 
            min-height: 95px;
        }
        
        div[data-testid="stMetricValue"] { 
            color: #ffffff !important; 
            font-size: 16px !important; 
            font-weight: 700 !important;
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        div[data-testid="stMetricLabel"] { 
            color: #8a99ad !important; 
            font-size: 11px !important; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .tech-badge {
            display: inline-block;
            background: #111827;
            color: #00E5FF;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid #00E5FF33;
            margin-right: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation header row
head_col1, head_col2 = st.columns([2.2, 1.8], vertical_alignment="center")

with head_col2:
    page = st.segmented_control(
        "Navigation",
        options=["⚡ Live Predictive Engine", "📊 Model Architecture & Diagnostics"],
        default="⚡ Live Predictive Engine",
        label_visibility="collapsed"
    )

with head_col1:
    if page == "📊 Model Architecture & Diagnostics":
        st.title("🛰️ Edge-Optimized Weather Engine")
    else:
        st.title("🎯 Real-Time Predictive Analysis")

st.markdown("---")

if not page:
    page = "⚡ Live Predictive Engine"

# ==========================================
# VIEW 1: DIAGNOSTICS
# ==========================================
if page == "📊 Model Architecture & Diagnostics":
    st.write("Production-ready diagnostic analysis interface validating deep layer feature maps.")
    
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    with r1_col1:
        st.metric(label="CORE BACKBONE", value="MobileNetV3-Small")
    with r1_col2:
        st.metric(label="EDGE FOOTPRINT", value=f"~{MODEL_SIZE_MB} MB")
    with r1_col3:
        st.metric(label="INFERENCE SPEED", value="< 20 ms (CPU)")
        
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    with r2_col1:
        st.metric(label="INPUT RESOLUTION", value="160 x 160 px")
    with r2_col2:
        st.metric(label="TARGET CLASSES", value="5 Weather Classes")
    with r2_col3:
        st.metric(label="OPTIMIZATION PIPELINE", value="Adam (LR=0.001)")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.subheader("Model Validation: Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Evaluated across isolated validation splits", use_container_width=True)
        else:
            st.info("Confusion matrix missing under 'outputs/confusion_matrix.png'. Run main.py first.")
            
    with col_right:
        st.subheader("Presentation Technical Notes")
        with st.expander("🔍 Why MobileNetV3-Small?", expanded=True):
            st.write(f"""
            * **Hardware Efficiency:** Compact footprint (**~{MODEL_SIZE_MB} MB**) designed for edge AI accelerators and microcontrollers.
            * **Sub-20ms Latency:** Low latency inference suitable for real-time CCTV and dashcam streams.
            * **Neural Architecture Search:** Hardware-aware NAS fine-tunes depthwise separable convolutions.
            * **Advanced Activations:** Employs Hard-Swish activation functions to lower compute complexity.
            """)
        with st.expander("📈 Training Constraints Overview"):
            st.write("""
            * **Dataset Scoping:** Filtered raw 11 Kaggle classes down to 5 distinct weather conditions.
            * **Loss Function:** Categorical Cross-Entropy with Adam Optimizer.
            * **Batch Sizing:** 32 images per backpropagation weight update.
            """)

# ==========================================
# VIEW 2: INFERENCE PREDICTOR
# ==========================================
elif page == "⚡ Live Predictive Engine":
    if model is None:
        st.error(f"System Check Error: Target parameter matrix file '{WEIGHTS_PATH.name}' is missing.")
    else:
        uploaded_file = st.file_uploader("DROP ATMOSPHERIC IMAGE RECORD TO PROCESS...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            user_img = Image.open(uploaded_file)
            
            with st.spinner("Processing deep tensor array..."):
                prob_array, latency_ms = run_inference(user_img)
                max_idx = np.argmax(prob_array)
                predicted_label = CLASSES[max_idx]
                confidence_pct = prob_array[max_idx] * 100
                
                # Dedicated visual separator between uploader bar and section titles
                st.markdown("<div style='margin-bottom: 35px;'></div>", unsafe_allow_html=True)
                
                # Single Row 3-Column Layout
                col_img, col_class, col_probs = st.columns([1, 1, 1.25], vertical_alignment="top")
                
                with col_img:
                    st.markdown("<h3 style='margin-bottom:12px; margin-top:0;'>📸 Analysis Target</h3>", unsafe_allow_html=True)
                    img_b64 = img_to_base64(user_img)
                    st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="height:220px; width:100%; object-fit:cover; border-radius:8px;" />', unsafe_allow_html=True)
                    
                with col_class:
                    st.markdown("<h3 style='margin-bottom:12px; margin-top:0;'>🧠 Output Vector</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="prediction-card">
                            <span style="color:#8a99ad; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.5px;">Top Predicted Class</span>
                            <h2 style="margin:4px 0 8px 0; color:#ffffff !important; font-size:28px; font-weight:700;">{predicted_label.upper()}</h2>
                            <div style="margin-bottom:10px;">
                                <span style="color:#00E5FF; font-size:14px; font-weight:600;">System Confidence: {confidence_pct:.2f}%</span>
                            </div>
                            <div style="display:flex; gap:4px; flex-wrap:wrap;">
                                <span class="tech-badge">⚡ {latency_ms:.1f} ms</span>
                                <span class="tech-badge">💾 ~{MODEL_SIZE_MB} MB</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col_probs:
                    st.markdown("<h3 style='margin-bottom:12px; margin-top:0;'>📊 Probability Breakdown</h3>", unsafe_allow_html=True)
                    
                    bars_html = ""
                    for name, prob in zip(CLASSES, prob_array):
                        pct = prob * 100
                        bars_html += f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 7px;">
                            <span style="color: #ffffff; font-size: 13px; font-weight: 500; width: 80px;">{name.capitalize()}</span>
                            <div style="flex-grow: 1; background-color: #111827; height: 8px; border-radius: 4px; margin: 0 10px; overflow: hidden;">
                                <div style="width: {pct:.1f}%; background-color: #00E5FF; height: 100%; border-radius: 4px;"></div>
                            </div>
                            <span style="color: #8a99ad; font-size: 12px; font-weight: 600; width: 45px; text-align: right;">{pct:.1f}%</span>
                        </div>
                        """
                    
                    st.markdown(f"""
                        <div class="prob-card">
                            {bars_html}
                        </div>
                    """, unsafe_allow_html=True)