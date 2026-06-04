"""
DeepFake Detector — Streamlit Web App
Upload images or videos and detect deepfakes with:
  - Visual CNN Analysis (EfficientNet-B4)
  - Emotion-Consistency Forensics (ECF) ★
  - Grad-CAM Explainability
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import json

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.5rem; font-weight:bold; color:#1E88E5; text-align:center;}
    .sub-header {font-size:1.2rem; color:#666; text-align:center;}
    .verdict-real {font-size:2rem; font-weight:bold; color:#4CAF50; text-align:center; padding:20px;}
    .verdict-fake {font-size:2rem; font-weight:bold; color:#F44336; text-align:center; padding:20px;}
    .metric-box {background:#f0f2f6; padding:15px; border-radius:10px; text-align:center;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Model (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    """Loads the trained model (cached across reruns)."""
    from src.model import load_model
    from config.config import MODEL_DIR, DEVICE
    
    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        return None
    
    model = load_model(model_path)
    return model


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/240/artificial-intelligence.png", width=100)
    st.title("🔒 DeepFake Detector")
    st.markdown("---")
    
    st.subheader("Settings")
    
    # Fusion weights
    visual_weight = st.slider("Visual CNN Weight", 0.0, 1.0, 0.6, 0.1)
    ecf_weight = 1.0 - visual_weight
    st.write(f"ECF Weight: {ecf_weight:.1f} (auto-balanced)")
    
    # Frames per video
    frames_per_video = st.slider("Frames to Extract (Video)", 5, 30, 15)
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **DeepFake Detector** uses:
    - 📸 EfficientNet-B4 (Visual)
    - 🎭 ECF Module (Emotion)
    - 🔥 Grad-CAM (Explainability)
    
    **Unique Feature:** Emotion-Consistency 
    Forensics cross-checks facial emotion 
    against speech sentiment.
    """)


# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">🔒 DeepFake Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Multimedia Authenticity Analysis System</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model
model = load_trained_model()

if model is None:
    st.warning("⚠️ No trained model found! Please train the model first:")
    st.code("python notebooks/02_train.py")
    st.stop()


# ──────────────────────────────────────────────
# TABS: Image vs Video
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎬 Video Detection", "ℹ️ About"])

# ════════════════════════════════════════════════
# TAB 1: IMAGE DETECTION
# ════════════════════════════════════════════════
with tab1:
    st.subheader("📸 Upload an Image for DeepFake Detection")
    
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Analyzing image..."):
                from src.preprocessing import detect_and_crop_face, preprocess_image
                from src.gradcam import generate_gradcam, overlay_gradcam_on_image, save_gradcam_result
                
                # Detect face
                face_crop, bbox = detect_and_crop_face(image, return_bbox=True)
                
                # Preprocess
                tensor = preprocess_image(face_crop).to(model.backbone.num_features and torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # Predict
                model.eval()
                with torch.no_grad():
                    probs = model.predict_proba(tensor)
                    pred = torch.argmax(probs, 1).item()
                    fake_prob = probs[0, 1].item()
                    real_prob = probs[0, 0].item()
                
                # Grad-CAM
                try:
                    heatmap = generate_gradcam(model, tensor, target_class=1)
                    overlay = overlay_gradcam_on_image(face_crop, heatmap)
                    gradcam_available = True
                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")
                    overlay = face_crop
                    gradcam_available = False
                
                # Display result
                if pred == 1:
                    st.markdown(f'<p class="verdict-fake">⚠️ FAKE — Confidence: {fake_prob*100:.1f}%</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="verdict-real">✅ REAL — Confidence: {real_prob*100:.1f}%</p>', unsafe_allow_html=True)
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Real Probability", f"{real_prob*100:.1f}%")
                m2.metric("Fake Probability", f"{fake_prob*100:.1f}%")
                m3.metric("Face Detected", "Yes" if bbox else "No (using full image)")
                
                # Show Grad-CAM
                if gradcam_available:
                    st.subheader("🔥 Grad-CAM Explanation")
                    st.image(overlay, caption="Red = regions suggesting fake", use_container_width=True)
                    st.info("Red regions indicate where the model found the strongest evidence of manipulation.")


# ════════════════════════════════════════════════
# TAB 2: VIDEO DETECTION (MAIN FOCUS)
# ════════════════════════════════════════════════
with tab2:
    st.subheader("🎬 Upload a Video for DeepFake Detection")
    st.info("🎯 This is the main feature: Video detection with ECF (Emotion-Consistency Forensics)")
    
    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        # Display video
        st.video(uploaded_video)
        
        st.markdown("---")
        
        # Analysis button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            from src.preprocessing import process_video_pipeline, preprocess_image
            from src.ecf import run_ecf_analysis
            from src.fusion import fuse_scores, generate_final_report
            from src.gradcam import generate_gradcam, overlay_gradcam_on_image
            from config.config import DEVICE
            
            progress = st.progress(0, text="Processing video...")
            
            # Step 1: Process video
            progress.progress(10, text="🎬 Extracting frames and audio...")
            video_data = process_video_pipeline(video_path)
            
            if video_data is None:
                st.error("❌ Could not process video. Please try another file.")
                st.stop()
            
            face_crops = video_data['face_crops']
            audio_path = video_data['audio_path']
            
            # Step 2: Visual CNN Analysis
            progress.progress(30, text="📸 Running visual CNN analysis...")
            
            frame_predictions = []
            fake_probs = []
            
            model.eval()
            for i, face in enumerate(face_crops):
                tensor = preprocess_image(face).to(DEVICE)
                with torch.no_grad():
                    probs = model.predict_proba(tensor)
                    pred = torch.argmax(probs, 1).item()
                    fake_prob = probs[0, 1].item()
                
                frame_predictions.append(pred)
                fake_probs.append(fake_prob)
            
            # Aggregate visual score (average fake probability across frames)
            visual_score = np.mean(fake_probs)
            visual_verdict = "FAKE" if visual_score >= 0.5 else "REAL"
            
            # Step 3: ECF Analysis
            progress.progress(60, text="🎭 Running ECF (Emotion-Consistency) analysis...")
            ecf_report = run_ecf_analysis(face_crops, audio_path)
            
            # Step 4: Fusion
            progress.progress(80, text="⚖️ Fusing scores...")
            fusion_result = fuse_scores(
                visual_score=visual_score,
                ecf_score=ecf_report['ecf_inconsistency_score'],
                visual_weight=visual_weight,
                ecf_weight=ecf_weight
            )
            
            # Step 5: Generate Grad-CAM for key frames
            progress.progress(90, text="🔥 Generating Grad-CAM heatmaps...")
            
            gradcam_images = []
            for i, face in enumerate(face_crops[:5]):  # Top 5 frames
                tensor = preprocess_image(face).to(DEVICE)
                try:
                    heatmap = generate_gradcam(model, tensor, target_class=1)
                    overlay = overlay_gradcam_on_image(face, heatmap)
                    gradcam_images.append(overlay)
                except:
                    gradcam_images.append(face)
            
            progress.progress(100, text="✅ Analysis complete!")
            
            # ── DISPLAY RESULTS ──
            st.markdown("---")
            st.subheader("📊 Detection Results")
            
            # Verdict
            if fusion_result['verdict'] == "FAKE":
                st.markdown(f'<p class="verdict-fake">⚠️ DEEPFAKE DETECTED — Confidence: {fusion_result["confidence_pct"]:.1f}%</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="verdict-real">✅ AUTHENTIC — Confidence: {fusion_result["confidence_pct"]:.1f}%</p>', unsafe_allow_html=True)
            
            # Score breakdown
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Score", f"{fusion_result['final_score']:.3f}")
            col2.metric("Visual Score", f"{visual_score:.3f}")
            col3.metric("ECF Score", f"{ecf_report['ecf_inconsistency_score']:.3f}")
            col4.metric("Confidence", fusion_result['confidence_label'])
            
            # Per-frame analysis
            st.subheader("🎞️ Per-Frame Analysis")
            frame_data = {
                'Frame': list(range(len(frame_predictions))),
                'Prediction': ['Fake' if p == 1 else 'Real' for p in frame_predictions],
                'Fake Probability': [f"{p:.3f}" for p in fake_probs],
            }
            st.dataframe(frame_data, use_container_width=True)
            
            # Visual score chart
            import pandas as pd
            chart_data = pd.DataFrame({
                'Frame': range(len(fake_probs)),
                'Fake Probability': fake_probs
            })
            st.line_chart(chart_data, x='Frame', y='Fake Probability')
            
            # ECF Report
            st.subheader("🎭 ECF Report (Emotion-Consistency Forensics) ★")
            
            ecf_col1, ecf_col2 = st.columns(2)
            with ecf_col1:
                st.info(f"**Facial Emotion:** {ecf_report['facial_emotion']}")
                st.info(f"**Emotion Distribution:** {ecf_report.get('emotion_distribution', 'N/A')}")
            with ecf_col2:
                st.info(f"**Speech Sentiment:** {ecf_report['speech_sentiment']}")
                st.info(f"**ECF Score:** {ecf_report['ecf_inconsistency_score']:.3f} ({ecf_report['verdict']})")
            
            if ecf_report.get('transcription'):
                st.text_area("Transcript", ecf_report['transcription'], height=100, disabled=True)
            
            # Grad-CAM Heatmaps
            if gradcam_images:
                st.subheader("🔥 Grad-CAM Heatmaps")
                st.info("Red regions = strongest evidence of manipulation")
                
                cols = st.columns(min(5, len(gradcam_images)))
                for i, (img, col) in enumerate(zip(gradcam_images, cols)):
                    with col:
                        st.image(img, caption=f"Frame {i}", use_container_width=True)
            
            # Full Report
            st.subheader("📋 Full Report")
            full_report = generate_final_report(fusion_result, ecf_report)
            st.code(full_report, language=None)
            
            # Save report
            from src.utils import save_report
            report_data = {
                **fusion_result,
                'ecf': ecf_report,
                'frame_predictions': frame_predictions,
                'frame_fake_probs': fake_probs,
            }
            save_report(report_data, f"report_{Path(video_path).stem}.json")


# ════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About DeepFake Detector")
    
    st.markdown("""
    ## 🔒 DeepFake Detector
    **AI-Powered Multimedia Authenticity Analysis System**
    
    ---
    
    ### 🏗️ System Architecture
    
    | Layer | Component | Output |
    |-------|-----------|--------|
    | Input | Image/Video upload | Raw media file |
    | Preprocessing | MTCNN face detection, frame extraction, audio separation | Normalized frames + audio |
    | Visual Analysis | EfficientNet-B4 CNN | Visual fake score |
    | ECF Module ★ | FER + Whisper + VADER | ECF inconsistency score |
    | Fusion | Weighted average | Final verdict + confidence |
    | XAI | Grad-CAM heatmap | Visual explanation |
    
    ---
    
    ### ★ Emotion-Consistency Forensics (ECF)
    
    Our **unique feature** that distinguishes this project from standard detectors:
    
    1. **FER** extracts facial emotion from video frames
    2. **Whisper** transcribes speech to text
    3. **VADER** analyzes sentiment of transcription
    4. **Cross-check:** Does the facial emotion match the speech sentiment?
    
    Deepfakes synthesize visual and audio independently, causing emotional mismatches.
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **CNN:** EfficientNet-B0 (via timm)
    - **Face Detection:** MTCNN (facenet-pytorch)
    - **Emotion:** FER library (AffectNet pre-trained)
    - **Speech:** OpenAI Whisper (small model)
    - **Sentiment:** VADER (NLTK)
    - **Explainability:** Grad-CAM
    - **Interface:** Streamlit
    - **Framework:** PyTorch
    
    ---
    
    ### 👥 Team
    - Sundas Iqbal (BSITM-A-23-31)
    - Laiba Fayyaz (BSITM-A-23-25)
    
    **Department:** BS IT 6th A (Morning)  
    **University:** University of Layyah
    """)