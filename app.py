import streamlit as st
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fire & Smoke Emergency System",
    page_icon="🔥",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.alert-box {
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #ff0000;
    background-color: #ffe5e5;
}

.safe-box {
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #00a000;
    background-color: #e8ffe8;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f5f5;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = (
    "runs/classify/runs/classification/"
    "yolo11_fire/weights/best.pt"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    model_status = True
except Exception as e:
    model_status = False
    model = None
    st.error(f"Model loading error: {e}")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔥 AI Fire & Smoke Emergency Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered visual fire detection with emergency alert generation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ System Settings")

threshold = st.sidebar.slider(
    "Emergency Confidence Threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01
)

st.sidebar.write("Current threshold:", f"{threshold * 100:.0f}%")

st.sidebar.divider()

st.sidebar.success("🟢 YOLO11 Model Loaded")

st.sidebar.write("Model: YOLO11n-cls")
st.sidebar.write("Test Accuracy: 98.01%")
st.sidebar.write("Classes: Fire / Non-Fire")

# --------------------------------------------------
# INPUT METHOD
# --------------------------------------------------

st.subheader("📷 Detection Input")

input_type = st.radio(
    "Select input method:",
    ["Upload Image", "Camera"],
    horizontal=True
)

image = None

if input_type == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:

    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# --------------------------------------------------
# DETECTION
# --------------------------------------------------

if image is not None and model_status:

    st.divider()

    with st.spinner("🤖 AI is analyzing the image..."):

        result = model.predict(
            image,
            verbose=False
        )[0]

    # Prediction
    predicted_class = result.names[result.probs.top1]

    confidence = float(result.probs.top1conf)

    predicted_class = predicted_class.lower()

    # Fire confidence
    if predicted_class == "fire":

        fire_confidence = confidence

    else:

        fire_confidence = 1 - confidence

    # --------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.subheader("🖼️ Input Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------
    # AI RESULT
    # --------------------------------------------------

    with col2:

        st.subheader("🤖 AI Analysis")

        st.metric(
            "Fire Confidence",
            f"{fire_confidence * 100:.2f}%"
        )

        st.write(
            "**Prediction:**",
            predicted_class.upper()
        )

        st.write(
            "**Model:** YOLO11n Classification"
        )

        st.write(
            "**Detection Time:**",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # --------------------------------------------------
        # EMERGENCY DECISION AGENT
        # --------------------------------------------------

        if fire_confidence >= threshold:

            st.markdown(
                """
                <div class="alert-box">

                <h2>🚨 FIRE EMERGENCY DETECTED</h2>

                <p><b>Status:</b> CRITICAL</p>

                <p><b>AI Confidence:</b> {:.2f}%</p>

                <p><b>Recommended Action:</b></p>

                <ul>
                    <li>Immediately inspect the location</li>
                    <li>Alert responsible personnel</li>
                    <li>Activate emergency response procedure</li>
                    <li>Evacuate people if required</li>
                </ul>

                </div>
                """.format(fire_confidence * 100),
                unsafe_allow_html=True
            )

            st.error(
                "🚨 EMERGENCY ALERT GENERATED"
            )

        else:

            st.markdown(
                """
                <div class="safe-box">

                <h2>🟢 AREA SAFE</h2>

                <p>No significant fire indication detected.</p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "No emergency alert required."
            )

# --------------------------------------------------
# SYSTEM INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📊 System Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "YOLO11 Accuracy",
        "98.01%"
    )

with col2:

    st.metric(
        "YOLOv8 Accuracy",
        "96.69%"
    )

with col3:

    st.metric(
        "Test Images",
        "151"
    )

with col4:

    st.metric(
        "Classes",
        "2"
    )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()

st.subheader("🏆 Model Comparison")

comparison = {
    "Model": ["YOLO11n", "YOLOv8n"],
    "Top-1 Accuracy": ["98.01%", "96.69%"],
    "Top-5 Accuracy": ["100%", "100%"],
    "Task": ["Fire Classification", "Fire Classification"]
}

st.table(comparison)

st.info(
    "YOLO11 was selected as the primary model because it achieved "
    "the highest test accuracy of 98.01%."
)

# --------------------------------------------------
# PROJECT LIMITATION
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ Project Note")

st.warning(
    "The current Kaggle dataset contains Fire and Non-Fire image "
    "classification labels. It does not contain bounding-box "
    "annotations or a separate Smoke class. Therefore, this MVP "
    "implements validated Fire/Non-Fire classification and emergency "
    "alert generation. Independent Smoke object detection requires "
    "an annotated Fire/Smoke detection dataset."
)

st.caption(
    "AI Fire & Smoke Emergency Monitoring System | Academic Project"
)