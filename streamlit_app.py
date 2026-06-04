import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="centered")

# ── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model("model/model.h5")

model = load_cnn_model()

CLASS_LABELS = {
    0: "✅ Benign (Safe / Harmless)",
    1: "⚠️ Malignant (Dangerous / Consult Doctor)"
}

# ── Skin Validator (Kovac's Rule) ─────────────────────────────
def is_valid_skin_image(img):
    img_rgb = np.array(img.convert("RGB"))
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    skin_mask = (R > 95) & (G > 40) & (B > 20) & \
                (R > G) & (R > B) & \
                (abs(R.astype(int) - G.astype(int)) > 15)
    skin_ratio = np.sum(skin_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
    return skin_ratio > 0.2

# ── UI ────────────────────────────────────────────────────────
st.title("🩺 Skin Disease Classification")
st.subheader("MobileNet CNN — Malignant vs Benign Detection (90% Accuracy)")
st.markdown("Upload a **dermoscopic skin lesion image** to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        if not is_valid_skin_image(img):
            st.error("❌ This doesn't appear to be a valid skin image. Please upload a dermoscopic image.")
        else:
            img_resized = img.resize((224, 224)).convert("RGB")
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            pred_index = int(np.argmax(preds[0]))
            label = CLASS_LABELS[pred_index]
            confidence = round(100 * float(np.max(preds[0])), 2)

            st.markdown("---")
            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence}%")
            st.progress(confidence / 100)

st.markdown("---")
st.caption("Built by Aditya Vinayak Sahu | MobileNet CNN | SRMIst")
