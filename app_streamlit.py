import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deteksi Helm — YOLOv8", layout="centered")


@st.cache_resource
def load_model():
    return YOLO("helm_yolov8_best.pt")


model = load_model()

st.title("Deteksi penggunaan helm — YOLOv8")
st.write("Upload gambar untuk mendeteksi apakah orang di dalamnya menggunakan helm atau tidak.")

conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with st.spinner("Mendeteksi..."):
        results = model.predict(image, conf=conf_threshold)
        annotated = results[0].plot()[:, :, ::-1]  # BGR -> RGB

    with col2:
        st.subheader("Hasil deteksi")
        st.image(annotated, use_container_width=True)

    # Ringkasan deteksi
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.subheader("Detail deteksi")
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"- **{cls_name}** — confidence: {conf:.2f}")
    else:
        st.info("Tidak ada objek terdeteksi pada threshold ini. Coba turunkan confidence threshold.")
