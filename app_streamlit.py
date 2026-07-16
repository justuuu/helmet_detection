import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
import os

st.set_page_config(page_title="Deteksi Helm — YOLOv8", layout="centered")


@st.cache_resource
def load_model():
    return YOLO("helm_yolov8_best.pt")


model = load_model()

st.title("Deteksi penggunaan helm — YOLOv8")
st.write("Deteksi apakah orang dalam gambar/video menggunakan helm atau tidak.")

conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)


def show_detection_details(results):
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.subheader("Detail deteksi")
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"- **{cls_name}** — confidence: {conf:.2f}")
    else:
        st.info("Tidak ada objek terdeteksi pada threshold ini. Coba turunkan confidence threshold.")


tab_upload, tab_camera, tab_video = st.tabs(["📁 Upload gambar", "📷 Kamera", "🎥 Upload video"])

# ---------- TAB 1: Upload gambar ----------
with tab_upload:
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], key="upload_img")

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

        show_detection_details(results)

# ---------- TAB 2: Kamera ----------
with tab_camera:
    st.caption("Klik tombol di bawah untuk mengaktifkan kamera.")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if not st.session_state.camera_active:
        if st.button("Aktifkan kamera", key="btn_activate_camera"):
            st.session_state.camera_active = True
            st.rerun()
    else:
        if st.button("Matikan kamera", key="btn_deactivate_camera"):
            st.session_state.camera_active = False
            st.rerun()

        camera_photo = st.camera_input("Ambil foto", key="camera_img")

        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")

            with st.spinner("Mendeteksi..."):
                results = model.predict(image, conf=conf_threshold)
                annotated = results[0].plot()[:, :, ::-1]  # BGR -> RGB

            st.subheader("Hasil deteksi")
            st.image(annotated, use_container_width=True)

            show_detection_details(results)

# ---------- TAB 3: Upload video ----------
with tab_video:
    st.caption("Video akan diproses frame-per-frame — untuk video panjang, proses bisa memakan waktu beberapa menit.")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], key="upload_video")

    if uploaded_video is not None:
        # Simpan video upload ke file sementara
        input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(input_path)

        if st.button("Proses video"):
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # frame dari cv2 sudah dalam format BGR — sesuai yang diharapkan ultralytics
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"Memproses frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            status_text.text("Selesai!")

            st.subheader("Hasil deteksi")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("Download video hasil deteksi", f, file_name="hasil_deteksi.mp4", mime="video/mp4")

            os.remove(input_path)
            os.remove(output_path)
