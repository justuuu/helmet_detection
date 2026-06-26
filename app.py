import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deteksi Helm Pengendara",
    page_icon="🪖",
    layout="wide"
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    .header-box {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border: 1px solid #3d4fd1;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 24px;
        text-align: center;
    }
    .header-box h1 { color: #ffffff; font-size: 2.2rem; margin: 0; }
    .header-box p  { color: #a0aec0; margin: 8px 0 0; font-size: 1rem; }

    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-box .value { font-size: 2rem; font-weight: bold; color: #fff; }
    .metric-box .label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }

    .helmet-badge {
        background: #1a3a2a;
        border: 1px solid #2ecc71;
        color: #2ecc71;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: bold;
        text-align: center;
        margin: 4px 0;
    }
    .no-helmet-badge {
        background: #3a1a1a;
        border: 1px solid #e74c3c;
        color: #e74c3c;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: bold;
        text-align: center;
        margin: 4px 0;
    }
    .warning-box {
        background: #3a2a1a;
        border: 1px solid #f39c12;
        color: #f39c12;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: bold;
        text-align: center;
    }
    .safe-box {
        background: #1a3a2a;
        border: 1px solid #2ecc71;
        color: #2ecc71;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: bold;
        text-align: center;
    }

    div[data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3561;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    # Coba load dari beberapa lokasi
    model_paths = [
        'best.pt',
        'helm_yolov8_best.pt',
        '/content/drive/MyDrive/helm_yolov8_best.pt'
    ]
    for path in model_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>🪖 Sistem Deteksi Helm Pengendara Motor</h1>
    <p>Simulasi CCTV Persimpangan — Computer Vision Mini Capstone</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                               help="Semakin tinggi = semakin ketat seleksi deteksi")

    st.markdown("---")
    st.markdown("### 📤 Upload Model")
    uploaded_model = st.file_uploader("Upload file best.pt", type=['pt'])
    if uploaded_model:
        with open('best.pt', 'wb') as f:
            f.write(uploaded_model.read())
        st.success("✅ Model berhasil diupload!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Info Model")
    model = load_model()
    if model:
        st.success("✅ Model aktif")
        names = list(model.names.values())
        for name in names:
            icon = "🪖" if 'with' in name.lower() or ('helmet' in name.lower() and 'without' not in name.lower()) else "❌"
            st.markdown(f"{icon} `{name}`")
    else:
        st.error("❌ Model belum diload")
        st.info("Upload file best.pt di atas")

    st.markdown("---")
    st.markdown("### ℹ️ Tentang")
    st.markdown("""
    **Model:** YOLOv8n  
    **Dataset:** Bike Helmet Detection  
    **mAP@0.5:** 0.821  
    **Precision:** 0.869  
    **Recall:** 0.722
    """)

# ============================================================
# MAIN CONTENT
# ============================================================
if not model:
    st.warning("⚠️ Upload file `best.pt` di sidebar kiri untuk mulai!")
    st.stop()

# Tab navigasi
tab1, tab2, tab3 = st.tabs(["📸 Deteksi Foto", "🎬 Deteksi Video", "📷 Webcam"])

# ============================================================
# TAB 1 — DETEKSI FOTO
# ============================================================
with tab1:
    st.markdown("### Upload Foto")
    uploaded_img = st.file_uploader("Pilih gambar (jpg/png)", type=['jpg', 'jpeg', 'png'], key="img")

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gambar Asli**")
            st.image(img, use_container_width=True)

        # Deteksi
        with st.spinner("🔍 Mendeteksi..."):
            results = model.predict(img_np, conf=conf_threshold, verbose=False)

        r = results[0]
        annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown(f"**Hasil Deteksi** (conf={conf_threshold})")
            st.image(annotated, use_container_width=True)

        # Hitung hasil
        class_names = model.names
        counts = {name: 0 for name in class_names.values()}
        for cls in r.boxes.cls:
            counts[class_names[int(cls)]] += 1

        # Tampilkan metrik
        st.markdown("---")
        st.markdown("### 📊 Hasil Deteksi")

        cols = st.columns(len(counts) + 1)
        total = sum(counts.values())

        with cols[0]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{total}</div>
                <div class="label">Total Terdeteksi</div>
            </div>""", unsafe_allow_html=True)

        for i, (name, count) in enumerate(counts.items()):
            with cols[i+1]:
                color = "#2ecc71" if 'with' in name.lower() and 'without' not in name.lower() else "#e74c3c"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="value" style="color:{color}">{count}</div>
                    <div class="label">{name}</div>
                </div>""", unsafe_allow_html=True)

        # Status pelanggaran
        st.markdown("")
        no_helmet_count = sum(v for k, v in counts.items() if 'without' in k.lower() or ('no' in k.lower()))
        if no_helmet_count > 0:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ PELANGGARAN TERDETEKSI: {no_helmet_count} pengendara tidak menggunakan helm!
            </div>""", unsafe_allow_html=True)
        elif total > 0:
            st.markdown("""
            <div class="safe-box">
                ✅ Semua pengendara menggunakan helm dengan benar
            </div>""", unsafe_allow_html=True)

# ============================================================
# TAB 2 — DETEKSI VIDEO
# ============================================================
with tab2:
    st.markdown("### Upload Video")
    st.info("Video akan diproses frame by frame dan menghasilkan video output dengan deteksi helm + tracking ByteTrack")

    uploaded_video = st.file_uploader("Pilih video (mp4/avi/mov)", type=['mp4', 'avi', 'mov'], key="vid")

    if uploaded_video:
        # Simpan video sementara
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()

        # Info video
        cap = cv2.VideoCapture(tfile.name)
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        col1, col2, col3 = st.columns(3)
        col1.metric("Resolusi", f"{width}x{height}")
        col2.metric("FPS", fps)
        col3.metric("Total Frame", total)

        if st.button("▶️ Mulai Proses Video", type="primary"):
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            cap = cv2.VideoCapture(tfile.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            status   = st.empty()
            stats    = st.empty()

            frame_count     = 0
            violation_count = 0
            class_names     = model.names
            no_helmet_ids   = [i for i, n in class_names.items()
                               if 'without' in n.lower() or ('no' in n.lower() and 'helmet' in n.lower())]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame,
                    conf=conf_threshold,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )
                annotated = results[0].plot()

                no_helm = sum(1 for cls in results[0].boxes.cls if int(cls) in no_helmet_ids)
                if no_helm > 0:
                    violation_count += 1
                    cv2.putText(annotated, f'PELANGGARAN: {no_helm} tanpa helm',
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(annotated, f'Frame: {frame_count}/{total}',
                            (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                out.write(annotated)
                frame_count += 1

                if frame_count % 10 == 0:
                    pct = frame_count / total
                    progress.progress(pct)
                    status.markdown(f"⏳ Memproses frame {frame_count}/{total} ({pct*100:.0f}%)")
                    pct_viol = 100*violation_count//frame_count if frame_count > 0 else 0
                    stats.markdown(f"⚠️ Frame pelanggaran: **{violation_count}** ({pct_viol}%)")

            cap.release()
            out.release()
            progress.progress(1.0)
            status.markdown("✅ Video selesai diproses!")

            # Tampilkan hasil
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Total Frame", frame_count)
            pct_viol = 100*violation_count//frame_count if frame_count > 0 else 0
            c2.metric("Frame Pelanggaran", f"{violation_count} ({pct_viol}%)")

            # Download
            with open(output_path, 'rb') as f:
                st.download_button(
                    "📥 Download Video Hasil",
                    f,
                    file_name="hasil_deteksi_helm.mp4",
                    mime="video/mp4",
                    type="primary"
                )

            os.unlink(tfile.name)

# ============================================================
# TAB 3 — WEBCAM
# ============================================================
with tab3:
    st.markdown("### Deteksi via Webcam")
    st.info("📷 Ambil foto dari webcam untuk dideteksi langsung")

    img_file = st.camera_input("Klik tombol untuk ambil foto")

    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        with st.spinner("🔍 Mendeteksi..."):
            results = model.predict(img_np, conf=conf_threshold, verbose=False)

        r = results[0]
        annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Foto Asli**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated, use_container_width=True)

        class_names = model.names
        counts = {name: 0 for name in class_names.values()}
        for cls in r.boxes.cls:
            counts[class_names[int(cls)]] += 1

        st.markdown("---")
        for name, count in counts.items():
            if 'without' in name.lower() or ('no' in name.lower()):
                st.markdown(f'<div class="no-helmet-badge">❌ {name}: {count} orang</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="helmet-badge">🪖 {name}: {count} orang</div>',
                           unsafe_allow_html=True)

        no_helmet_count = sum(v for k, v in counts.items()
                             if 'without' in k.lower() or 'no' in k.lower())
        st.markdown("")
        if no_helmet_count > 0:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ PELANGGARAN: {no_helmet_count} pengendara tidak menggunakan helm!
            </div>""", unsafe_allow_html=True)
        elif sum(counts.values()) > 0:
            st.markdown("""
            <div class="safe-box">
                ✅ Semua pengendara menggunakan helm
            </div>""", unsafe_allow_html=True)
