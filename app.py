import gradio as gr
from ultralytics import YOLO

# Load model — file ini harus diupload bareng ke Space (sejajar dengan app.py)
model = YOLO('helm_yolov8_best.pt')


def detect_helmet(image, conf_threshold):
    """Jalankan deteksi helm pada gambar yang diupload user."""
    if image is None:
        return None

    results = model.predict(image, conf=conf_threshold)
    annotated_image = results[0].plot()  # gambar dengan bounding box + label

    # plot() ngembaliin format BGR (OpenCV), Gradio butuh RGB
    annotated_image = annotated_image[:, :, ::-1]

    return annotated_image


demo = gr.Interface(
    fn=detect_helmet,
    inputs=[
        gr.Image(type="numpy", label="Upload gambar"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Confidence threshold"),
    ],
    outputs=gr.Image(type="numpy", label="Hasil deteksi"),
    title="Deteksi Penggunaan Helm — YOLOv8",
    description="Upload gambar untuk mendeteksi apakah orang di dalamnya menggunakan helm atau tidak.",
    examples=None,  # bisa diisi list path gambar contoh kalau ada
)

if __name__ == "__main__":
    demo.launch()
