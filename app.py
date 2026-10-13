"""
app.py — Optional browser demo (Gradio).

Upload a waste photo and the trained model predicts its class with
confidence scores. Low-confidence predictions are flagged for manual review.

Setup:
    pip install gradio
Run:
    python app.py
Then open the local URL that Gradio prints (default http://127.0.0.1:7860).
"""

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/waste_classifier.keras")
IMG_SIZE = (224, 224)
THRESHOLD = 0.80

try:
    import gradio as gr
except ImportError:
    raise SystemExit("Gradio is not installed. Run:  pip install gradio")


def predict(image):
    """Gradio callback: numpy image in -> (label dict, markdown decision)."""
    import tensorflow as tf

    if image is None:
        return {}, "### No image uploaded"
    if image.ndim == 2:                       # grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:                  # RGBA -> RGB
        image = image[..., :3]

    # feed RAW [0, 255] pixels — the model rescales internally
    img = tf.image.resize(image, IMG_SIZE).numpy().astype(np.float32)
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]

    result = {c: float(p) for c, p in zip(class_names, probs)}
    best, conf = class_names[int(np.argmax(probs))], float(np.max(probs))
    if conf >= THRESHOLD:
        note = f"### {best}  \nConfident prediction ({conf:.1%})."
    else:
        note = (f"### {best}  \n**Low confidence ({conf:.1%}) — "
                f"needs manual review.**")
    return result, note


if __name__ == "__main__":
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Train it first:  python src/train.py")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads((MODEL_PATH.parent / "class_names.json").read_text())

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Upload a waste photo"),
        outputs=[gr.Label(num_top_classes=6, label="Class probabilities"),
                 gr.Markdown(label="Decision")],
        title="♻️ Waste Image Classifier",
        description="CNN (MobileNetV2 + transfer learning) that classifies waste "
                    "images into 6 classes for automated segregation.",
    )
    demo.launch()
