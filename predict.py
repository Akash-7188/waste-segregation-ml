"""
predict.py — Classify one image (or every image in a folder), with a confidence check.

Predictions below --threshold are flagged for manual review, so the system
never silently guesses — the same principle used in real sorting systems.

Usage:
    python src/predict.py --image path/to/photo.jpg
    python src/predict.py --folder path/to/photos
    python src/predict.py --image photo.jpg --threshold 0.85
"""

import argparse
import json
from pathlib import Path

import numpy as np

IMG_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: Path):
    """Read and resize an image exactly like the training pipeline did.

    Note: pixels are returned RAW in [0, 255] — the model contains its own
    Rescaling layer, so we must NOT scale here (that would double-scale and
    break predictions).
    """
    import tensorflow as tf
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)   # float32, [0, 255]
    return img.numpy().astype(np.float32)[np.newaxis, ...]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/waste_classifier.keras"))
    ap.add_argument("--image", type=Path, default=None, help="single image to classify")
    ap.add_argument("--folder", type=Path, default=None, help="folder of images to classify")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="confidence below this is flagged for manual review")
    args = ap.parse_args()

    if args.image is None and args.folder is None:
        raise SystemExit("ERROR: provide --image PATH or --folder PATH")
    if not args.model.exists():
        raise SystemExit(f"ERROR: model not found at {args.model} — train first: "
                         f"python src/train.py")

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    class_names = json.loads((args.model.parent / "class_names.json").read_text())

    paths = [args.image] if args.image else sorted(
        p for p in args.folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

    print(f"{'image':<32}{'prediction':<15}{'confidence':<12}status")
    print("-" * 69)
    for p in paths:
        probs = model.predict(load_image(p), verbose=0)[0]
        idx = int(np.argmax(probs))
        label, conf = class_names[idx], float(probs[idx])
        status = "OK" if conf >= args.threshold else "LOW CONFIDENCE -> manual review"
        print(f"{p.name:<32}{label:<15}{conf:>8.1%}   {status}")


if __name__ == "__main__":
    main()
