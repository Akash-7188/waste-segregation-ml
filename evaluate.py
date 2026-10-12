"""
evaluate.py — Evaluate the trained model on the held-out test split.

Produces:
    assets/confusion_matrix.png         heatmap of predicted vs. true labels
    reports/classification_report.txt   per-class precision / recall / F1

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/waste_classifier.keras --data_dir data/dataset/test
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/waste_classifier.keras"))
    ap.add_argument("--data_dir", type=Path, default=Path("data/dataset/test"))
    ap.add_argument("--assets_dir", type=Path, default=Path("assets"))
    ap.add_argument("--report_dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)
    class_names = test_ds.class_names

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    accuracy = float(np.trace(cm) / np.sum(cm))

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix (test accuracy {accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(args.assets_dir / "confusion_matrix.png", dpi=150)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=3, zero_division=0)
    (args.report_dir / "classification_report.txt").write_text(report)

    print(f"\nTest accuracy: {accuracy:.2%}\n")
    print(report)
    print(f"Confusion matrix saved to {args.assets_dir / 'confusion_matrix.png'}")
    print(f"Report saved to {args.report_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
