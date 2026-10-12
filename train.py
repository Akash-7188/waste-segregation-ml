"""
train.py — Train the waste image classifier with transfer learning.

Phase 1 (default): MobileNetV2 base (pre-trained on ImageNet) is FROZEN and
    only the new classification head is trained. Fast, needs little data.
Phase 2 (optional, --finetune): the top 30 layers of the base are unfrozen
    and re-trained very gently (lr = 1e-5) to squeeze out extra accuracy.

Usage:
    python src/train.py --epochs 20            # phase 1
    python src/train.py --epochs 5 --finetune  # phase 2 (run after phase 1)

Outputs:
    models/waste_classifier.keras    trained model
    models/class_names.json          class names in output order
    assets/training_history.png      accuracy/loss curves
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works on machines without a display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (224, 224)


def build_model(n_classes: int) -> tf.keras.Model:
    """MobileNetV2 (frozen, ImageNet weights) + custom classification head."""
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False  # phase 1: don't touch what ImageNet already taught it

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # pixels -> [-1, 1] as MobileNetV2 expects
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)                        # prevents memorising (overfitting)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="waste_classifier")


def plot_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history.history[key], label=f"train {key}")
        ax.plot(history.history[f"val_{key}"], label=f"val {key}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Training curves saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/dataset"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--finetune", action="store_true",
                    help="phase 2: unfreeze the top 30 base layers and re-train gently")
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--assets_dir", type=Path, default=Path("assets"))
    args = ap.parse_args()

    # ---------- load data ----------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir / "train", image_size=IMG_SIZE,
        batch_size=args.batch, shuffle=True, seed=42)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir / "val", image_size=IMG_SIZE, batch_size=args.batch)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model_path = args.out_dir / "waste_classifier.keras"

    # ---------- build / load model ----------
    if args.finetune:
        if not model_path.exists():
            raise SystemExit("ERROR: no model to fine-tune — run phase 1 first "
                             "(python src/train.py --epochs 20)")
        model = tf.keras.models.load_model(model_path)
        base = model.get_layer("mobilenetv2_1.00_224")
        base.trainable = True
        for layer in base.layers[:-30]:  # keep all but the top 30 layers frozen
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # very small LR!
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print("\n=== Phase 2: fine-tuning the top 30 base layers (lr = 1e-5) ===")
    else:
        model = build_model(len(class_names))
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print("\n=== Phase 1: training the classification head (MobileNetV2 frozen) ===")

    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_accuracy")]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)

    # ---------- save outputs ----------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    (args.out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    plot_history(history, args.assets_dir / "training_history.png")

    best_val = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val:.2%}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
