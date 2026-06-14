"""
Plant Disease Detector — model training (transfer learning with MobileNetV2).

Trains an image classifier on the PlantVillage dataset.
For v1, start with ONE crop (e.g. tomato) to keep training fast and easy.

Expected data layout (download PlantVillage from Kaggle, then arrange like this):

    ml/data/
      train/
        Tomato___Early_blight/   *.jpg
        Tomato___Late_blight/    *.jpg
        Tomato___healthy/        *.jpg
        ...
      val/
        Tomato___Early_blight/   *.jpg
        ...                       (same class folders as train)

Run:
    python train.py --data-dir data --epochs 10

Outputs (loaded later by the backend):
    backend/model/plant_model.keras
    backend/model/class_names.json
"""
import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # transfer learning: freeze the pre-trained layers

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def compute_class_weights(data_dir, class_names):
    """Inverse-frequency ('balanced') weights so under-represented classes
    (e.g. Potato_healthy, mosaic virus) aren't drowned out by the big ones."""
    exts = (".jpg", ".jpeg", ".png")
    counts = []
    for name in class_names:
        folder = os.path.join(data_dir, "train", name)
        counts.append(sum(1 for f in os.listdir(folder) if f.lower().endswith(exts)))
    total = sum(counts)
    n = len(class_names)
    weights = {i: total / (n * c) for i, c in enumerate(counts)}
    print("Class weights (inverse-frequency):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {counts[i]} imgs -> weight {weights[i]:.2f}")
    return weights


def fine_tune(model, base, train_ds, val_ds, epochs, unfreeze_layers, class_weight=None):
    """Phase 2: unfreeze the top layers of the conv base and train at a low
    learning rate. Improves accuracy and reduces class confusion (e.g. the
    early-blight recall) that the frozen-base head can't resolve on its own."""
    base.trainable = True
    # Keep the bottom (generic) layers frozen; only fine-tune the top ones.
    for layer in base.layers[:-unfreeze_layers]:
        layer.trainable = False
    # BatchNorm layers should stay in inference mode while fine-tuning.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"\nFine-tuning: {trainable}/{len(base.layers)} base layers unfrozen, lr=1e-5")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=5,
        help="phase-2 epochs with top base layers unfrozen (0 to skip)",
    )
    parser.add_argument(
        "--unfreeze-layers", type=int, default=30,
        help="how many of the top MobileNetV2 layers to unfreeze in phase 2",
    )
    parser.add_argument(
        "--no-class-weights", action="store_true",
        help="disable inverse-frequency class weighting (on by default)",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join("..", "backend", "model")
    )
    args = parser.parse_args()

    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    print(f"Found {len(class_names)} classes: {class_names}")

    class_weight = None if args.no_class_weights else compute_class_weights(args.data_dir, class_names)

    model, base = build_model(len(class_names))

    print(f"\nPhase 1: training head with frozen base ({args.epochs} epochs)")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight)

    if args.fine_tune_epochs > 0:
        fine_tune(model, base, train_ds, val_ds, args.fine_tune_epochs,
                  args.unfreeze_layers, class_weight=class_weight)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, "plant_model.keras"))
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model and class_names.json to: {args.out_dir}")


if __name__ == "__main__":
    main()
