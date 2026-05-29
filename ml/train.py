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
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--out-dir", default=os.path.join("..", "backend", "model")
    )
    args = parser.parse_args()

    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    print(f"Found {len(class_names)} classes: {class_names}")

    model = build_model(len(class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, "plant_model.keras"))
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model and class_names.json to: {args.out_dir}")


if __name__ == "__main__":
    main()
