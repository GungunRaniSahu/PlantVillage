"""
Plant Disease Detector — model evaluation.

Loads the trained model and measures how good it actually is on the
validation set: overall accuracy, a per-class precision / recall / F1
report, and a confusion-matrix image you can drop into your README.

Run (from the ml/ folder, with .venv activated):
    python evaluate.py
    python evaluate.py --data-dir data --split val

Outputs (written to ml/reports/ by default):
    reports/metrics.json          machine-readable metrics
    reports/classification_report.txt
    reports/confusion_matrix.png  pretty heatmap of predictions vs. truth
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")  # no display needed; just save files
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_eval_dataset(data_dir, split, class_names):
    """Load the eval split WITHOUT shuffling so labels line up with preds."""
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, split),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=class_names,  # force the model's training order
    )
    return ds


def plot_confusion_matrix(cm, class_names, out_path, normalize=True):
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (row-normalized)" if normalize else "Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8,
                color="white" if val > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="val", help="subfolder to evaluate (val/test)")
    ap.add_argument(
        "--model", default=os.path.join("..", "backend", "model", "plant_model.keras")
    )
    ap.add_argument(
        "--class-names",
        default=os.path.join("..", "backend", "model", "class_names.json"),
    )
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    with open(args.class_names) as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes.")

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    ds = load_eval_dataset(args.data_dir, args.split, class_names)

    print("Running predictions...")
    y_true, y_pred, confidences = [], [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        confidences.extend(np.max(probs, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    acc = accuracy_score(y_true, y_pred)
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(args.out_dir, exist_ok=True)

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, normalize=True)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report_txt}\n")

    metrics = {
        "split": args.split,
        "num_samples": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "per_class": {
            name: {
                "precision": round(report_dict[name]["precision"], 4),
                "recall": round(report_dict[name]["recall"], 4),
                "f1": round(report_dict[name]["f1-score"], 4),
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- summary to console ----
    print("\n" + "=" * 48)
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print(f"  Mean confidence : {metrics['mean_confidence']:.4f}")
    print(f"  Eval samples    : {metrics['num_samples']}")
    print("=" * 48)
    print(report_txt)
    print(f"Saved: {cm_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"Saved: {os.path.join(args.out_dir, 'classification_report.txt')}")


if __name__ == "__main__":
    main()
