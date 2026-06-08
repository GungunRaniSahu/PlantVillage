"""
Grad-CAM for the Plant Disease Detector.

Grad-CAM produces a heatmap showing *which part of the leaf* the model
looked at when making its prediction — great for explaining the model's
decision (and it looks fantastic in a demo / README).

This handles the fact that our model wraps a *nested* MobileNetV2 submodel:
we split the network around that base so gradients can flow back to the
last convolutional feature maps.

CLI:
    python gradcam.py path/to/leaf.jpg
    python gradcam.py leaf.jpg --out heatmap.png --alpha 0.5

Importable:
    from gradcam import make_gradcam_heatmap, overlay_heatmap
"""
import argparse
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


# --------------------------------------------------------------------------
# Core Grad-CAM
# --------------------------------------------------------------------------
def _find_base_submodel(model):
    """Find the nested MobileNetV2 functional submodel inside the model."""
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    # Prefer the one that looks like MobileNet; else the deepest sub-model.
    for l in candidates:
        if "mobilenet" in l.name.lower():
            return l
    if candidates:
        return max(candidates, key=lambda l: len(l.layers))
    raise ValueError("No nested conv base (Model) found inside this model.")


def _split_around_base(model, base):
    """Return (pre_layers, post_layers): layers before and after the base."""
    layers = model.layers
    base_idx = layers.index(base)
    # Skip the InputLayer (index 0); it just defines the input tensor.
    pre = [l for l in layers[:base_idx] if not isinstance(l, tf.keras.layers.InputLayer)]
    post = layers[base_idx + 1:]
    return pre, post


def _call(layer, x):
    """Call a layer in inference mode, tolerating layers w/o a training kwarg."""
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Compute a Grad-CAM heatmap.

    img_array : (1, H, W, 3) float32, RAW 0-255 pixels (same as the backend
                feeds the model — preprocessing happens inside the graph).
    Returns   : (heatmap [h, w] in 0..1, predicted_index, confidence)
    """
    base = _find_base_submodel(model)
    pre_layers, post_layers = _split_around_base(model, base)

    # Feature maps = output of the conv base (7x7x1280 for MobileNetV2).
    base_features = tf.keras.models.Model(base.input, base.output)

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    for layer in pre_layers:  # data augmentation (no-op at inference) + preprocess
        x = _call(layer, x)

    with tf.GradientTape() as tape:
        conv_out = base_features(x, training=False)
        tape.watch(conv_out)
        y = conv_out
        for layer in post_layers:  # GAP -> dropout -> dense(softmax)
            y = _call(layer, y)
        preds = y
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importance per channel

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    confidence = float(preds[0, pred_index])
    return heatmap.numpy(), int(pred_index), confidence


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def overlay_heatmap(pil_image, heatmap, alpha=0.4, colormap="jet"):
    """Blend a heatmap over the original PIL image. Returns a PIL.Image (RGB)."""
    from PIL import Image
    import matplotlib.cm as cm

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    base = np.asarray(img).astype("float32")

    hm = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    hm = np.asarray(hm).astype("float32") / 255.0

    color = cm.get_cmap(colormap)(hm)[..., :3] * 255.0  # (H, W, 3)
    blended = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype("uint8")
    return Image.fromarray(blended)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    import json
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a leaf image")
    ap.add_argument(
        "--model", default=os.path.join("..", "backend", "model", "plant_model.keras")
    )
    ap.add_argument(
        "--class-names",
        default=os.path.join("..", "backend", "model", "class_names.json"),
    )
    ap.add_argument("--out", default="gradcam.png")
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.class_names) as f:
        class_names = json.load(f)

    pil = Image.open(args.image).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(pil), axis=0).astype("float32")

    heatmap, idx, conf = make_gradcam_heatmap(arr, model)
    overlay = overlay_heatmap(pil, heatmap, alpha=args.alpha)
    overlay.save(args.out)

    print(f"Prediction : {class_names[idx]}  ({conf*100:.1f}% confidence)")
    print(f"Saved heatmap overlay -> {args.out}")


if __name__ == "__main__":
    main()
