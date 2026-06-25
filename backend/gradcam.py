"""
Grad-CAM helper for the backend (self-contained, numpy-only colormap).

Produces a heatmap overlay showing which region of the leaf drove the
prediction. Kept dependency-light (no matplotlib) so the API stays lean.
"""
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)


def _find_base_submodel(model):
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    for l in candidates:
        if "mobilenet" in l.name.lower():
            return l
    if candidates:
        return max(candidates, key=lambda l: len(l.layers))
    raise ValueError("No nested conv base found inside this model.")


def _split_around_base(model, base):
    layers = model.layers
    base_idx = layers.index(base)
    pre = [l for l in layers[:base_idx] if not isinstance(l, tf.keras.layers.InputLayer)]
    post = layers[base_idx + 1:]
    return pre, post


def _call(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """img_array: (1, H, W, 3) float32 raw 0-255. Returns heatmap [h, w] in 0..1."""
    base = _find_base_submodel(model)
    pre_layers, post_layers = _split_around_base(model, base)
    base_features = tf.keras.models.Model(base.input, base.output)

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    for layer in pre_layers:
        x = _call(layer, x)

    last = len(post_layers) - 1
    with tf.GradientTape() as tape:
        conv_out = base_features(x, training=False)
        tape.watch(conv_out)
        y = conv_out
        for i, layer in enumerate(post_layers):
            # For the final Dense, use PRE-softmax logits: when the model is
            # near 100% the softmax gradient ~0, which makes the heatmap noise.
            if i == last and isinstance(layer, tf.keras.layers.Dense):
                y = tf.linalg.matmul(y, layer.kernel)
                if layer.use_bias:
                    y = y + layer.bias
            else:
                y = _call(layer, y)
        preds = y
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.squeeze(conv_out @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _jet_colormap(values):
    """Map values in 0..1 to a jet-like RGB array (..., 3), uint8 — no matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    four = 4.0 * v
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0, 1)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0, 1)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype("uint8")


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """Blend the heatmap over the original image. Returns a PIL.Image (RGB)."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    base = np.asarray(img).astype("float32")

    hm = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    hm = np.asarray(hm).astype("float32") / 255.0

    color = _jet_colormap(hm).astype("float32")
    blended = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype("uint8")
    return Image.fromarray(blended)


def gradcam_overlay_base64(pil_image, img_array, model, pred_index=None, alpha=0.4):
    """Convenience: heatmap -> overlay -> 'data:image/png;base64,...' string."""
    import base64
    import io

    heatmap = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
    overlay = overlay_heatmap(pil_image, heatmap, alpha=alpha)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
