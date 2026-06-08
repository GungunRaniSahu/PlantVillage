"""
Plant Disease Detector — FastAPI backend.

Loads the trained Keras model (from ml training) and exposes:
  GET  /health   -> status check (is the model loaded?)
  POST /predict  -> upload a leaf image; returns disease + confidence + advice

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload
"""
import io
import json
import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

load_dotenv()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
IMG_SIZE = (224, 224)

app = FastAPI(title="Plant Disease Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names: list[str] = []


@app.on_event("startup")
def load_model():
    """Load the trained model + class names if they exist."""
    global model, class_names
    model_path = os.path.join(MODEL_DIR, "plant_model.keras")
    names_path = os.path.join(MODEL_DIR, "class_names.json")

    if os.path.exists(model_path) and os.path.exists(names_path):
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        with open(names_path) as f:
            class_names = json.load(f)
        print(f"Loaded model with {len(class_names)} classes.")
    else:
        print("WARNING: No trained model found. Train one in ../ml first.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded. Train a model in ../ml first.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    except Exception:
        raise HTTPException(400, "Could not read image. Please upload a valid image file.")

    arr = np.expand_dims(np.array(img), axis=0).astype("float32")
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))

    label = class_names[idx]
    confidence = float(preds[idx])

    # Grad-CAM explainability: heatmap of where the model "looked".
    heatmap = None
    try:
        from gradcam import gradcam_overlay_base64

        heatmap = gradcam_overlay_base64(img, arr, model, pred_index=idx)
    except Exception as e:
        print(f"Grad-CAM skipped: {e}")

    return {
        "disease": label.replace("___", " — ").replace("_", " "),
        "raw_label": label,
        "confidence": round(confidence, 4),
        "advice": build_advice(label, confidence),
        "heatmap": heatmap,
    }


def build_advice(label: str, confidence: float) -> str:
    """Use Groq to generate friendly treatment advice (optional)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Set GROQ_API_KEY in backend/.env to enable AI treatment advice."

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        readable = label.replace("___", " ").replace("_", " ")
        prompt = (
            f"A plant leaf was classified as '{readable}' "
            f"(model confidence {confidence:.0%}). "
            "In 2-3 simple sentences for a farmer, explain what this means and "
            "the main treatment steps. If it is healthy, just reassure them briefly."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Could not fetch AI advice: {e})"
