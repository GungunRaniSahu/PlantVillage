# backend — FastAPI service

Loads the trained model and serves predictions to the frontend.

## Setup

> **Use Python 3.11 or 3.12.** TensorFlow has no wheels for Python 3.13/3.14,
> so a venv built on those versions can't install the dependencies.

```bash
cd backend
py -3.11 -m venv .venv        # or python3.11 on macOS/Linux
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tip: when a venv is active, prefer `python -m pip ...` and `python -m uvicorn ...`
so commands always run inside that environment (not a stray global Python).

## Run

```bash
python -m uvicorn main:app --reload
```

API runs at http://localhost:8000

- `GET  /health`  — check if the model is loaded
- `POST /predict` — form-data field `file` = a leaf image
- Interactive docs: http://localhost:8000/docs

### `/predict` response

```json
{
  "disease": "Potato — Early blight",
  "raw_label": "Potato___Early_blight",
  "confidence": 0.98,
  "top3": [
    { "disease": "Potato — Early blight", "confidence": 0.98 },
    { "disease": "Tomato Early blight", "confidence": 0.01 },
    { "disease": "Potato — Late blight", "confidence": 0.01 }
  ],
  "low_confidence": false,
  "heatmap": "data:image/png;base64,..."
}
```

`low_confidence` is `true` when the top score is below `CONFIDENCE_THRESHOLD`
(0.60), so the UI can warn that the image may not be a clear pepper/potato/tomato
leaf. `heatmap` is a Grad-CAM overlay showing where the model looked.

> AI treatment advice (Groq) is currently disabled in `main.py`. The code is
> kept (commented out) if you want to re-enable it later.

> The model is loaded from `backend/model/` (created by training in `../ml`).
> If you haven't trained yet, `/health` will show `model_loaded: false`.
