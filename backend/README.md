# backend — FastAPI service

Loads the trained model and serves predictions to the frontend.

## Setup

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## (Optional) Enable AI treatment advice

```bash
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env and paste your GROQ_API_KEY
```

The backend auto-loads `.env` via python-dotenv if present. If no key is set,
predictions still work — they just return a placeholder instead of AI advice.

## Run

```bash
uvicorn main:app --reload
```

API runs at http://localhost:8000

- `GET  /health`  — check if the model is loaded
- `POST /predict` — form-data field `file` = a leaf image; returns disease + confidence + advice
- Interactive docs: http://localhost:8000/docs

> The model is loaded from `backend/model/` (created by training in `../ml`).
> If you haven't trained yet, `/health` will show `model_loaded: false`.
