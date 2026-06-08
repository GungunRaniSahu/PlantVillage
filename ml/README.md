# ml — Model training

Trains the plant-disease image classifier and saves it for the backend to use.

## 1. Get the dataset

Download the **PlantVillage** dataset (free) from Kaggle, e.g.
search "PlantVillage dataset" on https://www.kaggle.com/datasets.

For v1, keep only **one crop** (e.g. tomato classes) to start small.

Arrange the images like this (you split into train/val yourself, ~80/20):

```
ml/data/
  train/
    Tomato___Early_blight/   *.jpg
    Tomato___Late_blight/    *.jpg
    Tomato___healthy/        *.jpg
    ...
  val/
    Tomato___Early_blight/   *.jpg
    ...
```

> `ml/data/` is gitignored — datasets are too large to commit.

## 2. Install dependencies

```bash
cd ml
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

> Tip: training is much faster on a GPU. If you don't have one, use
> **Google Colab** (free GPU) — upload this script + your data there.

## 3. Train

```bash
python train.py --data-dir data --epochs 10
```

This writes the trained model to `../backend/model/`:
- `plant_model.keras` — the model
- `class_names.json` — the list of disease labels (order matters)

The backend reads these automatically on startup.

## 4. Evaluate (accuracy + confusion matrix)

Measure how good the trained model actually is on the validation set:

```bash
python evaluate.py
```

This writes to `ml/reports/`:
- `metrics.json` — accuracy, macro/weighted F1, mean confidence, per-class scores
- `classification_report.txt` — precision / recall / F1 per class
- `confusion_matrix.png` — a heatmap of predictions vs. truth (great for the README)

## 5. Grad-CAM (explainability)

See *which part of the leaf* the model focused on for its prediction:

```bash
python gradcam.py path/to/leaf.jpg --out gradcam.png
```

The same logic is wired into the backend, so the web app shows the heatmap
next to every prediction automatically.

## Next steps / research extensions

- Expand from one crop to all 38 PlantVillage classes.
- Fine-tune (unfreeze the top layers of MobileNetV2) for higher accuracy.
- ~~Add Grad-CAM heatmaps~~ ✅ done (`gradcam.py` + backend integration).
- ~~Report accuracy + confusion matrix~~ ✅ done (`evaluate.py`).
