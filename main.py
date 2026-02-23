from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io
import json
from typing import Optional

app = FastAPI(title="EduPredict ML API", version="1.0.0")

# Allow all origins so GitHub Pages frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model store (per session via model_id)
model_store = {}


# ─────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────
class TrainRequest(BaseModel):
    model_id: str
    x_col: str
    enrollment_col: Optional[str] = None
    salary_col: Optional[str] = None
    job_col: Optional[str] = None

class PredictRequest(BaseModel):
    model_id: str
    input_value: float
    prediction_type: str  # "enrollment" | "salary" | "job"

class BulkPredictRequest(BaseModel):
    model_id: str
    input_values: list[float]


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def train_rf_model(X, y):
    """Train Random Forest with fallback to Linear Regression if too few rows."""
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    if len(X) < 10:
        # Too few rows for RF — use Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        model_type = "linear_regression"
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        model_type = "random_forest"

    r2   = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae  = mean_absolute_error(y, preds)
    corr = float(np.corrcoef(X.flatten(), y)[0, 1]) if len(X) > 1 else 0.0

    return {
        "model": model,
        "model_type": model_type,
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "correlation": round(corr, 4),
        "n": int(len(X)),
        "x_values": X.flatten().tolist(),
        "y_values": y.tolist(),
        "predictions": preds.tolist(),
    }


def quality_label(r2):
    if r2 >= 0.9: return "Excellent"
    if r2 >= 0.7: return "Good"
    if r2 >= 0.4: return "Fair"
    return "Weak"


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "EduPredict ML API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV and return column info."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {str(e)}")

    if len(df) < 3:
        raise HTTPException(400, "CSV must have at least 3 rows")

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise HTTPException(400, "Need at least 2 numeric columns")

    # Build column summaries
    col_info = []
    for col in numeric_cols:
        vals = df[col].dropna()
        col_info.append({
            "name": col,
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "samples": vals.head(4).tolist(),
            "count": int(vals.count()),
        })

    # Generate a model_id and store the raw data
    import hashlib, time
    model_id = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()[:10]
    model_store[model_id] = {"df": df, "models": {}}

    # Preview (first 7 rows as list of dicts)
    preview = df.head(7).fillna("").to_dict(orient="records")

    return {
        "model_id": model_id,
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": col_info,
        "all_columns": df.columns.tolist(),
        "preview": preview,
    }


@app.post("/train")
async def train(req: TrainRequest):
    """Train Random Forest models for all selected targets."""
    if req.model_id not in model_store:
        raise HTTPException(404, "Dataset not found. Please upload again.")

    df = model_store[req.model_id]["df"]

    if req.x_col not in df.columns:
        raise HTTPException(400, f"Column '{req.x_col}' not found in dataset")

    X = df[req.x_col].dropna().values
    results = {}

    targets = {
        "enrollment": req.enrollment_col,
        "salary":     req.salary_col,
        "job":        req.job_col,
    }

    trained_any = False
    for pred_type, col in targets.items():
        if not col:
            continue
        if col not in df.columns:
            raise HTTPException(400, f"Column '{col}' not found")

        # Align X and Y (drop rows where either is NaN)
        combined = df[[req.x_col, col]].dropna()
        Xc = combined[req.x_col].values
        Yc = combined[col].values

        if len(Xc) < 3:
            continue

        result = train_rf_model(Xc, Yc)
        model_store[req.model_id]["models"][pred_type] = {
            "fitted": result["model"],
            "model_type": result["model_type"],
            "x_col": req.x_col,
            "y_col": col,
            "x_values": result["x_values"],
            "y_values": result["y_values"],
            "predictions": result["predictions"],
        }
        results[pred_type] = {
            "model_type": result["model_type"],
            "r2": result["r2"],
            "rmse": result["rmse"],
            "mae": result["mae"],
            "correlation": result["correlation"],
            "n": result["n"],
            "quality": quality_label(result["r2"]),
            "x_col": req.x_col,
            "y_col": col,
            "x_values": result["x_values"],
            "y_values": result["y_values"],
            "predictions": result["predictions"],
        }
        trained_any = True

    if not trained_any:
        raise HTTPException(400, "No valid target columns provided")

    return {"status": "trained", "model_id": req.model_id, "models": results}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Predict a single value for a given prediction type."""
    if req.model_id not in model_store:
        raise HTTPException(404, "Model not found. Please upload and train again.")

    models = model_store[req.model_id]["models"]
    if req.prediction_type not in models:
        raise HTTPException(400, f"Model '{req.prediction_type}' not trained yet")

    m = models[req.prediction_type]
    X_input = np.array([[req.input_value]])
    prediction = float(m["fitted"].predict(X_input)[0])

    return {
        "prediction_type": req.prediction_type,
        "input_value": req.input_value,
        "predicted_value": round(prediction, 4),
        "x_col": m["x_col"],
        "y_col": m["y_col"],
        "model_type": m["model_type"],
    }


@app.post("/predict/bulk")
async def bulk_predict(req: BulkPredictRequest):
    """Predict multiple values across all trained models."""
    if req.model_id not in model_store:
        raise HTTPException(404, "Model not found. Please upload and train again.")

    models = model_store[req.model_id]["models"]
    X_input = np.array(req.input_values).reshape(-1, 1)

    results = {}
    for pred_type, m in models.items():
        preds = m["fitted"].predict(X_input)
        results[pred_type] = [round(float(p), 4) for p in preds]

    return {
        "model_id": req.model_id,
        "input_values": req.input_values,
        "predictions": results,
    }


@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    """Get info about stored models for a session."""
    if model_id not in model_store:
        raise HTTPException(404, "Model not found")
    store = model_store[model_id]
    return {
        "model_id": model_id,
        "trained_models": list(store["models"].keys()),
        "rows": len(store["df"]),
    }
