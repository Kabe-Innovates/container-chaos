# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Import our new modules
from backend.metrics import PREDICTION_COUNT, DRIFT_COUNT, get_metrics
from backend.drift import check_and_log_drift

app = FastAPI(title="Amazon Revenue Predictor - Phase 3")

model = joblib.load("models/model_v1.pkl")
feature_cols = joblib.load("models/features.pkl")

class PredictRequest(BaseModel):
    discount_percent: float
    discounted_price: float
    price: float
    quantity_sold: int
    rating: float
    review_count: int

@app.post("/predict")
def predict(request: PredictRequest):
    PREDICTION_COUNT.inc() # Ding! 1 prediction made.
    
    req_dict = request.dict()
    
    # 🚨 Phase 3: Check for Drift before predicting
    if check_and_log_drift(req_dict):
        DRIFT_COUNT.inc() # Sound the alarm!
        print(f"⚠️ DRIFT DETECTED & LOGGED: {req_dict}")

    try:
        df = pd.DataFrame([req_dict])[feature_cols]
        pred = model.predict(df)[0]
        # We still return a prediction so the API doesn't crash during the stress test
        return {"total_revenue": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics_endpoint():
    """Expose metrics for Prometheus."""
    return get_metrics()