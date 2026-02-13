from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import joblib
import os
import sys

# Load Config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Load Drift and Metrics
from backend.drift import check_and_log_drift
from backend.metrics import PREDICTION_COUNT, DRIFT_COUNT, get_metrics

app = FastAPI(title="Amazon Revenue Predictor - Enterprise Edition")

# Ensure directories exist and load model via Config
Config.ensure_dirs_exist()
try:
    model = joblib.load(Config.get_model_path())
    feature_cols = joblib.load(Config.get_features_path())
    print("✅ Model loaded successfully via Config.")
except Exception as e:
    print(f"⚠️ Warning: Model not loaded. Error: {e}")
    model = None

class PredictRequest(BaseModel):
    discount_percent: float
    discounted_price: float
    price: float
    quantity_sold: int
    rating: float
    review_count: int

@app.post("/predict")
def predict(request: Union[PredictRequest, List[PredictRequest]]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is missing!")

    # 1. Normalize input to a list so we can process batches easily
    is_batch = isinstance(request, list)
    requests_list = request if is_batch else [request]
    req_dicts = [req.dict() for req in requests_list]
    
    # 2. Update Metrics & Check Drift for every item in the batch
    PREDICTION_COUNT.inc(len(req_dicts))
    for req_dict in req_dicts:
        # --- PHASE 3: THE DRIFT TRAP ---
        if check_and_log_drift(req_dict):
            DRIFT_COUNT.inc()
            
    try:
        # 3. Batch Predict (Blazing fast in Pandas/Sklearn)
        df = pd.DataFrame(req_dicts)[feature_cols]
        preds = model.predict(df)
        
        # 4. Format Output
        results = [{"total_revenue": round(float(p), 2)} for p in preds]
        
        # Return a list if they sent a list, or a single dict if they sent a single dict
        return results if is_batch else results[0]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return get_metrics()

@app.get("/health")
def health():
    return {"status": "alive", "model_loaded": model is not None}