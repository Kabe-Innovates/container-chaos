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
from backend.drift import check_and_log_drift, log_audit_data

app = FastAPI(title="Amazon Revenue Predictor - Model V2")

# Ensure directories exist and load model via Config
Config.ensure_dirs_exist()
try:
    model = joblib.load(Config.get_model_path())
    feature_cols = joblib.load(Config.get_features_path())
except Exception as e:
    model = None

class PredictRequest(BaseModel):
    discount_percent: float
    discounted_price: float
    price: float
    quantity_sold: int
    rating: float
    review_count: int

@app.post("/predict")
@app.post("/predict/")
def predict(request: Union[PredictRequest, List[PredictRequest]]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file missing")

    # Normalize input for batch processing
    is_batch = isinstance(request, list)
    requests_list = request if is_batch else [request]
    req_dicts = [req.dict() for req in requests_list]
    
    # Background Logging & Drift Detection
    for req_dict in req_dicts:
        log_audit_data(req_dict)
        check_and_log_drift(req_dict)
            
    try:
        # Inference using updated model_v2 features
        df = pd.DataFrame(req_dicts)[feature_cols]
        preds = model.predict(df)
        
        # Format strictly as JSON results
        results = [{"total_revenue": round(float(p), 2)} for p in preds]
        return results if is_batch else results[0]
        
    except Exception:
        # Generic error to avoid leaking system info in HTML
        raise HTTPException(status_code=500, detail="Internal Prediction Error")

@app.get("/health")
def health():
    return {"status": "alive", "model": os.getenv('MODEL_NAME')}