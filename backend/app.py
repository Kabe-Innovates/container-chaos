from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from backend.drift import check_and_log_drift

app = FastAPI()
Config.ensure_dirs_exist()
model = joblib.load(Config.get_model_path())
feature_cols = joblib.load(Config.get_features_path())

class PredictRequest(BaseModel):
    discount_percent: float
    discounted_price: float
    price: float
    quantity_sold: int
    rating: float
    review_count: int

@app.post("/predict")
def predict(request: PredictRequest):
    req_dict = request.dict()
    
    # --- PHASE 3: THE DRIFT TRAP ---
    check_and_log_drift(req_dict)
        
    try:
        df = pd.DataFrame([req_dict])[feature_cols]
        pred = model.predict(df)[0]
        return {"total_revenue": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
