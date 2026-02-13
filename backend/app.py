from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib
from backend.drift import check_and_log_drift

app = FastAPI()
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
    req_dict = request.dict()
    
    # --- PHASE 3: THE DRIFT TRAP ---
    check_and_log_drift(req_dict)
        
    try:
        df = pd.DataFrame([req_dict])[feature_cols]
        pred = model.predict(df)[0]
        return {"total_revenue": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
