from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

app = FastAPI(title="Amazon Revenue Predictor")

# Load model and feature list from the mounted volume
Config.ensure_dirs_exist()
MODEL_PATH = Config.get_model_path()
FEATURES_PATH = Config.get_features_path()

try:
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Model not loaded. Error: {e}")
    model = None

# The exact schema required by the judges
class PredictRequest(BaseModel):
    discount_percent: float
    discounted_price: float
    price: float
    quantity_sold: int
    rating: float
    review_count: int

@app.post("/predict")
def predict_revenue(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is missing!")
    
    try:
        # 1. Convert incoming JSON to a Pandas DataFrame (1 row)
        input_data = request.dict()
        df = pd.DataFrame([input_data])
        
        # 2. Enforce the exact column order used during training
        df = df[feature_cols]
        
        # 3. Predict
        prediction = model.predict(df)[0]
        
        # 4. Return the required contract
        return {"total_revenue": round(prediction, 2)}
        
    except Exception as e:
        # NEVER crash the server. Catch the error and return a 500.
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "alive", "model_loaded": model is not None}