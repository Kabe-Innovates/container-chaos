import pandas as pd
import joblib
import os
import requests
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Add parent directory to path to import Config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Configuration
DATA_PATH = "../data/final_X_and_Y.csv"
CALLBACK_URL = "https://sample.com"
MODEL_V2_PATH = "../models/model_v2.pkl"

def run_final_retrain():
    print(f"📂 Loading final dataset: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("❌ Error: final_X_and_Y.csv not found in data/ directory.")
        return

    # 1. Load and Extract specific fields
    # Fields: customer_region, discount_percent, discounted_price, order_date, order_id, 
    # payment_method, price, product_category, product_id, quantity_sold, rating, 
    # review_count, total_revenue
    df = pd.read_csv(DATA_PATH)
    
    extracted_fields = [
        "customer_region", "discount_percent", "discounted_price", "order_date", 
        "order_id", "payment_method", "price", "product_category", "product_id", 
        "quantity_sold", "rating", "review_count", "total_revenue"
    ]
    
    # Check if all fields exist
    available_fields = [f for f in extracted_fields if f in df.columns]
    df_extracted = df[available_fields]
    print(f"✅ Extracted {len(available_fields)} fields.")

    # 2. Prepare Training Data
    # We use the features defined in your Config
    features = Config.get_features()
    target = Config.get_target_column()
    
    X = df_extracted[features].fillna(0)
    y = df_extracted[target].fillna(0)

    # 3. Retrain the Model
    print("🧠 Retraining Model (V2) with final dataset...")
    model = RandomForestRegressor(
        n_estimators=Config.get_n_estimators(), 
        random_state=Config.get_random_state()
    )
    model.fit(X, y)

    # 4. Local Evaluation
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"📊 Final Model MAE: {mae:.4f}")

    # 5. Save the new model
    joblib.dump(model, MODEL_V2_PATH)
    print(f"💾 Model saved to {MODEL_V2_PATH}")

    # 6. Send Result JSON to Callback URL
    result_payload = {
        "status": "retraining_complete",
        "model_version": "v2",
        "metrics": {
            "mae": round(mae, 4)
        },
        "data_points": len(df_extracted),
        "features_used": features
    }

    try:
        print(f"📡 Sending results to {CALLBACK_URL}...")
        response = requests.post(CALLBACK_URL, json=result_payload)
        if response.status_code == 200:
            print("✅ Successfully notified the host.")
        else:
            print(f"⚠️ Host responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to send JSON: {e}")

if __name__ == "__main__":
    run_final_retrain()