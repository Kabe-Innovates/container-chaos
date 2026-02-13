import csv
import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest

# --- STARTUP: TRAIN THE DYNAMIC DETECTOR ---
print("🌲 Booting Dynamic Drift Detector (Isolation Forest)...")
drift_detector = None
features = ['discount_percent', 'discounted_price', 'price', 'quantity_sold', 'rating', 'review_count']

try:
    # 1. Read the baseline data you trained Model V1 on
    baseline_df = pd.read_csv('data/raw.csv')
    X_baseline = baseline_df[features].fillna(0)
    
    # 2. Train the Forest to recognize "normal" data
    # contamination=0.01 means we assume 1% of our training data might be natural outliers
    drift_detector = IsolationForest(random_state=42, contamination=0.01)
    drift_detector.fit(X_baseline)
    print("✅ Dynamic Drift Detector Armed and Ready.")
except Exception as e:
    print(f"⚠️ Warning: Could not train dynamic drift detector: {e}")


# --- THE TRAP LOGIC ---
def check_and_log_drift(data_dict):
    is_drift = False
    
    # 1. HARD CONSTRAINTS (The Safety Net)
    # Even ML models make mistakes, so we keep the absolute physics laws
    if data_dict.get('price', 0) <= 0 or data_dict.get('quantity_sold', 0) < 0:
        is_drift = True
    elif data_dict.get('discounted_price', 0) > data_dict.get('price', 0):
        is_drift = True
        
    # 2. DYNAMIC ML DETECTION (The Magic)
    if not is_drift and drift_detector is not None:
        try:
            # Convert incoming JSON to a DataFrame safely
            df_input = pd.DataFrame([data_dict])
            # Ensure it only has the features we trained the forest on
            df_input = df_input[features].fillna(0)
            
            # Predict: 1 = Normal Data | -1 = Anomaly (Drift)
            prediction = drift_detector.predict(df_input)[0]
            if prediction == -1:
                print(f"🕵️‍♂️ DYNAMIC DRIFT CAUGHT: {data_dict}")
                is_drift = True
        except Exception as e:
            print(f"⚠️ Dynamic detector failed on input: {e}")

    # 3. LOGGING
    if is_drift:
        os.makedirs('data', exist_ok=True)
        file_exists = os.path.isfile('data/drift_log.csv')
        with open('data/drift_log.csv', 'a', newline='') as f:
            row = {'timestamp': datetime.now().isoformat()}
            row.update(data_dict)
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    return is_drift