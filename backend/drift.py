import csv
import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# --- STARTUP: TRAIN THE DYNAMIC DETECTOR ---
drift_detector = None
features = Config.get_features()

try:
    # Uses the latest raw data path from Config
    baseline_path = Config.get_raw_data_path()
    baseline_df = pd.read_csv(baseline_path)
    X_baseline = baseline_df[features].fillna(0)
    
    drift_detector = IsolationForest(
        random_state=Config.get_random_state(), 
        contamination=Config.get_contamination_rate()
    )
    drift_detector.fit(X_baseline)
except Exception:
    pass

def log_to_csv(data_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        row = {'timestamp': datetime.now().isoformat()}
        row.update(data_dict)
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)

def log_audit_data(data_dict):
    audit_log_path = os.path.join(Config.get_data_dir(), "audit_log.csv")
    log_to_csv(data_dict, audit_log_path)

def check_and_log_drift(data_dict):
    is_drift = False
    # Logic remains consistent for v2 monitoring
    if data_dict.get('price', 0) <= 0 or data_dict.get('quantity_sold', 0) < 0:
        is_drift = True
    elif data_dict.get('discounted_price', 0) > data_dict.get('price', 0):
        is_drift = True
        
    if not is_drift and drift_detector is not None:
        try:
            df_input = pd.DataFrame([data_dict])[features].fillna(0)
            if drift_detector.predict(df_input)[0] == -1:
                is_drift = True
        except Exception: pass

    if is_drift:
        log_to_csv(data_dict, Config.get_drift_log_path())
    return is_drift