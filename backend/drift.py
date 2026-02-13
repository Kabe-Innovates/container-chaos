# backend/drift.py
import csv
import os
from datetime import datetime

# Because of Docker volumes, this saves directly to your laptop's data folder
DRIFT_LOG_FILE = 'data/drift_log.csv'

def check_and_log_drift(input_dict: dict) -> bool:
    """Checks for data drift. Returns True if drifted."""
    is_drift = False
    
    # Heuristics for drift (Catching the organizer's bad data)
    # E.g., negative prices, absurd quantities, impossible ratings
    if input_dict.get('price', 0) <= 0 or input_dict.get('discounted_price', 0) < 0:
        is_drift = True
    elif input_dict.get('quantity_sold', 0) < 1 or input_dict.get('quantity_sold', 0) > 500:
        is_drift = True
    elif input_dict.get('rating', 0) < 1.0 or input_dict.get('rating', 0) > 5.0:
        is_drift = True
        
    # If it's garbage, log it so we can ask for labels in Phase 4
    if is_drift:
        _log_to_csv(input_dict)
        
    return is_drift

def _log_to_csv(input_dict: dict):
    os.makedirs(os.path.dirname(DRIFT_LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(DRIFT_LOG_FILE)
    
    with open(DRIFT_LOG_FILE, 'a', newline='') as f:
        # Add a timestamp to the row
        row_data = {'timestamp': datetime.now().isoformat()}
        row_data.update(input_dict)
        
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)