import csv, os
from datetime import datetime

def check_and_log_drift(data_dict):
    is_drift = False
    # The Trap Conditions:
    if data_dict.get('price', 0) <= 0 or data_dict.get('quantity_sold', 0) < 0 or data_dict.get('rating', 0) > 5.0:
        is_drift = True
        
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
