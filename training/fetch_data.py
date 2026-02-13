import pandas as pd
import requests
import os
import sys
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Ensure data directory exists
Config.ensure_dirs_exist()

url = "https://container-chaos-b2c6bde72854.herokuapp.com/api/amazon/train-test"
print(f"📡 Fetching data from {url}...")

response = requests.get(url)

if response.status_code == 200:
    try:
        # Try to parse as JSON first
        data = response.json()
        # If it's a dictionary with a specific key (like 'data'), adjust accordingly.
        # Assuming it's a flat list of records:
        if isinstance(data, dict):
             # Grab the first key if it's nested (e.g., {"train": [...]})
             key = list(data.keys())[0]
             df = pd.DataFrame(data[key])
        else:
             df = pd.DataFrame(data)
    except:
        # Fallback to CSV parsing
        df = pd.read_csv(StringIO(response.text))
    
    # Save the raw data
    raw_path = Config.get_raw_data_path()
    df.to_csv(raw_path, index=False)
    
    print(f"✅ Data successfully saved to {raw_path}")
    print("\n" + "="*30)
    print("🚨 CRITICAL: DATA TYPES DETECTED")
    print("="*30)
    print(df.dtypes)
    print("\n" + "="*30)
    print("FIRST 2 ROWS")
    print("="*30)
    print(df.head(2))
else:
    print(f"❌ FAILED. Status Code: {response.status_code}")
    print(response.text)