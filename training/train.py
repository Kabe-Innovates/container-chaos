import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

print("🚀 Starting Phase 1: Model Training...")

Config.ensure_dirs_exist()

# 1. Load data
df = pd.read_csv(Config.get_raw_data_path())

# 2. Select features
features = Config.get_features()
target = Config.get_target_column()

# Fill any missing values with 0 (Bulletproofing against dirty data)
X = df[features].fillna(0)
y = df[target].fillna(0)

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=Config.get_test_size(), 
    random_state=Config.get_random_state()
)

# 4. Train the Model
print("🧠 Training Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=Config.get_n_estimators(), 
    random_state=Config.get_random_state()
)
model.fit(X_train, y_train)

# 5. Quick Evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"📊 Model Evaluation - Mean Absolute Error: {mae:.2f}")

# 6. Save Model and Feature List
joblib.dump(model, Config.get_model_path())

# CRITICAL: We save the feature list so our FastAPI knows the exact column order later
joblib.dump(features, Config.get_features_path())

print(f"💾 SUCCESS: Model saved to {Config.get_model_path()}")