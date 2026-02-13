import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

print("🚀 Starting Phase 1: Model Training...")

# 1. Load data
df = pd.read_csv('../data/raw.csv')

# 2. Select features (Numerical only for V1 speed!)
features = [
    'discount_percent', 
    'discounted_price', 
    'price', 
    'quantity_sold', 
    'rating', 
    'review_count'
]
target = 'total_revenue'

# Fill any missing values with 0 (Bulletproofing against dirty data)
X = df[features].fillna(0)
y = df[target].fillna(0)

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("🧠 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Quick Evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"📊 Model Evaluation - Mean Absolute Error: {mae:.2f}")

# 6. Save Model and Feature List
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/model_v1.pkl')

# CRITICAL: We save the feature list so our FastAPI knows the exact column order later
joblib.dump(features, '../models/features.pkl')

print("💾 SUCCESS: Model saved to ../models/model_v1.pkl")