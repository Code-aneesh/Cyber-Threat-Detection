import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import joblib

# ✅ Load Data
print("📥 Loading Data...")
X_train = pd.read_csv("data/processed/X_train_selected.csv").values
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test_selected.csv").values
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"✅ Loaded Data: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ Loaded Data: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 🔍 Check for NaN values
print("🔍 Checking for NaN values in y_train and y_test...")
nan_count_train = np.isnan(y_train).sum()
nan_count_test = np.isnan(y_test).sum()
print(f"NaN values in y_train: {nan_count_train}, NaN values in y_test: {nan_count_test}")

# ✅ Handle NaN values in y_train
if nan_count_train > 0:
    print("⚠️ NaN values detected in y_train! Handling them...")
    y_train = y_train[~np.isnan(y_train)]
    X_train = X_train[:len(y_train)]  # Adjust X_train size
    print(f"✅ NaN values handled! Updated X_train: {X_train.shape}, y_train: {y_train.shape}")

# ✅ Handle NaN values in y_test
if nan_count_test > 0:
    print("⚠️ NaN values detected in y_test! Handling them...")
    y_test = y_test[~np.isnan(y_test)]
    X_test = X_test[:len(y_test)]  # Adjust X_test size
    print(f"✅ NaN values handled! Updated X_test: {X_test.shape}, y_test: {y_test.shape}")

# 🚀 Train Random Forest Model
print("🚀 Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model Training Complete!")

# 📊 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy on Test Data: {accuracy:.4f}")

# ✅ Save Model
joblib.dump(model, "models/random_forest_model.pkl")
print("💾 Model saved as models/random_forest_model.pkl")
