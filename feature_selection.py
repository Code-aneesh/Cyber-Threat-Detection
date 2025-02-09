import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# Load Processed Data
# -----------------------
print("\n📥 Loading processed data...")
X_train = pd.read_csv("data/processed/X_train.csv").values
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

print(f"✅ Data Loaded: X_train: {X_train.shape}, y_train: {y_train.shape}")

# -----------------------
# Feature Selection (Using Chi-Squared Test)
# -----------------------
print("\n🔍 Performing feature selection...")

# Select K best features (e.g., top 10)
k = 10  # You can adjust this value based on your requirements
selector = SelectKBest(chi2, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the selected feature indices
selected_features = selector.get_support(indices=True)
selected_feature_names = [f"Feature {i+1}" for i in selected_features]
print(f"✅ Top {k} Features Selected: {selected_feature_names}")

# -----------------------
# Save Selected Features
# -----------------------
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Saving the selected features and labels
pd.DataFrame(X_train_selected).to_csv(f"{output_dir}/X_train_selected.csv", index=False)
pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train_selected.csv", index=False)

print(f"✅ Feature selection complete. Saved selected features at {output_dir}/X_train_selected.csv")

# -----------------------
# Optional: Feature Scaling after Selection
# -----------------------
print("\n📊 Scaling selected features...")
scaler = MinMaxScaler()
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
print("✅ Scaling complete!")

# Save the scaled selected features
pd.DataFrame(X_train_selected_scaled).to_csv(f"{output_dir}/X_train_selected_scaled.csv", index=False)
print(f"✅ Scaled selected features saved at {output_dir}/X_train_selected_scaled.csv")
