import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define paths
data_raw_path = "data/raw"
data_processed_path = "data/processed"
model_path = "models"

# Ensure directories exist
os.makedirs(data_processed_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Load raw data
print("📥 Loading raw data...")
X = pd.read_csv(os.path.join(data_raw_path, "X_data.csv"))
y = pd.read_csv(os.path.join(data_raw_path, "y_data.csv"))

# Check for missing values
print(f"Before cleaning: X shape: {X.shape}, y shape: {y.shape}")
X.dropna(inplace=True)
y = y.loc[X.index]  # Keep labels consistent with X

# Save cleaned data
X.to_csv(os.path.join(data_processed_path, "X_cleaned.csv"), index=False)
y.to_csv(os.path.join(data_processed_path, "y_cleaned.csv"), index=False)
print(f"✅ After cleaning: X shape: {X.shape}, y shape: {y.shape}")

# Split data
print("✂️ Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
print("📊 Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
pd.DataFrame(X_train_scaled).to_csv(os.path.join(data_processed_path, "X_train.csv"), index=False)
pd.DataFrame(X_test_scaled).to_csv(os.path.join(data_processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(data_processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(data_processed_path, "y_test.csv"), index=False)
print("✅ Data preparation complete!")

# Train model
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, os.path.join(model_path, "random_forest.pkl"))
print("✅ Model training complete and saved!")

# Evaluate model
print("📊 Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")
print("📄 Classification Report:")
print(classification_report(y_test, y_pred))
