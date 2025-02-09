import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score

PROCESSED_DATA_PATH = "data/processed/cleaned/"
MODEL_PATH = "models/random_forest_model.pkl"

# Load test data
print("📥 Loading test data...")
X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed/X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed/Y_test.csv"))

# Load model
model = joblib.load(MODEL_PATH)
print("🤖 Model Loaded!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.4f}")
