import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
import joblib

# -----------------------
# Load Preprocessed Datasets
# -----------------------
print("\n🔄 Merging both datasets...")
unsw_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/projects/cyber/data/cleaned/UNSW_NB15_clean.csv")
cicids_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/projects/cyber/data/cleaned/CICIDS2017_clean.csv")

# Merge datasets
dataset = pd.concat([unsw_df, cicids_df], ignore_index=True)
print("✅ Datasets merged!")

# -----------------------
# Separate Features & Labels
# -----------------------
print("\n🔍 Splitting features and labels...")
X = dataset.drop(columns=['label'])  # Features
y = dataset['label']  # Target Label

# Check and fix NaN values in target (y)
initial_nan_y = y.isna().sum()
print(f"⚠️ Initial NaN in y: {initial_nan_y} / {len(y)}")

# Drop rows where y is NaN
dataset = dataset.dropna(subset=['label'])
X = dataset.drop(columns=['label'])
y = dataset['label']

print(f"✅ After Fix: NaN in y: {y.isna().sum()} / {len(y)}")

# -----------------------
# Encode Categorical Columns
# -----------------------
print("\n🛠 Encoding categorical features...")

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Columns: {categorical_columns}")

# Apply Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

print("✅ Categorical encoding complete!")

# -----------------------
# Handle Missing & Infinite Values
# -----------------------
print("\n🛠 Checking for missing or infinite values...")

# Drop columns where all values are NaN
all_nan_columns = X.columns[X.isna().all()].tolist()
if all_nan_columns:
    X.drop(columns=all_nan_columns, inplace=True)
    print(f"✅ Dropped all-NaN columns, remaining features: {X.shape[1]}")
else:
    print(f"✅ No all-NaN columns found. Features count: {X.shape[1]}")

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill remaining NaN values with 0
X.fillna(0, inplace=True)

print("✅ Fixed missing/infinite values!")

# -----------------------
# Scale Features
# -----------------------
print("\n📊 Scaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Feature scaling complete!")

# -----------------------
# Split Data (80% Train, 20% Test)
# -----------------------
print("\n🔀 Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("✅ Data splitting complete!")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

# Final NaN check
print(f"🔍 After Split: NaN in y_train: {np.isnan(y_train).sum()}, NaN in y_test: {np.isnan(y_test).sum()}")

# -----------------------
# Save Processed Data
# -----------------------
output_dir = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

print("\n✅ Feature Scaling & Data Splitting Complete!")
