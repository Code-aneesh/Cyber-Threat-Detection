import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# -----------------------
# Load Preprocessed Datasets
# -----------------------
print("\n🔄 Loading preprocessed datasets...")
unsw_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/projects/cyber/data/cleaned/UNSW_NB15_clean.csv")
cicids_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/projects/cyber/data/cleaned/CICIDS2017_clean.csv")

# Merge datasets
print("\n🔄 Merging both datasets...")
dataset = pd.concat([unsw_df, cicids_df], ignore_index=True)
print("✅ Datasets merged!")

# -----------------------
# Separate Features & Labels
# -----------------------
print("\n🔍 Splitting features and labels...")
X = dataset.drop(columns=['label'])  # Features
y = dataset['label']  # Target Label (Attack or Normal)

# Drop rows where target label is NaN
y_nan_count = y.isna().sum()
print(f"⚠️ Initial NaN in y: {y_nan_count} / {len(y)}")
dataset.dropna(subset=['label'], inplace=True)
X = dataset.drop(columns=['label'])
y = dataset['label']
print(f"✅ After Fix: NaN in y: {y.isna().sum()} / {len(y)}")

# -----------------------
# Encoding Categorical Columns
# -----------------------
print("\n🛠 Identifying categorical features...")
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Columns: {categorical_columns}")

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    # Handling missing values by replacing NaN with a placeholder value before encoding
    X[col] = X[col].fillna('Unknown')  # Fill NaN with a placeholder before encoding
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le  # Save encoder for later use

print("✅ Categorical encoding complete!")

# -----------------------
# Handle Missing & Infinite Values
# -----------------------
print("\n🛠 Handling missing and infinite values...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN

# Drop columns that are entirely NaN
X = X.dropna(axis=1, how='all')
print(f"✅ Dropped all-NaN columns, remaining features: {X.shape[1]}")

# Fill NaN values with column median (better than filling with 0)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())


print("✅ Missing values handled!")

# -----------------------
# Split Data (Using Stratified Sampling)
# -----------------------
print("\n🔀 Splitting data into training and testing sets (Stratified)...")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("✅ Stratified data splitting complete!")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

# -----------------------
# Scale Features (After Splitting to Avoid Data Leakage)
# -----------------------
print("\n📊 Scaling features (MinMaxScaler)...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply same transformation to test set

print("✅ Feature scaling complete!")

# -----------------------
# Save Processed Data & Transformers
# -----------------------
output_dir = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed"
os.makedirs(output_dir, exist_ok=True)

# Save processed datasets
pd.DataFrame(X_train_scaled).to_csv(f"{output_dir}/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv(f"{output_dir}/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

# Save encoders & scaler for future use
with open(f"{output_dir}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open(f"{output_dir}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Data Preprocessing & Feature Scaling Complete!")
