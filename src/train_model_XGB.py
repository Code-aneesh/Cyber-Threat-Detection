import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load Data
print("📥 Loading Data...")
X_train = pd.read_csv("data/processed/X_train_selected.csv")
y_train = pd.read_csv("data/processed/y_train.csv")["label"]

# 📊 Check Feature Correlation with Target
print("\n🔍 Checking for Data Leakage (Feature Correlation)...")
data_corr = X_train.copy()
data_corr["label"] = y_train  # Add target variable

# Compute correlation matrix
corr_matrix = data_corr.corr()

# Sort by correlation with target
target_corr = corr_matrix["label"].drop("label").sort_values(ascending=False)
print(target_corr)

# 🔥 Plot Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# ✅ Remove High-Correlation Features (Data Leakage Prevention)
# 🚨 Find actual feature names corresponding to indices
features_to_remove = [1, 5, 9]  # Indices from correlation output
feature_names = X_train.columns  # Get actual column names

# Get column names using indices
features_to_remove_names = [feature_names[i] for i in features_to_remove]

print(f"🚨 Removing Features: {features_to_remove_names}")  

# Drop features correctly
X_train = X_train.drop(columns=features_to_remove_names)


# ✅ Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(f"✅ Data Split: X_train: {X_train.shape}, X_test: {X_test.shape}")

# 🚀 Train Random Forest Model
print("\n🚀 Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
rf_model.fit(X_train, y_train)
print(f"✅ Random Forest Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
joblib.dump(rf_model, "models/random_forest_model.pkl")

# 🚀 Train Logistic Regression Model
print("\n🚀 Training Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy")
lr_model.fit(X_train, y_train)
print(f"✅ Logistic Regression Accuracy: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
joblib.dump(lr_model, "models/logistic_regression_model.pkl")

# 🚀 Train XGBoost Model
print("\n🚀 Training XGBoost Model...")
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="accuracy")
xgb_model.fit(X_train, y_train)
print(f"✅ XGBoost Accuracy: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
joblib.dump(xgb_model, "models/xgboost_model.pkl")

# ✅ Compare Model Performance
print("\n📊 Comparing the Models' Accuracy:")
print(f"Random Forest Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"Logistic Regression Accuracy: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
print(f"XGBoost Accuracy: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

best_model = max(
    [("Random Forest", rf_scores.mean()), 
     ("Logistic Regression", lr_scores.mean()), 
     ("XGBoost", xgb_scores.mean())], 
    key=lambda x: x[1]
)
print(f"\n🚀 Best Model: {best_model[0]} with Accuracy: {best_model[1]:.4f}")
