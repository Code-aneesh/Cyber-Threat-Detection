import pandas as pd

# Load the processed datasets
X_train = pd.read_csv("data/processed/X_train_selected.csv")
X_test = pd.read_csv("data/processed/X_test_selected.csv")

# ✅ Check for overlapping data
overlap = X_train.merge(X_test, how="inner")
print(f"🔍 Overlapping rows between X_train and X_test: {len(overlap)}")

if len(overlap) > 0:
    print("❌ Data leakage detected! Training and test sets have common samples.")
else:
    print("✅ No data leakage detected.")
