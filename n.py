import pandas as pd

# Load selected features and target labels
X_train = pd.read_csv("data/processed/X_train_selected.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

X_test = pd.read_csv("data/processed/X_test_selected.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Check row alignment
print(f"✅ X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"✅ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Ensure no mismatches
assert X_train.shape[0] == y_train.shape[0], "❌ X_train and y_train row mismatch!"
assert X_test.shape[0] == y_test.shape[0], "❌ X_test and y_test row mismatch!"

print("✅ X and Y data are correctly aligned!")
