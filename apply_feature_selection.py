import os
import pandas as pd

# Define file paths
X_test_path = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed/X_test.csv"
X_train_selected_path = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed/X_train_selected.csv"
X_test_selected_path = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/processed/X_test_selected.csv"

try:
    # Load the test dataset
    print("📥 Loading X_test dataset...")
    X_test = pd.read_csv(X_test_path)
    print(f"✅ X_test loaded successfully! Shape: {X_test.shape}")

    # Load selected features from training set
    print("\n📥 Loading selected feature columns from X_train_selected...")
    selected_columns = pd.read_csv(X_train_selected_path).columns
    print(f"✅ Selected features loaded! Number of features: {len(selected_columns)}")

    # Apply feature selection
    print("\n🔄 Applying feature selection to X_test...")
    X_test_selected = X_test[selected_columns]
    print(f"✅ Feature selection applied! New shape: {X_test_selected.shape}")

    # Save the transformed test dataset
    os.makedirs(os.path.dirname(X_test_selected_path), exist_ok=True)
    X_test_selected.to_csv(X_test_selected_path, index=False)
    print(f"✅ X_test_selected saved successfully at {X_test_selected_path}")

except FileNotFoundError as e:
    print(f"❌ File Not Found: {e}")
except KeyError as e:
    print(f"❌ KeyError: Some selected features are missing in X_test -> {e}")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
